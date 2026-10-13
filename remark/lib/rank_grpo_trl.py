"""Rank-GRPO (Zhu et al., ICLR 2026) on TRL 1.14's GRPOTrainer.

The reward function returns one reward per rank position, shape (N, rec_num). Advantages are computed per position
(group mean baseline; std from the group or from the whole batch, see `adv_norm`) and broadcast to the tokens of the
item at that position. Items are the lines that Rank-GRPO's parser accepts ("Title (YYYY)"): the completion is split
at newline tokens (a newline belongs to the line it ends), lines are numbered by their order among item lines, and
tokens of other lines (blank lines, a stray "</think>" prefix) get no position advantage. For well-formed lists this
is identical to the released trainer's newline counting; the released version shifts every position after a blank line
(observed in a 4B run that learned to prefix "\n\n</think>\n\n"). Length
shaping follows Rank-GRPO: +end_of_list_reward on EOS when the list has exactly rec_num items, early_stop_penalty on
EOS when it has fewer, extra_token_penalty on every token after the rec_num-th item.

Deviation from the released trainer: importance weights are token-level instead of item-level. Training is on-policy
(num_iterations=1, one optimizer step per generation batch), so the ratio is 1 and both give the same gradient.

options (both off by default; the default path is unchanged):
- rb_table (R2, exact Rao-Blackwellization of Rank-GRPO's estimator): {request prompt text: [ground-truth list of each
  responder]}. The per-position advantage of a list becomes the mean over the request's responders of the group-scaled
  advantage under each responder's own 0/1 hits, (1/n_x) sum_r (h_r - mean_G h_r) / (std_G h_r + 1e-4) (rb_advantages).
- dyn_factor (R5, DAPO-style dynamic sampling): each process generates for dyn_factor x its prompts (the extra rows drawn
  uniformly from the training rows, seeded per process) and keeps as many groups as Rank-GRPO uses: groups whose lists
  carry some reward signal first (sampling order), then zero-signal groups only to fill (dyn_select).

E4 option (10-07, off by default): seq_level = ordinary sequence-level GRPO. Each list gets one scalar reward, its
recall@20 (sum of the per-position rewards over the first rec_num positions / number of ground-truth items of the row:
one responder's items for the per-responder reward, the union of the responders' items for the union reward); the
group-scaled advantage (r - mean_G r) / (std_G r + 1e-4) is applied to every completion token (no rank positions).
The length shaping (end bonus, early stop, overflow) is added per token exactly as in the default path, with the
shaping scale computed from the per-position rewards as there, so length control is identical across the two paths.

E1 isolation options (10-07, off by default): adv_norm = "fixed" divides the group-mean-centred per-position advantages by
the constant adv_scale instead of a group or batch std (common advantage scale across reward variants; with P0 the
shaping scale is then 1 / (adv_scale + 1e-4)); shaping_scale_const replaces the reward-unit shaping scale by a constant.
"""

from __future__ import annotations

import re

import numpy as np
import torch
from accelerate.utils import gather
from trl import GRPOTrainer


def group_advantages(rewards: torch.Tensor, G: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-GRPO's per-group scaling. rewards (N, R) -> advantages (N, R), std (N, R)."""
    R = rewards.size(-1)
    grouped = rewards.view(-1, G, R)
    adv = (grouped - grouped.mean(dim=1, keepdim=True)).view(-1, R)
    std = grouped.std(dim=1, keepdim=True).expand(-1, G, -1).reshape(-1, R)
    return adv / (std + 1e-4), std


def rb_advantages(rb: torch.Tensor, mask: torch.Tensor, G: int) -> tuple[torch.Tensor, torch.Tensor]:
    """. rb (N, M, R): hits of every list against each of its request's responders (padded to M); mask (N, M):
    1 for a real responder. Returns the responder mean of the per-responder group-scaled advantages (N, R) and the
    per-responder group stds of real responders (flat, for the shaping scale)."""
    N, M, R = rb.shape
    g = rb.view(-1, G, M, R)
    mean = g.mean(dim=1, keepdim=True)
    std = g.std(dim=1, keepdim=True)
    a = (g - mean) / (std + 1e-4)
    m = mask.view(-1, G, M)[:, 0, :].float()  # the same responders for every list of a group
    n = m.sum(dim=1).clamp(min=1.0)
    adv = (a * m[:, None, :, None]).sum(dim=2) / n[:, None, None]
    used = std.squeeze(1)[m.bool()]  # (real responders, R)
    return adv.reshape(N, R), used.reshape(-1)


def dyn_select(rewards: torch.Tensor, G: int, n_proc: int, keep_groups: int) -> tuple[torch.Tensor, dict]:
    """. rewards (N_total, R) over all processes, each process holding a contiguous block of whole groups.
    Per process keep `keep_groups` groups: those with some reward signal (nonzero std over the G lists at some position)
    in sampling order, then zero-signal groups in sampling order if needed. Returns the kept global row indices (process
    blocks in order, rows of a kept group in order) and stats."""
    R = rewards.size(-1)
    grouped = rewards.view(n_proc, -1, G, R)
    n_groups = grouped.size(1)
    signal = grouped.std(dim=2).sum(dim=-1) > 0  # (P, groups)
    rows, kept_signal = [], 0
    for p in range(n_proc):
        sig = [i for i in range(n_groups) if bool(signal[p, i])]
        zero = [i for i in range(n_groups) if not bool(signal[p, i])]
        keep = sorted((sig + zero)[:keep_groups])
        kept_signal += min(len(sig), keep_groups)
        for gi in keep:
            start = (p * n_groups + gi) * G
            rows.extend(range(start, start + G))
    stats = {"dyn/generated_signal_share": float(signal.float().mean()),
             "dyn/kept_signal_share": kept_signal / (n_proc * keep_groups)}
    return torch.tensor(rows, device=rewards.device, dtype=torch.long), stats


class RankGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, rec_num: int = 20, adv_norm: str = "group", end_of_list_reward: float = 0.1,
                 extra_token_penalty: float = -0.1, early_stop_penalty: float = -0.1, shaping_units: str = "fixed",
                 line_positions: str = "items", exact_requires_items: bool = False, rb_table: dict | None = None,
                 rb_catalog=None, dyn_factor: int = 1, seq_level: bool = False, adv_scale: float | None = None,
                 shaping_scale_const: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_level = bool(seq_level)  # E4
        self._seq_ngt = None
        if self.seq_level and (rb_table is not None or int(dyn_factor) > 1 or adv_norm != "group"):
            raise ValueError("seq_level is defined with adv_norm='group' and without rb_table / dyn_factor")
        self.rb_table = rb_table
        self.rb_max = max((len(v) for v in rb_table.values()), default=1) if rb_table else 0
        self._rb_rewards = self._rb_mask = None
        self._rb_catalog = rb_catalog
        if rb_table is not None and adv_norm != "group":
            raise ValueError("rb_table needs adv_norm='group'")
        self.dyn_factor = int(dyn_factor)
        self._dyn_rng = np.random.default_rng(int(self.args.seed) * 1000 + 17 + self.accelerator.process_index)
        if adv_norm not in ("group", "batch", "fixed"):
            raise ValueError(f"adv_norm must be 'group', 'batch' or 'fixed', got {adv_norm}")
        if (adv_norm == "fixed") != (adv_scale is not None):
            raise ValueError("adv_scale is required with, and only with, adv_norm='fixed'")
        self.adv_scale, self.shaping_scale_const = adv_scale, shaping_scale_const  # E1 isolation
        if len(self.reward_funcs) != 1:
            raise ValueError("RankGRPOTrainer expects exactly one per-position reward function")
        self.rec_num, self.adv_norm = rec_num, adv_norm
        if shaping_units not in ("fixed", "reward"):
            raise ValueError(f"shaping_units must be 'fixed' or 'reward', got {shaping_units}")
        # fixed: Rank-GRPO's constants added to normalized advantages; reward (P0): the constants are in reward
        # units and divided by the same scale as the item rewards, so their weight does not shrink with the reward std
        self.shaping_units = shaping_units
        if line_positions not in ("items", "lines"):
            raise ValueError(f"line_positions must be 'items' or 'lines', got {line_positions}")
        # items: rank positions go to parser-valid item lines only (malformed lines get no advantage);
        # lines (P0c): every non-blank line takes a position and malformed lines are zero-reward misses, as in the
        # released trainer; rewards (computed per valid item) are re-aligned to line positions before the gather
        self.line_positions = line_positions
        # (Reddit-Books): with line positions, a one-word 20th line still counted as "exactly rec_num items" and
        # earned the end bonus; with exact_requires_items the bonus needs rec_num parser-valid items among the first
        # rec_num lines, and a terminated list with fewer valid items pays the early-stop penalty instead
        self.exact_requires_items = exact_requires_items
        self._valid_local = None
        self.end_of_list_reward, self.extra_token_penalty = end_of_list_reward, extra_token_penalty
        self.early_stop_penalty = early_stop_penalty
        tok = self._tokenizer
        # number of "\n" in each vocabulary entry's decoded text (byte-level BPE merges newlines into other tokens);
        # the embedding table is larger than the tokenizer (Qwen3.5: 248,320 vs 248,077) and vLLM can sample the
        # unused ids, which decode to nothing
        pieces = [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tok))]
        n_vocab = max(len(tok), self.model.config.get_text_config().vocab_size)
        self._piece = pieces + [""] * (n_vocab - len(tok))
        self._nl_count = torch.tensor([p.count("\n") for p in self._piece], dtype=torch.long)
        if line_positions == "lines":  # text of each token without special tokens (blank-line test)
            txt = [tok.decode([i], skip_special_tokens=True, clean_up_tokenization_spaces=False) for i in range(len(tok))]
            self._piece_text = txt + [""] * (n_vocab - len(tok))
        self._rank_rewards = None

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        keys = [k for k in inputs[0] if k not in ("prompt", "completion", "completion_ids")]
        reward_kwargs = {k: [ex[k] for ex in inputs] for k in keys}
        out = self.reward_funcs[0](prompts=prompts, completions=completions, completion_ids=completion_ids_list,
                                   **reward_kwargs)
        if self.line_positions == "lines":
            texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
            out = [self._to_line_positions(t, r) for t, r in zip(texts, out)]
            first = [[y for y in t.split("\n") if not self._blank(y)][:self.rec_num] for t in texts]
            bad = [sum(1 for x in f if not self._is_item(x)) / self.rec_num for f in first]
            self._valid_local = torch.tensor([sum(1 for x in f if self._is_item(x)) for f in first], device=device)
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["rank/malformed_line_frac"].append(
                self.accelerator.gather(torch.tensor([sum(bad) / max(len(bad), 1)], device=device)).mean().item())
        if self.rb_table is not None and self.model.training:
            self._rb_hits(prompts, completions, inputs, device)
        per_pos = torch.tensor(out, dtype=torch.float32, device=device) * float(self.reward_weights[0])
        if per_pos.shape != (len(prompts), self.rec_num):
            raise ValueError(f"reward shape {tuple(per_pos.shape)} != ({len(prompts)}, {self.rec_num})")
        self._rank_rewards = gather(per_pos)  # (N_total, rec_num)
        if self.seq_level:  # E4: recall denominator = ground-truth items of each row
            ngt = torch.tensor([float(len(g)) for g in reward_kwargs["groundtruth_with_release_year"]], device=device)
            self._seq_ngt = gather(ngt)
        return self._rank_rewards.sum(dim=1, keepdim=True)  # list-level reward for TRL's own logging

    def _rb_hits(self, prompts, completions, inputs, device) -> None:
        """0/1 hits of each completion against every responder of its request, with Rank-GRPO's own parser
        and matcher (parsed once per completion), realigned to rank positions exactly as the per-responder reward."""
        from lib.crs_rewards import RG  # noqa: F401  (puts Rank-GRPO's libs on the path)
        from libs.metrics_align import evaluate_direct_match_aligned
        from libs.utils import process_rec_raw
        R, M = self.rec_num, self.rb_max
        catalog = self._rb_catalog
        hits = torch.zeros(len(prompts), M, R)
        mask = torch.zeros(len(prompts), M)
        for i, (p, c, ex) in enumerate(zip(prompts, completions, inputs)):
            text = c[0]["content"] if isinstance(c, list) else c
            gts_all = self.rb_table[p[0]["content"] if isinstance(p, list) else p]
            mask[i, :len(gts_all)] = 1.0
            err, item = process_rec_raw({"raw_recs": text, "seen_titles": ex["seen_titles"]}, "raw_recs", "recs")
            if err:
                continue
            lmap = self._line_map(text) if self.line_positions == "lines" else None
            for j, gts in enumerate(gts_all):
                item["groundtruth_with_release_year"] = gts
                h = evaluate_direct_match_aligned(item=item, rec_num=R, seen_field="seen_titles", rec_field="recs",
                                                  gt_field="groundtruth_with_release_year", gt_catalog=catalog,
                                                  title_normalizer=None, year_tolerance=2).astype(np.float64)
                if lmap is not None:
                    h = [float(h[q]) if 0 <= q < len(h) else 0.0 for q in lmap]
                hits[i, j] = torch.as_tensor(h, dtype=torch.float32)
        self._rb_rewards = gather(hits.to(device))
        self._rb_mask = gather(mask.to(device))

    def _line_map(self, text: str) -> list:
        """For each of the first rec_num non-blank lines: the index of its item among the parser-valid lines, or -1
        (the mapping _to_line_positions applies)."""
        out, j = [-1] * self.rec_num, 0
        for k, ln in enumerate([x for x in text.split("\n") if not self._blank(x)][:self.rec_num]):
            if self._is_item(ln):
                out[k] = j
                j += 1
        return out

    ITEM_LINE = re.compile(r"(.+?)\s+\((\d{4})\)")  # Rank-GRPO process_rec_raw's item pattern
    THINK_TAGS = ("<think>", "</think>")

    @classmethod
    def _blank(cls, line: str) -> bool:
        """P0c: a blank line, or one holding only a think tag (the 35B-A3B SFT writes "</think>" before every
        list), takes no rank position."""
        t = line.strip()
        return not t or t in cls.THINK_TAGS

    @staticmethod
    def _is_item(line: str) -> bool:
        """Exactly the Rank-GRPO parser's decision for one line."""
        from lib.crs_rewards import process_rec_raw
        err, item = process_rec_raw({"raw": line}, "raw", "recs")
        return (not err) and len(item["recs"]) == 1

    def _to_line_positions(self, text: str, per_item: list) -> list:
        """Re-align a reward vector indexed by valid item (parser order) to non-blank line positions."""
        R = self.rec_num
        out, j = [0.0] * R, 0
        for k, ln in enumerate([x for x in text.split("\n") if not self._blank(x)][:R]):
            if self._is_item(ln):
                out[k] = float(per_item[j]) if j < len(per_item) else 0.0
                j += 1
        return out

    def _segment(self, completion_ids, completion_mask):
        """Per token: rank position of the item line it belongs to (0-based), -1 for tokens of non-item lines before
        the rec_num-th item, rec_num for every token after it (overflow). Also the number of item lines per row."""
        R = self.rec_num
        nl = self._nl_count.to(completion_ids.device)[completion_ids] * completion_mask.long()
        line = (torch.cumsum(nl, dim=1) - nl).tolist()  # raw line index per token
        ids, lens = completion_ids.tolist(), completion_mask.sum(dim=1).tolist()
        seg = torch.full_like(completion_ids, -1)
        items = []
        if self.line_positions == "lines":
            return self._segment_lines(line, ids, lens, seg, completion_ids.device)
        for b in range(len(ids)):
            L = int(lens[b])
            text: dict[int, list[str]] = {}
            for t in range(L):
                text.setdefault(line[b][t], []).append(self._piece[ids[b][t]])
            pos, k, done_at = {}, 0, None
            for ln in sorted(text):
                if done_at is not None:
                    pos[ln] = R
                elif self.ITEM_LINE.match("".join(text[ln]).strip()):
                    pos[ln] = k
                    k += 1
                    if k == R:
                        done_at = ln
                else:
                    pos[ln] = -1
            seg[b, :L] = torch.tensor([pos[line[b][t]] for t in range(L)], dtype=seg.dtype)
            items.append(k if done_at is None else R + sum(
                1 for ln in text if ln > done_at and self.ITEM_LINE.match("".join(text[ln]).strip())))
        return seg, torch.tensor(items, device=completion_ids.device)

    def _segment_lines(self, line, ids, lens, seg, device):
        """P0c: every non-blank line takes the next rank position; tokens after the rec_num-th line are overflow."""
        R = self.rec_num
        n_lines = []
        for b in range(len(ids)):
            L = int(lens[b])
            text: dict[int, list[str]] = {}
            for t in range(L):
                text.setdefault(line[b][t], []).append(self._piece_text[ids[b][t]])
            pos, k, done_at = {}, 0, None
            for ln in sorted(text):
                if done_at is not None:
                    pos[ln] = R
                elif not self._blank("".join(text[ln])):
                    pos[ln] = k
                    k += 1
                    if k == R:
                        done_at = ln
                else:
                    pos[ln] = -1
            seg[b, :L] = torch.tensor([pos[line[b][t]] for t in range(L)], dtype=seg.dtype)
            extra = sum(1 for ln in text if done_at is not None and ln > done_at and not self._blank("".join(text[ln])))
            n_lines.append(k + extra)
        return seg, torch.tensor(n_lines, device=device)

    def _generate_and_score_completions(self, inputs):
        mode = "train" if self.model.training else "eval"
        G = self.num_generations if mode == "train" else self.num_generations_eval
        dyn = self.dyn_factor > 1 and mode == "train"
        n_local = len(inputs)
        if dyn:  # (dyn_factor - 1) x more whole groups, rows drawn uniformly from the training rows
            extra = self._dyn_rng.integers(0, len(self.train_dataset), (self.dyn_factor - 1) * n_local // G)
            inputs = list(inputs) + [dict(self.train_dataset[int(i)]) for i in extra for _ in range(G)]
        output = super()._generate_and_score_completions(inputs)
        rewards, dstats = self._rank_rewards, {}
        if dyn:
            n_exp = len(inputs)
            keep, dstats = dyn_select(rewards, G, self.accelerator.num_processes, n_local // G)
            p = self.accelerator.process_index
            local = keep[(keep >= p * n_exp) & (keep < (p + 1) * n_exp)] - p * n_exp
            rewards = rewards[keep]
            for k, v in list(output.items()):
                if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == n_exp:
                    output[k] = v[local]
            if self._valid_local is not None:
                self._valid_local = self._valid_local[local]
            output["num_items_in_batch"] = self.accelerator.gather(output["completion_mask"].sum()).sum()
        output["advantages"], stats = self._rank_advantages(
            rewards, output["completion_ids"], output["completion_mask"], G, self.accelerator.process_index)
        stats.update({k: torch.tensor([v], device=self.accelerator.device) for k, v in dstats.items()})
        for k, v in stats.items():
            self._metrics[mode][k].append(self.accelerator.gather(v).mean().item())
        return output

    def _rank_advantages(self, rewards, completion_ids, completion_mask, G, process_index):
        """rewards: (N_total, R) over all processes; returns local per-token advantages (N_local, T) and stats."""
        R = self.rec_num
        grouped = rewards.view(-1, G, R)
        rb = self.rb_table is not None and self._rb_rewards is not None and self.model.training
        if rb:  # responder mean of per-responder group-scaled advantages; shaping scale from their stds
            adv, std = rb_advantages(self._rb_rewards, self._rb_mask, G)
            self._rb_rewards = None
        else:
            adv = (grouped - grouped.mean(dim=1, keepdim=True)).view(-1, R)
            if self.adv_norm == "group":
                std = grouped.std(dim=1, keepdim=True).expand(-1, G, -1).reshape(-1, R)
            elif self.adv_norm == "fixed":  # E1 isolation: one constant scale for every position, group and reward variant
                std = torch.full_like(rewards, float(self.adv_scale))
            else:
                std = rewards.std(dim=0, keepdim=True).expand_as(rewards)
            adv = adv / (std + 1e-4)
        if self.seq_level:  # E4: one group-scaled scalar advantage per list (TRL's training mode only)
            rec = rewards.sum(dim=1) / self._seq_ngt.clamp(min=1.0)
            rg = rec.view(-1, G)
            seq_adv = ((rg - rg.mean(dim=1, keepdim=True)) / (rg.std(dim=1, keepdim=True) + 1e-4)).view(-1)
        if self.shaping_units == "reward":
            # mean per-position scale actually used for the items (group: over groups with a nonzero std)
            used = std[std > 1e-6]
            shaping_scale = 1.0 / (float(used.mean()) + 1e-4) if used.numel() else 1.0
        else:
            shaping_scale = 1.0
        if self.shaping_scale_const is not None:  # E1 isolation (a)
            shaping_scale = float(self.shaping_scale_const)

        n_local = completion_ids.size(0)
        adv_local = adv[process_index * n_local:(process_index + 1) * n_local]

        seg, items_emitted = self._segment(completion_ids, completion_mask)
        valid = completion_mask.bool()
        content = valid & (seg >= 0) & (seg < R)
        overflow = valid & (seg >= R)
        last = (completion_mask.sum(dim=1) - 1).clamp(min=0)
        pos = torch.arange(completion_ids.size(1), device=completion_ids.device).unsqueeze(0)
        eos_mask = (pos == last.unsqueeze(1)) & valid & (completion_ids == self._tokenizer.eos_token_id)
        terminated = eos_mask.any(dim=1)

        adv_tok = adv_local.gather(1, seg.clamp(min=0, max=R - 1)) * content.float() * (~eos_mask).float()
        if self.seq_level:  # E4: the list's scalar advantage on every completion token instead of the per-position one
            adv_tok = seq_adv[process_index * n_local:(process_index + 1) * n_local].unsqueeze(1) * valid.float()
        has_overflow = overflow.any(dim=1)
        exact = (items_emitted >= R) & ~has_overflow & terminated
        early = (items_emitted < R) & terminated
        valid_local = getattr(self, "_valid_local", None)
        if getattr(self, "exact_requires_items", False) and valid_local is not None:
            full = valid_local.to(exact.device) >= R
            early = early | (exact & ~full)
            exact = exact & full
        adv_tok = adv_tok + shaping_scale * self.extra_token_penalty * overflow.float()
        adv_tok = adv_tok + shaping_scale * self.end_of_list_reward * (exact.float().unsqueeze(1) * eos_mask.float())
        adv_tok = adv_tok + shaping_scale * self.early_stop_penalty * (early.float().unsqueeze(1) * eos_mask.float())
        grp_std = grouped.std(dim=1)  # (B, R)
        dev = adv_tok.device
        stats = {
            "rank/frac_group_zero_std": (grp_std.sum(dim=1) == 0).float().mean().reshape(1),
            "rank/hits_per_list": rewards.sum(dim=1).mean().reshape(1),
            "rank/reward_pos1": rewards[:, 0].mean().reshape(1),
            "rank/reward_pos10": rewards[:, min(9, R - 1)].mean().reshape(1),
            "rank/items_emitted": items_emitted.float().mean().reshape(1),
            "rank/frac_exact_len": exact.float().mean().reshape(1),
            "rank/frac_terminated": terminated.float().mean().reshape(1),
            "rank/shaping_scale": torch.tensor([shaping_scale]),
        }
        if self.seq_level:
            stats["seq/recall"] = rec.mean().reshape(1)
            stats["seq/frac_group_zero_std"] = (rg.std(dim=1) == 0).float().mean().reshape(1)
        return adv_tok, {k: v.to(dev) for k, v in stats.items()}
