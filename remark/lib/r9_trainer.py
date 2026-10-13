"""ProtoQA: RankGRPOTrainer adapter for lists of rec_num = 10 free-text answers.

Overrides only what depends on the "Title (Year)" format or on Reddit-v2 responders:
- `_is_item`: a non-blank line is an item iff r9_common.parse_answer leaves an answer (no year regex). P0c line
  positions are forced (`line_positions="lines"`): every non-blank line takes the next rank position, lines that parse to
  nothing are zero-reward misses, the end bonus needs exactly rec_num lines, tokens after the rec_num-th line pay the
  overflow penalty (parent's `_segment_lines` / `_rank_advantages`, unchanged).
- `ITEM_LINE`: only used by the parent's "items" mode, which this class refuses; set to "has a letter or digit".
- `_rb_hits` (arm b, exact Rao-Blackwellization of arm a =  on ProtoQA): per-cluster first-hit vectors at line
  positions (lib/r9_rewards.line_cluster_hits). With rb_weights="respondents" (default) every respondent is a row of
  `_rb_rewards` (its cluster's vector) with mask 1, M = the largest respondent count of the training questions, so both
  the advantage (mean over respondents of the per-respondent group-scaled advantages) and the P0c shaping scale (1 /
  mean of the nonzero per-respondent group stds) follow the per-responder definitions exactly. rb_weights="clusters"
  puts one row per cluster with the cluster count as the mask value: identical advantages (rb_advantages divides by
  mask.sum()), but the shaping scale then averages stds over clusters, not respondents (shown in n928_r9_unit.py).
- `_calculate_rewards`: parent's, plus the reward function's per-batch stats (r9/...) in the logs.
"""

from __future__ import annotations

import re

import torch
from accelerate.utils import gather

from lib.r9_common import parse_answer
from lib.r9_rewards import line_cluster_hits
from lib.rank_grpo_trl import RankGRPOTrainer


class R9RankGRPOTrainer(RankGRPOTrainer):
    ITEM_LINE = re.compile(r".*?[^\W_]")  # unused: "items" mode is refused below

    def __init__(self, *args, rec_num: int = 10, line_positions: str = "lines", rb_weights: str = "respondents",
                 **kwargs):
        if line_positions != "lines":
            raise ValueError("ProtoQA uses P0c line positions only (line_positions='lines')")
        if rb_weights not in ("respondents", "clusters"):
            raise ValueError(rb_weights)
        super().__init__(*args, rec_num=rec_num, line_positions="lines", **kwargs)
        self.rb_weights = rb_weights
        if self.rb_table is not None:  # {prompt text: [cluster counts of each training board with that text]}
            boards = [c for v in self.rb_table.values() for c in v]  # (one training text has two boards)
            self.r9_M = max(sum(c) for c in boards) if rb_weights == "respondents" else max(len(c) for c in boards)

    @staticmethod
    def _is_item(line: str) -> bool:
        return bool(parse_answer(line))

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        out = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        stats = getattr(self.reward_funcs[0], "last_stats", None) or {}
        mode = "train" if self.model.training else "eval"
        for k, v in stats.items():
            self._metrics[mode][k].append(
                self.accelerator.gather(torch.tensor([float(v)], device=self.accelerator.device)).mean().item())
        return out

    def _rb_hits(self, prompts, completions, inputs, device) -> None:
        R, M = self.rec_num, self.r9_M
        hits = torch.zeros(len(prompts), M, R)
        mask = torch.zeros(len(prompts), M)
        for i, (p, c, ex) in enumerate(zip(prompts, completions, inputs)):
            text = c[0]["content"] if isinstance(c, list) else c
            counts = list(ex["cluster_counts"])
            key = p[0]["content"] if isinstance(p, list) else p  # as the parent's _rb_hits
            assert counts in self.rb_table[key], ex.get("qid")  # training labels of this prompt
            H = torch.as_tensor(line_cluster_hits(text, ex["all_clusters"], R), dtype=torch.float32)  # (C, R)
            if self.rb_weights == "respondents":
                idx = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
                hits[i, :len(idx)] = H[idx]
                mask[i, :len(idx)] = 1.0
            else:
                hits[i, :len(counts)] = H
                mask[i, :len(counts)] = torch.tensor(counts, dtype=torch.float32)
        self._rb_rewards = gather(hits.to(device))
        self._rb_mask = gather(mask.to(device))
