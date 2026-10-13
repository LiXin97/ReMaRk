"""Gradient-variance sketch (registered 2026-10-04 15:06 UTC; run last, no selection).

2B Reddit-v2 SFT @1500; 100 training requests with >= 2 responders (seed 42). Per request two independent groups of G = 8
lists (temperature 1) and a reference group of 32 lists. Gradient of the GRPO policy term (no KL, no clip), restricted to
the last decoder layer's parameters, projected by a fixed-seed count-sketch to 4,096 dims. Estimators per group: the
per-responder updates g_r (Rank-GRPO advantages, one per responder), their mean (exact RB), the union update (ReMaRk
advantages). Reference direction: the RB update on the 32-list group. Reported: (1) the share of the per-responder update
variance due to the responder draw, E_req E_group[sum_r ||g_r - mean_r g_r||^2 / n] / E_req E[||g_r - mean g_r over groups
and responders||^2]; (2) mean and median cosine of g_r, the RB mean and the union update to the reference direction, with
bootstrap CIs over requests.

Implementation (fixed before the run,  "implementation note"):
- requests: training rows of the per-row split grouped by prompt text (as the trainer's rb_table); the trainer's prompt
  filter (<= 2,048 prompt tokens) first, then >= 2 rows (responders), then 100 drawn with numpy default_rng(42) from the
  eligible prompts sorted by sha1; the union row (grpo_union) is matched by prompt text.
- sampling: vLLM, the SFT chat template (as TRL), temperature 1.0, top_p 1, max_tokens 320 (P0's cap), groups A / B / ref as
  three separate calls with seeds 1000*q + 1 / 2 / 3; completions cut after the first EOS (TRL's completion mask).
- advantages exactly as the trainer (P0c line positions): per responder r, Rank-GRPO's exp_inf hits against r's list
  realigned to line positions, (h - mean_G h) / (std_G h + 1e-4); RB = rb_advantages (responder mean of those); union =
  the consensus reward on the grpo_union row (weights 1), group-scaled. Responders with an empty list are kept (zero hits,
  as in training).
- gradient: d/dtheta sum_i sum_p A[i,p] sum_{t in line p of list i, t != EOS} log pi(y_t), theta = the last decoder layer;
  divided by the group's completion tokens (bnpo with the group as the batch). Length-shaping terms (end bonus, overflow,
  early stop) are excluded: they are not part of the advantage estimators compared here. By linearity, each list gets a
  sketched per-position basis (one autograd.grad per non-empty line position) and every estimator is a combination of it.
- count-sketch: per parameter element a bucket in [0, 4096) and a sign, torch.Generator seed 20261004.

Usage (node, /tmp/qenv): select (once) -> sample --shard k --nshards 4 -> grad --shard k --nshards 4 -> analyze (CPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "remark"))
REQ_DIR = REPO / "remark/outputs/crs/g2_gradsketch"  # requests.json (select) always lives here
OUT = Path(os.environ.get("G2_OUT", str(REQ_DIR)))  # smoke tests write elsewhere (G2_OUT)
FORMAL = OUT == REQ_DIR
SFT = REPO / "remark/outputs/crs/sft/qwen35_2B/checkpoint-1500"
SINGLE = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"
UNION = REPO / "remark/data/crs/grpo_union/train"
N_REQ, G, G_REF, R, DIM, MAX_TOK, MAX_PROMPT = 100, 8, 32, 20, 4096, 320, 2048
SKETCH_SEED = 20261004


def sha(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def cmd_select(a) -> None:
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SFT))
    single, union = load_from_disk(str(SINGLE)), load_from_disk(str(UNION))
    rows: dict[str, list[int]] = {}
    for i, pr in enumerate(single["prompt"]):
        rows.setdefault(pr[0]["content"], []).append(i)
    n_req = len(rows)

    def n_tok(c: str) -> int:
        return len(tok(tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                               add_generation_prompt=True))["input_ids"])
    multi = [c for c, ix in rows.items() if len(ix) >= 2]
    elig = sorted((c for c in multi if n_tok(c) <= MAX_PROMPT), key=sha)
    pick = np.random.default_rng(42).choice(len(elig), N_REQ, replace=False)
    uidx = {pr[0]["content"]: i for i, pr in enumerate(union["prompt"])}
    reqs = []
    for q, j in enumerate(sorted(int(x) for x in pick)):
        c = elig[j]
        ex = [single[i] for i in rows[c]]
        u = union[uidx[c]]
        reqs.append({"q": q, "sha1": sha(c), "prompt": c, "seen": ex[0]["seen_titles"],
                     "responders": [e["groundtruth_with_release_year"] for e in ex],
                     "union_gt": u["groundtruth_with_release_year"], "union_w": u["groundtruth_weights"]})
    assert FORMAL, "select writes the registered request draw"
    OUT.mkdir(parents=True, exist_ok=True)
    n_resp = [len(r["responders"]) for r in reqs]
    empty = sum(1 for r in reqs for g in r["responders"] if not g)
    man = {"requests_total": n_req, "with_ge2_responders": len(multi), "eligible_after_prompt_filter": len(elig),
           "drawn": N_REQ, "responders_mean": float(np.mean(n_resp)), "responders_min": min(n_resp),
           "responders_max": max(n_resp), "empty_responder_lists": empty, "responder_lists": sum(n_resp),
           "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "sft": str(SFT), "single": str(SINGLE), "union": str(UNION)}
    (OUT / "requests.json").write_text(json.dumps(reqs))
    (OUT / "select_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


def load_reqs(a) -> list[dict]:
    reqs = json.loads((REQ_DIR / "requests.json").read_text())
    reqs = [r for r in reqs if r["q"] % a.nshards == a.shard]
    return reqs[:a.limit] if a.limit else reqs


def cmd_sample(a) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path.insert(0, str(REPO / "remark/scripts"))
    from crs_eval import prepare_qwen35
    tok = AutoTokenizer.from_pretrained(str(SFT))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"samples_{a.shard}.jsonl"
    done = {json.loads(l)["q"] for l in out.open()} if out.exists() else set()
    reqs = [r for r in load_reqs(a) if r["q"] not in done]
    if not reqs:
        return
    llm = LLM(model=str(SFT), gpu_memory_utilization=0.5, max_model_len=MAX_PROMPT + MAX_TOK + 64,
              seed=a.shard, **prepare_qwen35(SFT))
    eos = tok.eos_token_id
    for r in reqs:
        ids = tok(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False,
                                          add_generation_prompt=True), add_special_tokens=False)["input_ids"]
        rec = {"q": r["q"], "prompt_ids": ids}
        for name, n, s in (("A", G, 1), ("B", G, 2), ("ref", G_REF, 3)):
            sp = SamplingParams(n=n, temperature=1.0, top_p=1.0, max_tokens=MAX_TOK, seed=1000 * r["q"] + s)
            o = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
            comps = []
            for c in o.outputs:
                t = list(c.token_ids)
                if eos in t:
                    t = t[:t.index(eos) + 1]
                comps.append(t)
            rec[name] = comps
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[sample] q={r['q']}", flush=True)


class Shaper:
    """The trainer's line-position machinery (rank_grpo_trl.RankGRPOTrainer, P0c) without building a trainer."""

    def __init__(self, tok, n_vocab: int):
        import torch
        from lib.rank_grpo_trl import RankGRPOTrainer
        t = object.__new__(RankGRPOTrainer)
        t.rec_num, t.line_positions = R, "lines"
        pieces = [tok.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tok))]
        t._piece = pieces + [""] * (n_vocab - len(tok))
        t._nl_count = torch.tensor([p.count("\n") for p in t._piece], dtype=torch.long)
        txt = [tok.decode([i], skip_special_tokens=True, clean_up_tokenization_spaces=False) for i in range(len(tok))]
        t._piece_text = txt + [""] * (n_vocab - len(tok))
        self.t = t

    def segment(self, comp_ids: list[int]):
        import torch
        ids = torch.tensor([comp_ids])
        seg, _ = self.t._segment(ids, torch.ones_like(ids))
        return seg[0]

    def to_lines(self, text: str, per_item: list) -> list:
        return self.t._to_line_positions(text, per_item)


def group_scale(h: np.ndarray) -> np.ndarray:
    """(n, R) -> (h - mean) / (std + 1e-4) over the group (torch's unbiased std, as the trainer)."""
    return (h - h.mean(0, keepdims=True)) / (h.std(0, ddof=1, keepdims=True) + 1e-4)


def cmd_grad(a) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from lib.crs_rewards import RG, make_consensus_reward, make_reward_func_individual
    sys.path.insert(0, str(RG))
    from libs.data import load_catalog
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(str(SFT))
    model = AutoModelForImageTextToText.from_pretrained(str(SFT), dtype=torch.bfloat16).to(dev)
    model.eval()
    layers = model.model.language_model.layers
    last = len(layers) - 1
    for p in model.parameters():
        p.requires_grad_(False)
    params = [(n, p) for n, p in layers[last].named_parameters()]
    for _, p in params:
        p.requires_grad_(True)
    D = sum(p.numel() for _, p in params)
    gen = torch.Generator(device="cpu").manual_seed(SKETCH_SEED)
    bucket = torch.randint(0, DIM, (D,), generator=gen, dtype=torch.int64).to(dev)
    sign = (torch.randint(0, 2, (D,), generator=gen).float() * 2 - 1).to(dev)
    pinfo = {"layer_index": last, "layer_type": type(layers[last]).__name__, "n_params": D,
             "names": [f"{n} {tuple(p.shape)}" for n, p in params]}
    (OUT / "grad_params.json").write_text(json.dumps(pinfo, indent=1))
    n_vocab = model.lm_head.weight.shape[0]
    sh = Shaper(tok, n_vocab)
    catalog = load_catalog(str(RG / "reddit_catalog/gt_catalog.pkl"))
    indiv = make_reward_func_individual(rec_num=R, gt_catalog=catalog)
    cons = make_consensus_reward(catalog)
    reqs = {r["q"]: r for r in load_reqs(a)}
    samples = [json.loads(l) for l in (OUT / f"samples_{a.shard}.jsonl").open()]
    vdir = OUT / "vectors"
    vdir.mkdir(parents=True, exist_ok=True)

    def basis(prompt_ids: list[int], comp: list[int]) -> torch.Tensor:
        """(R, DIM) sketched gradient of each line position's summed log-prob (EOS excluded)."""
        seg = sh.segment(comp).to(dev)
        is_eos = torch.tensor([t == tok.eos_token_id for t in comp], device=dev)
        out = torch.zeros(R, DIM, device=dev)
        live = [p for p in range(R) if bool(((seg == p) & ~is_eos).any())]
        if not live:
            return out
        ids = torch.tensor([prompt_ids + comp], device=dev)
        logits = model(input_ids=ids, logits_to_keep=len(comp) + 1).logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(comp, device=dev)[:, None])[:, 0]
        for k, p in enumerate(live):
            m = (seg == p) & ~is_eos
            gs = torch.autograd.grad(lp[m].sum(), [q for _, q in params], retain_graph=k < len(live) - 1)
            flat = torch.cat([g.reshape(-1).float() for g in gs])
            out[p].index_add_(0, bucket, flat * sign)
        return out

    def rewards(r: dict, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """per-responder hits (n, lists, R) and union rewards (lists, R), on line positions."""
        comps = [[{"role": "assistant", "content": t}] for t in texts]
        hr = []
        for gts in r["responders"]:
            out = indiv(completions=comps, groundtruth_with_release_year=[gts] * len(texts),
                        seen_titles=[r["seen"]] * len(texts))
            hr.append([sh.to_lines(t, o) for t, o in zip(texts, out)])
        u = cons(completions=comps, groundtruth_with_release_year=[r["union_gt"]] * len(texts),
                 groundtruth_weights=[r["union_w"]] * len(texts), seen_titles=[r["seen"]] * len(texts))
        return np.array(hr, dtype=np.float64), np.array([sh.to_lines(t, o) for t, o in zip(texts, u)], dtype=np.float64)

    for s in samples:
        q = s["q"]
        f = vdir / f"q{q:03d}.npz"
        if q not in reqs or f.exists():
            continue
        r, res = reqs[q], {}
        for name in ("A", "B", "ref"):
            comps = s[name]
            texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
            hr, hu = rewards(r, texts)
            B = torch.stack([basis(s["prompt_ids"], c) for c in comps]).double().cpu().numpy()  # (lists, R, DIM)
            ntok = sum(len(c) for c in comps)
            adv_r = np.stack([group_scale(h) for h in hr])  # (n, lists, R)
            g_r = np.einsum("nlr,lrd->nd", adv_r, B) / ntok
            g_rb = np.einsum("lr,lrd->d", adv_r.mean(0), B) / ntok
            g_u = np.einsum("lr,lrd->d", group_scale(hu), B) / ntok
            res[name] = {"g_r": g_r, "g_rb": g_rb, "g_u": g_u, "hits_r": hr.sum(), "hits_u": hu.sum(), "ntok": ntok,
                         "zero_std_r": int(sum(1 for h in hr if np.all(h.std(0) == 0)))}
        np.savez(f, **{f"{k}_{n}": v for n, d in res.items() for k, v in d.items()})
        print(f"[grad] q={q} n={len(r['responders'])} hitsA={res['A']['hits_r']:.0f} uA={res['A']['hits_u']:.0f}", flush=True)


def cos(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else float("nan")


def cmd_analyze(a) -> None:
    reqs = json.loads((REQ_DIR / "requests.json").read_text())
    per = []
    for r in reqs:
        f = OUT / "vectors" / f"q{r['q']:03d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        ref = z["g_rb_ref"]
        gr = {k: z[f"g_r_{k}"] for k in ("A", "B")}  # (n, DIM) each
        n = gr["A"].shape[0]
        within = np.mean([np.sum((gr[k] - gr[k].mean(0)) ** 2) / n for k in ("A", "B")])
        allv = np.concatenate([gr["A"], gr["B"]])
        total = np.sum((allv - allv.mean(0)) ** 2) / len(allv)
        cr = [cos(v, ref) for k in ("A", "B") for v in gr[k]]
        cr = [c for c in cr if not np.isnan(c)]
        per.append({"q": r["q"], "n": n, "within": float(within), "total": float(total),
                    "cos_r": float(np.mean(cr)) if cr else float("nan"), "cos_r_defined": len(cr), "cos_r_all": 2 * n,
                    "cos_rb": float(np.nanmean([cos(z[f"g_rb_{k}"], ref) for k in ("A", "B")])),
                    "cos_u": float(np.nanmean([cos(z[f"g_u_{k}"], ref) for k in ("A", "B")])),
                    "ref_zero": bool(np.linalg.norm(ref) == 0)})
    rng = np.random.default_rng(42)
    m = len(per)
    W, T = np.array([p["within"] for p in per]), np.array([p["total"] for p in per])
    idx = rng.integers(0, m, (2000, m))
    share = W.sum() / T.sum()
    bs = W[idx].sum(1) / T[idx].sum(1)
    res = {"requests": m, "share_responder_draw": {"est": float(share), "ci": [float(np.percentile(bs, 2.5)),
                                                                            float(np.percentile(bs, 97.5))]}}
    for key in ("cos_r", "cos_rb", "cos_u"):
        v = np.array([p[key] for p in per])
        ok = ~np.isnan(v)
        vv = v[ok]
        b = vv[np.random.default_rng(42).integers(0, len(vv), (2000, len(vv)))]
        res[key] = {"requests": int(ok.sum()), "mean": float(vv.mean()), "median": float(np.median(vv)),
                    "mean_ci": [float(np.percentile(b.mean(1), 2.5)), float(np.percentile(b.mean(1), 97.5))],
                    "median_ci": [float(np.percentile(np.median(b, 1), 2.5)), float(np.percentile(np.median(b, 1), 97.5))]}
    res["ref_zero_requests"] = sum(p["ref_zero"] for p in per)
    res["g_r_cosine_defined_share"] = sum(p["cos_r_defined"] for p in per) / max(1, sum(p["cos_r_all"] for p in per))
    res["per_request"] = per
    dst = REPO / "remark/outputs/crs/analysis/n931_g2_gradsketch.json" if FORMAL else OUT / "analysis.json"
    dst.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_request"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "sample", "grad", "analyze"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first k requests of the shard")
    a = ap.parse_args()
    {"select": cmd_select, "sample": cmd_sample, "grad": cmd_grad, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
