"""Reward-noise diagnostic, generalized: any dataset in our format and any SFT policy.

Identical protocol to crs_consensus_check.py: 500 train-split contexts with >= 2 responders (seed 42), 8 lists
per context at temperature 1, list-level NDCG@20 against each responder's items (S) and against all responders with
fractional weights (C). Reports (a) the std of a fixed list's score across the responders (cross_row_noise_S) vs the
std across the 8 lists (within_group_std_S), (b) the share of groups with all 8 lists at zero under S vs under C,
(c) the mean responder count. Added: --data/--catalog for Reddit-Books, --tag for the output names, and the Qwen3.5
processor-config preparation (crs_eval.prepare_qwen35) so our SFT checkpoints load in vLLM.

Usage (GPU node, system python3 with vLLM):
  python3 remark/scripts/crs_consensus_check2.py --model <sft> --data <sft_dataset> --catalog <catalog.pkl> --tag books_qwen2B
Output: outputs/crs/consensus_check/result_<tag>.json, per_context_<tag>.jsonl
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
RG = REPO / "remark/third_party/Rank-GRPO"
sys.path.insert(0, str(RG / "libs"))
from utils import process_rec_raw  # noqa: E402

G = 8


def parse(text):
    item = {"raw": text}
    r = process_rec_raw(item, "raw", "rec")
    recs = (r[1] if isinstance(r, tuple) else item)["rec"]
    return [(t, int(y)) for t, y in recs][:20]


def ndcg_weighted(recs, gt_w, seen, catalog):
    """gt_w: {(title, year): weight}; a rec hits the first unmatched gt with same title and |dy|<=2."""
    used, gains = set(), []
    for r in recs:
        g_hit = 0.0
        if r[0] not in seen and r in catalog:
            for g, w in gt_w.items():
                if g not in used and r[0] == g[0] and abs(r[1] - g[1]) <= 2:
                    used.add(g); g_hit = w
                    break
        gains.append(g_hit)
    dcg = sum(x / np.log2(i + 2) for i, x in enumerate(gains))
    ideal = sorted(gt_w.values(), reverse=True)[:20]
    idcg = sum(x / np.log2(i + 2) for i, x in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-contexts", type=int, default=500)
    ap.add_argument("--out", default=str(REPO / "remark/outputs/crs/consensus_check"))
    ap.add_argument("--data", default=str(REPO / "data/raw/reddit_v2/processed_datasets/sft_dataset"))
    ap.add_argument("--catalog", default=str(RG / "reddit_catalog/gt_catalog.pkl"))
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    catalog = set(tuple(x) for x in pickle.load(open(a.catalog, "rb")))
    tr = load_from_disk(str(Path(a.data) / "train"))
    prompts = tr["prompt"]
    groups = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[p[0]["content"]].append(i)
    multi = [c for c, rows in groups.items() if len(rows) >= 2]
    random.seed(42)
    ctxs = random.sample(multi, min(a.n_contexts, len(multi)))
    from vllm import LLM, SamplingParams
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from crs_eval import prepare_qwen35
    llm = LLM(model=a.model, gpu_memory_utilization=0.85, max_model_len=8192, **prepare_qwen35(Path(a.model)))
    outs = llm.chat([[{"role": "user", "content": c}] for c in ctxs],
                    SamplingParams(n=G, temperature=1.0, top_p=1.0, max_tokens=1024, seed=42))
    zero_S, zero_C, std_S, std_C, noise_S, rhos, rows_meta = [], [], [], [], [], [], []
    for c, o in zip(ctxs, outs):
        rows = groups[c]
        samples = [parse(x.text) for x in o.outputs]
        row_gts, seen = [], set()
        for i in rows:
            it = tr[i]
            row_gts.append({(g[0], int(g[1])): 1.0 for g in it["groundtruth_with_release_year"]})
            seen |= set(it["seen_titles"])
        # consensus weights: fraction of responders recommending each item
        cw = defaultdict(float)
        for gw in row_gts:
            for g in gw:
                cw[g] += 1.0 / len(row_gts)
        S = np.array([[ndcg_weighted(s, gw, seen, catalog) for s in samples] for gw in row_gts])  # rows x G
        C = np.array([ndcg_weighted(s, dict(cw), seen, catalog) for s in samples])
        zero_S.append(float(np.mean([(S[r] == 0).all() for r in range(len(rows))])))
        zero_C.append(float((C == 0).all()))
        std_S.append(float(np.mean(S.std(axis=1)))); std_C.append(float(C.std()))
        noise_S.append(float(np.mean(S.std(axis=0))))
        advC = C - C.mean()
        for r in range(len(rows)):
            advS = S[r] - S[r].mean()
            if advS.std() > 0 and advC.std() > 0:
                rhos.append(abs(spearmanr(advS, advC).correlation))
        rows_meta.append({"n_rows": len(rows), "S": S.tolist(), "C": C.tolist()})
    res = {"tag": a.tag, "model": a.model, "data": a.data, "n_contexts": len(ctxs), "mean_rows_per_context": float(np.mean([len(groups[c]) for c in ctxs])),
           "zero_group_rate_S": float(np.mean(zero_S)), "zero_group_rate_C": float(np.mean(zero_C)),
           "within_group_std_S": float(np.mean(std_S)), "within_group_std_C": float(np.mean(std_C)),
           "cross_row_noise_S": float(np.mean(noise_S)),
           "mean_abs_spearman_advS_advC": float(np.mean(rhos)) if rhos else None, "n_rho": len(rhos)}
    zS, zC = res["zero_group_rate_S"], res["zero_group_rate_C"]
    rho = res["mean_abs_spearman_advS_advC"] or 1.0
    res["verdict"] = ("PASS" if (zC <= 0.7 * zS and rho < 0.8) else "PRUNE" if zC > 0.9 * zS else "INCONCLUSIVE")
    (out / f"result_{a.tag}.json").write_text(json.dumps(res, indent=2))
    with open(out / f"per_context_{a.tag}.jsonl", "w") as f:
        for m in rows_meta:
            f.write(json.dumps(m) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
