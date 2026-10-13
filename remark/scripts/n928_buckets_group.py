"""Per-bucket per-row test differences along the group-scale path Rank-GRPO (row 1) -> row 8 (request
sampling) -> row 7 (ReMaRk, bounded-sum reward), in the format of outputs/crs/analysis/test_2B_decomposition_buckets.json.
Buckets by n = rows (responders) of the request in the test split; per bucket the mean per-row R@20 difference a - b and a
95% percentile CI from a paired bootstrap over the bucket's requests (2,000 resamples, seed 42). Per-row R@20 with
crs_breakdown.per_row (Rank-GRPO's matcher). A check pair from the existing file is recomputed alongside.
Usage: env/bin/python remark/scripts/n928_buckets_group.py
Output: outputs/crs/analysis/test_2B_decomposition_buckets_group.json
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_breakdown import per_row  # noqa: E402
from crs_eval import DATA, RG  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
T = REPO / "remark/outputs/crs/test"
OUT = REPO / "remark/outputs/crs/analysis/test_2B_decomposition_buckets_group.json"
PAIRS = [("test_2B_singlethr_group_s3407_p0c_step2000", "test_2B_single_group_s3407_step2000"),   # 8 - 1
         ("test_2B_union_group_step2000", "test_2B_singlethr_group_s3407_p0c_step2000"),          # 7 - 8
         ("test_2B_union_group_step2000", "test_2B_single_group_s3407_step2000"),                 # 7 - 1 (total)
         ("test_2B_union_s3407_step1500", "test_2B_singlethr_s3407_step1750")]                    # check vs existing file
BUCKETS = {"n=1": (1, 1), "n=2-4": (2, 4), "n=5-19": (5, 19), "n>=20": (20, 10**9), "n>=2": (2, 10**9),
           "n>=5": (5, 10**9), "all": (1, 10**9)}


def main() -> None:
    ds = load_from_disk(str(DATA / "test"))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    ctx = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx[p[0]["content"]].append(i)
    groups = list(ctx.values())
    n_of = np.array([len(g) for g in groups])
    cache = {}
    out = []
    for a, b in PAIRS:
        for r in (a, b):
            if r not in cache:
                cache[r] = per_row(T / r, ds, catalog)[0]
        d = cache[a] - cache[b]
        rec = {"a": a, "b": b}
        for name, (lo, hi) in BUCKETS.items():
            gs = [g for g, n in zip(groups, n_of) if lo <= n <= hi]
            sums = np.array([d[g].sum() for g in gs]); cnt = np.array([len(g) for g in gs])
            rng = np.random.default_rng(42)
            idx = rng.integers(0, len(gs), (2000, len(gs)))
            boot = sums[idx].sum(1) / cnt[idx].sum(1)
            rec[name] = {"diff": float(sums.sum() / cnt.sum()), "ci": [float(np.quantile(boot, 0.025)),
                                                                   float(np.quantile(boot, 0.975))],
                         "rows": int(cnt.sum()), "requests": len(gs)}
        out.append(rec)
        print(a, "-", b, {k: (round(100 * v["diff"], 2), [round(100 * x, 2) for x in v["ci"]]) for k, v in rec.items()
                          if isinstance(v, dict)})
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
