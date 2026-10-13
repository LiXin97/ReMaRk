"""Per-bucket paired bootstrap (over contexts, 2,000, seed 42) of per-row R@20 between two crs_eval.py runs.

--unit submission resamples Reddit-v2 submissions (normalised opening post, crs_neighbor_prior.submission_key) within
each bucket: a drawn submission brings all its contexts that fall in the bucket; the per-row estimand is unchanged.

Usage: env/bin/python remark/scripts/crs_bucket_boot.py --a <run> --b <run> [--a ... --b ...] [--split validation]
       [--unit context|submission]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_breakdown import BUCKETS, per_row  # noqa: E402
from crs_eval import DATA, RG  # noqa: E402
from crs_neighbor_prior import submission_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", action="append", required=True)
    ap.add_argument("--b", action="append", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default="")
    ap.add_argument("--unit", choices=["context", "submission"], default="context")
    a = ap.parse_args()
    ds = load_from_disk(str(DATA / a.split))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    ctx = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx[p[0]["content"]].append(i)
    groups = list(ctx.values())
    sizes = np.array([len(g) for g in groups])
    sub_id = None
    if a.unit == "submission":
        keys = {}
        sub_id = np.array([keys.setdefault(submission_key(c), len(keys)) for c in ctx])
    cache, results = {}, []
    rng = np.random.default_rng(42)
    for ra, rb in zip(a.a, a.b):
        for r in (ra, rb):
            if r not in cache:
                cache[r] = per_row(Path(r), ds, catalog)[0]
        d = cache[ra] - cache[rb]
        sums = np.array([d[g].sum() for g in groups])
        res = {"a": Path(ra).name, "b": Path(rb).name}
        if sub_id is not None:
            res["unit"] = "submission"
        buckets = dict(BUCKETS, **{"n>=2": (2, 10**9), "n>=5": (5, 10**9), "all": (1, 10**9)})
        for name, (lo, hi) in buckets.items():
            sel = np.where((sizes >= lo) & (sizes <= hi))[0]
            if sub_id is None:
                draws = rng.integers(0, len(sel), size=(2000, len(sel)))
                boot = np.array([sums[sel[i]].sum() / sizes[sel[i]].sum() for i in draws])
            else:  # clusters = submissions among the bucket's contexts
                cl = defaultdict(list)
                for j in sel:
                    cl[sub_id[j]].append(j)
                cs = np.array([sums[c].sum() for c in cl.values()])
                cn = np.array([sizes[c].sum() for c in cl.values()], dtype=float)
                draws = rng.integers(0, len(cl), size=(2000, len(cl)))
                boot = cs[draws].sum(axis=1) / cn[draws].sum(axis=1)
            res[name] = {"diff": float(sums[sel].sum() / sizes[sel].sum()),
                         "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
            if sub_id is not None:
                res[name].update(n_contexts=int(len(sel)), n_submissions=len(cl))
        results.append(res)
        print(res["a"][-40:], "vs", res["b"][-40:], " ".join(f"{k}:{v['diff']:+.4f}[{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]" for k, v in res.items() if isinstance(v, dict)))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
