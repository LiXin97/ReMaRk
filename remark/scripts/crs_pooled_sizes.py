"""Pooled cross-size estimate (registered secondary,  2026-09-29 ~16:50): the mean over sizes of the per-size paired
per-row R@20 difference (a - b), with a paired bootstrap that resamples test requests once and applies the same draw to
every size (2,000 resamples, seed 42).

Usage: env/bin/python remark/scripts/crs_pooled_sizes.py --pair <a_dir> <b_dir> [--pair ...] --out <json>
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
from crs_breakdown import per_row  # noqa: E402
from crs_eval import DATA, RG  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = load_from_disk(str(DATA / a.split))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    ctx = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx[p[0]["content"]].append(i)
    groups = list(ctx.values())
    sizes = np.array([len(g) for g in groups], dtype=float)
    sums = []
    per_size = []
    for ra, rb in a.pair:
        d = per_row(Path(ra), ds, catalog)[0] - per_row(Path(rb), ds, catalog)[0]
        s = np.array([d[g].sum() for g in groups])
        sums.append(s)
        per_size.append({"a": Path(ra).name, "b": Path(rb).name, "diff": float(s.sum() / sizes.sum())})
    sums = np.stack(sums)  # (n_pairs, n_contexts)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    boot = np.array([(sums[:, i].sum(axis=1) / sizes[i].sum()).mean() for i in draws])
    point = float(np.mean([p["diff"] for p in per_size]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out = {"split": a.split, "n_contexts": len(groups), "per_size": per_size, "pooled_mean_diff": point,
           "ci95": [float(lo), float(hi)], "n_boot": 2000, "seed": 42}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
