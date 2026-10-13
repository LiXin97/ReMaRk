"""Breakdown of a paired CRS comparison (two crs_eval.py outputs on the same split): per-row and request-level R@20,
differences by the number of rows (responders) per context, and item recall@20 by train popularity.

Usage: env/bin/python remark/scripts/crs_breakdown.py --a <run dir> --b <run dir> [--split validation] [--out <json>]
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
from crs_eval import DATA, RG, hits_for, recall_ndcg  # noqa: E402

BUCKETS = {"n=1": (1, 1), "n=2-4": (2, 4), "n=5-19": (5, 19), "n>=20": (20, 10**9)}


def per_row(run: Path, ds, catalog):
    r20, gt_hit = [], []
    for r, it in zip(map(json.loads, open(run / "per_instance.jsonl")), ds):
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        hits, matched = hits_for([(t, int(y)) for t, y in r["recs"]], gts, set(it["seen_titles"]), catalog, "truncate")
        r20.append(recall_ndcg(hits, len(gts), 20)[0])
        got = {gi for gi, pos in matched if pos < 20}
        gt_hit.append([(g, int(gi in got)) for gi, g in enumerate(gts)])
    return np.array(r20), gt_hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ds = load_from_disk(str(DATA / a.split))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    ra, ha = per_row(Path(a.a), ds, catalog)
    rb, hb = per_row(Path(a.b), ds, catalog)
    ctx = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx[p[0]["content"]].append(i)
    n = np.zeros(len(ds), int)
    for g in ctx.values():
        n[g] = len(g)
    out = {"a": a.a, "b": a.b, "split": a.split,
           "per_row": {"a": float(ra.mean()), "b": float(rb.mean()), "diff": float((ra - rb).mean())},
           "request_level": {"a": float(np.mean([ra[g].mean() for g in ctx.values()])),
                             "b": float(np.mean([rb[g].mean() for g in ctx.values()]))}}
    out["request_level"]["diff"] = out["request_level"]["a"] - out["request_level"]["b"]
    out["by_n"] = {}
    for name, (lo, hi) in BUCKETS.items():
        m = (n >= lo) & (n <= hi)
        out["by_n"][name] = {"row_share": float(m.mean()), "a": float(ra[m].mean()), "b": float(rb[m].mean()),
                             "diff": float((ra - rb)[m].mean())}
    pop = {tuple(json.loads(k)): v for k, v in json.load(open(RG.parents[1] / "outputs/crs/train_gt_popularity.json")).items()}
    popv = np.array([pop.get(g, 0) for row in ha for g, _ in row])
    qa = np.array([h for row in ha for _, h in row]); qb = np.array([h for row in hb for _, h in row])
    q50, q80 = np.quantile(popv, [0.5, 0.8])
    out["item_recall_by_pop"] = {b: {"a": float(qa[m].mean()), "b": float(qb[m].mean()), "diff": float((qa - qb)[m].mean())}
                                 for b, m in {"unseen(0)": popv == 0, "tail(<=median)": popv <= q50,
                                              "mid": (popv > q50) & (popv <= q80), "head(top20%)": popv > q80}.items()}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
