"""Paired bootstrap of CRS eval runs under two resampling units (Reddit-v2 contexts; Reddit-v2 submissions) and two
weightings (per row; per request), written to one JSON.

Every cell equals what crs_compare.py gives with the same --unit / --weighting (same functions, same draws: seed 42,
2,000 resamples, one draw matrix per unit shared by all pairs and metrics). Submission = normalised opening post
(crs_neighbor_prior.submission_key); the point estimates do not depend on the unit, only the CIs do.

Usage: env/bin/python remark/scripts/crs_compare_units.py --a <run dir> --b <run dir> [--a ... --b ...] \
           --split test --out remark/outputs/crs/analysis/test_2B_submission_clustered.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_compare import SUBMISSION_KEY, context_groups, paired_stats, per_row, submission_clusters  # noqa: E402
from crs_eval import DATA, REPO, RG  # noqa: E402

WEIGHTINGS = {"per_row": "row", "per_request": "unit"}


def rel(p: str | Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", action="append", required=True)
    ap.add_argument("--b", action="append", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert len(a.a) == len(a.b)
    ds = load_from_disk(str(DATA / a.split))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    ctx = context_groups([p[0]["content"] for p in ds["prompt"]])
    groups = list(ctx.values())
    clusters = submission_clusters(list(ctx))
    units = {"context": None, "submission": clusters}
    draws = {}
    for u, cl in units.items():
        n = len(groups) if cl is None else len(cl)
        draws[u] = np.random.default_rng(a.seed).integers(0, n, size=(a.n_boot, n))  # as crs_compare.py per unit
    cache: dict[str, dict[str, np.ndarray]] = {}
    pairs, table = [], []
    for ra, rb in zip(a.a, a.b):
        for r in (ra, rb):
            if r not in cache:
                cache[r] = per_row(Path(r), ds, catalog)
        res = {"a": rel(ra), "b": rel(rb)}
        for wname, w in WEIGHTINGS.items():
            res[wname] = {u: {key: paired_stats(cache[ra][key], cache[rb][key], groups, draws[u], w, cl)
                              for key in cache[ra]} for u, cl in units.items()}
            for key in ("R@20", "N@20"):
                c, s = res[wname]["context"][key], res[wname]["submission"][key]
                assert (c["a"], c["b"], c["diff"]) == (s["a"], s["b"], s["diff"])
                table.append({"a": Path(ra).name, "b": Path(rb).name, "weighting": wname, "metric": key,
                              "diff": c["diff"], "ci_context": [c["ci_low"], c["ci_high"]],
                              "ci_submission": [s["ci_low"], s["ci_high"]],
                              "ci_gt_0": {"context": c["ci_low"] > 0, "submission": s["ci_low"] > 0}})
        pairs.append(res)
        r20 = [res[w][u]["R@20"] for w in WEIGHTINGS for u in units]
        print(f"{Path(ra).name} - {Path(rb).name}: R@20 row {r20[0]['diff']:+.4f} ctx [{r20[0]['ci_low']:+.4f}, "
              f"{r20[0]['ci_high']:+.4f}] sub [{r20[1]['ci_low']:+.4f}, {r20[1]['ci_high']:+.4f}] | request "
              f"{r20[2]['diff']:+.4f} ctx [{r20[2]['ci_low']:+.4f}, {r20[2]['ci_high']:+.4f}] sub "
              f"[{r20[3]['ci_low']:+.4f}, {r20[3]['ci_high']:+.4f}]")
    sub_sizes = Counter(len(c) for c in clusters)
    out = {"data": rel(DATA), "split": a.split, "n_rows": len(ds), "n_requests": len(groups),
           "n_submissions": len(clusters), "submission_key": SUBMISSION_KEY,
           "requests_per_submission": {str(k): sub_sizes[k] for k in sorted(sub_sizes)},
           "rows_in_multi_request_submissions": int(sum(len(groups[j]) for c in clusters if len(c) > 1 for j in c)),
           "n_boot": a.n_boot, "seed": a.seed,
           "definitions": {"per_row": "row-weighted mean of truncate metrics (Rank-GRPO protocol; primary)",
                           "per_request": "mean over requests (contexts) of the within-request mean",
                           "context": "resample requests (crs_compare.py --unit context)",
                           "submission": "resample submissions; all rows of all requests of a drawn submission enter "
                                         "together (crs_compare.py --unit submission)",
                           "diff": "a - b", "ci": "2.5 / 97.5 percentiles of the bootstrap distribution"},
           "summary": table, "pairs": pairs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n_rows", "n_requests", "n_submissions", "requests_per_submission")}))


if __name__ == "__main__":
    main()
