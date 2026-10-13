"""C readout: frozen-configuration confirmation on Reddit-Books-Late (frozen configuration, no re-selection).

Inputs: outputs/crs/books_late/test/late_<run>_step<k>/per_instance.jsonl (crs_eval on data/crs/reddit_books_late,
catalog of that build). Scoring = crs_compare.per_row (crs_eval hits_for, truncate mode, R@20 / N@20 per row).
Pairs (difference = ReMaRk - comparator): paired bootstrap over requests (contexts), 2,000 resamples, numpy
default_rng(42), per row (primary) and per request (each context weighted once). Primary: Books 4B s3407 ReMaRk -
Rank-GRPO (Table 2), success = per-row CI > 0.
Catalog sensitivity (amendment): every pair again with each row's ground truth restricted to items whose (title, year)
is in the ORIGINAL Books catalog (data/crs/reddit_books/catalog.pkl); rows left without ground truth are dropped.
References: popularity baseline (top-20 training ground-truth items of the late build's training split, seen titles
removed, crs_popularity_baseline definition) and 2B s3407 ReMaRk @1500 alone.
Usage: env/bin/python remark/scripts/n933_c_readout.py -> outputs/crs/analysis/n933_c_books_late.json
"""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_compare import context_groups, paired_stats  # noqa: E402
from crs_eval import hits_for, recall_ndcg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
LATE = REPO / "remark/data/crs/reddit_books_late"
ORIG_CAT = REPO / "remark/data/crs/reddit_books/catalog.pkl"
T = C / "books_late/test"
P = "reddit_books_{}_p0c_cap384_ex"
RUN = {"4B_remark": (P.format("4B_union_group_s3407"), 2000), "4B_rankgrpo": (P.format("4B_single_group_s3407"), 2000),
       "4B_batch": (P.format("4B_single_batch_s3407"), 2000),
       "4B_remark_s42": (P.format("4B_union_group_s42"), 2000), "4B_batch_s42": (P.format("4B_single_batch_s42"), 2000),
       "2B_remark_s42": (P.format("2B_union_group_s42"), 2000), "2B_batch_s42": (P.format("2B_single_batch_s42"), 2000),
       "2B_remark_s1234": (P.format("2B_union_group_s1234"), 2000),
       "2B_batch_s1234": (P.format("2B_single_batch_s1234"), 2000),
       "R1d_remark": (P.format("2B_union_group_s3407") + "_ms8000", 2000),
       "R1d_rankgrpo": (P.format("2B_single_group_s3407") + "_ms8000", 3250),
       "2B_remark_s3407": (P.format("2B_union_group_s3407"), 1500),
       "2B_registered_union_batch": (P.format("2B_union_batch_s3407"), 2000)}
PAIRS = [("primary: 4B s3407 ReMaRk - Rank-GRPO", "4B_remark", "4B_rankgrpo"),
         ("4B s3407 ReMaRk - per-responder batch", "4B_remark", "4B_batch"),
         ("4B s42 ReMaRk - per-responder batch", "4B_remark_s42", "4B_batch_s42"),
         ("2B s42 ReMaRk - per-responder batch", "2B_remark_s42", "2B_batch_s42"),
         ("2B s1234 ReMaRk - per-responder batch", "2B_remark_s1234", "2B_batch_s1234"),
         ("R1d 2B plateau ReMaRk @2000 - Rank-GRPO @3250", "R1d_remark", "R1d_rankgrpo")]
KS = (20,)


def rows_of(key: str) -> list[dict]:
    run, step = RUN[key]
    p = T / f"late_{run}_step{step}" / "per_instance.jsonl"
    return [json.loads(line) for line in open(p)]


def scores(recs_by_row: list, ds, catalog: set, keep_gt=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row R@20, N@20 and a mask of rows with >= 1 ground-truth item (after the optional GT filter)."""
    n = len(ds)
    R, N, M = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    for i, (recs, it) in enumerate(zip(recs_by_row, ds)):
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        if keep_gt is not None:
            gts = [g for g in gts if g in keep_gt]
        if not gts:
            continue
        M[i] = True
        hits, _ = hits_for([(t, int(y)) for t, y in recs], gts, set(it["seen_titles"]), catalog, "truncate")
        R[i], N[i] = recall_ndcg(hits, len(gts), 20)
    return R, N, M


def pair_stats(xa, xb, mask, prompts) -> dict:
    idx = np.flatnonzero(mask)
    groups_all = context_groups([prompts[i] for i in idx])
    groups = list(groups_all.values())
    draws = np.random.default_rng(42).integers(0, len(groups), size=(2000, len(groups)))
    out = {}
    for w in ("row", "unit"):
        out[w] = paired_stats(xa[idx], xb[idx], groups, draws, w)
    out["n_rows"], out["n_requests"] = int(len(idx)), len(groups)
    return out


def main() -> None:
    ds = load_from_disk(str(LATE / "sft_dataset/test"))
    catalog = set(tuple(x) for x in pickle.load(open(LATE / "catalog.pkl", "rb")))
    orig = set(tuple(x) for x in pickle.load(open(ORIG_CAT, "rb")))
    prompts = [p[0]["content"] for p in ds["prompt"]]
    recs = {}
    for k in RUN:
        rows = rows_of(k)
        assert len(rows) == len(ds)
        r = [None] * len(ds)
        for x in rows:
            r[x["i"]] = x["recs"]
        recs[k] = r
    # popularity baseline: training split of the late build (same threads as the Books training split)
    tr = load_from_disk(str(LATE / "sft_dataset/train"))
    cnt = Counter((g[0], int(g[1])) for gts in tr["groundtruth_with_release_year"] for g in gts)
    ranked = sorted(cnt, key=lambda x: (-cnt[x], x[0], x[1]))
    pop = []
    for it in ds:
        seen = set(it["seen_titles"])
        pop.append([[t, y] for t, y in (x for x in ranked[:200] if x[0] not in seen)][:20])
    recs["popularity"] = pop
    res = {"protocol": " C + amendment", "data_sha256": json.load(open(C / "books_late/manifest.json"))["sha256_combined"],
           "n_rows": len(ds), "n_requests": len(set(prompts)), "runs": {k: f"late_{v[0]}_step{v[1]}" for k, v in RUN.items()},
           "arms": {}, "pairs": {}, "sensitivity_orig_catalog_gt": {}}
    full = {k: scores(v, ds, catalog) for k, v in recs.items()}
    sens = {k: scores(v, ds, catalog, keep_gt=orig) for k, v in recs.items()}
    for k, (R, N, M) in full.items():
        grp = context_groups(prompts)
        res["arms"][k] = {"R@20_row": float(R[M].mean()), "N@20_row": float(N[M].mean()),
                          "R@20_request": float(np.mean([R[g].mean() for g in grp.values()]))}
    for label, a, b in PAIRS:
        Ra, Na, Ma = full[a]
        Rb, Nb, Mb = full[b]
        res["pairs"][label] = {"a": a, "b": b, "R@20": pair_stats(Ra, Rb, Ma & Mb, prompts),
                               "N@20": pair_stats(Na, Nb, Ma & Mb, prompts)["row"]}
        Ra, _, Ma = sens[a]
        Rb, _, Mb = sens[b]
        res["sensitivity_orig_catalog_gt"][label] = pair_stats(Ra, Rb, Ma & Mb, prompts)
    d42 = res["pairs"]["2B s42 ReMaRk - per-responder batch"]["R@20"]["row"]["diff"]
    d1234 = res["pairs"]["2B s1234 ReMaRk - per-responder batch"]["R@20"]["row"]["diff"]
    res["2B_two_seed_mean_vs_batch_row"] = (d42 + d1234) / 2
    prim = res["pairs"]["primary: 4B s3407 ReMaRk - Rank-GRPO"]["R@20"]["row"]
    res["primary_success"] = bool(prim["ci_low"] > 0)
    res["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    (C / "analysis/n933_c_books_late.json").write_text(json.dumps(res, indent=1))
    for k, v in res["arms"].items():
        print(f"{k:28s} R@20 row {100 * v['R@20_row']:.2f}  request {100 * v['R@20_request']:.2f}")
    for label, v in res["pairs"].items():
        r, u, s = v["R@20"]["row"], v["R@20"]["unit"], res["sensitivity_orig_catalog_gt"][label]["row"]
        print(f"{label}: row {100 * r['diff']:+.2f} [{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}]  request "
              f"{100 * u['diff']:+.2f} [{100 * u['ci_low']:+.2f}, {100 * u['ci_high']:+.2f}]  | orig-catalog GT row "
              f"{100 * s['diff']:+.2f} [{100 * s['ci_low']:+.2f}, {100 * s['ci_high']:+.2f}]")
    print("primary success:", res["primary_success"], "| 2B two-seed mean vs batch (row):", f"{100 * res['2B_two_seed_mean_vs_batch_row']:+.2f}")


if __name__ == "__main__":
    main()
