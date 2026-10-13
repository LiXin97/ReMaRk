"""Paired bootstrap between CRS eval runs (outputs of crs_eval.py) on the same split.

Per-row truncate R@k / N@k are recomputed from each run's per_instance.jsonl (the protocol of crs_eval.py). The
resampling unit is the conversation context (rows sharing a prompt; --unit conversation groups ReDial/INSPIRED rows by
dialogue), 2,000 resamples, seed 42; a resampled
statistic is the row-weighted mean over the drawn contexts.
--unit submission (Reddit-v2) resamples Reddit submissions instead: Reddit-v2 splits one submission into several
contexts (conversation paths sharing the opening post;  incident), and a drawn submission brings all rows of all
its contexts. The released data carry no submission id; submission = normalised opening post
(crs_neighbor_prior.submission_key, the definition). The estimand does not change with the unit: --weighting row is
the per-row mean, --weighting unit the mean over contexts (requests) of the within-context mean, both computed over the
drawn submissions as ratio estimates; only the CI changes.

Usage: python3 remark/scripts/crs_compare.py --a <run dir> --b <run dir> [--a ...] --split test --out <json>
       [--unit context|conversation|submission] [--weighting row|unit]
       (A and B are paired: difference = A - B for each --a/--b pair; the dirs contain per_instance.jsonl)
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
from crs_eval import DATA, KS, RG, hits_for, recall_ndcg  # noqa: E402
from crs_neighbor_prior import submission_key  # noqa: E402

SUBMISSION_KEY = "normalised opening post (crs_neighbor_prior.submission_key; first USER turn, whitespace-collapsed)"


def per_row(run: Path, ds, catalog) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in open(run / "per_instance.jsonl")]
    assert len(rows) == len(ds), (run, len(rows), len(ds))
    out = {f"{m}@{k}": np.zeros(len(ds)) for m in ("R", "N") for k in KS}
    for r, it in zip(rows, ds):
        recs = [(t, int(y)) for t, y in r["recs"]]
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        hits, _ = hits_for(recs, gts, set(it["seen_titles"]), catalog, "truncate")
        for k in KS:
            rk, nk = recall_ndcg(hits, len(gts), k)
            out[f"R@{k}"][r["i"]], out[f"N@{k}"][r["i"]] = rk, nk
    return out


def context_groups(prompts: list[str]) -> dict[str, list[int]]:
    """Row indices per context (rows sharing a prompt), in first-appearance order."""
    ctx: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        ctx[p].append(i)
    return ctx


def submission_clusters(contexts: list[str]) -> list[list[int]]:
    """Indices into `contexts` grouped by Reddit-v2 submission (first-appearance order)."""
    sub: dict[str, list[int]] = defaultdict(list)
    for j, c in enumerate(contexts):
        sub[submission_key(c)].append(j)
    return list(sub.values())


def paired_stats(xa: np.ndarray, xb: np.ndarray, groups: list[list[int]], draws: np.ndarray, weighting: str,
                 clusters: list[list[int]] | None = None) -> dict[str, float]:
    """Paired bootstrap of mean(xa) - mean(xb). groups = row indices per context (or dialogue); draws index groups,
    or clusters of groups (submissions) when `clusters` is given."""
    counts = np.array([len(g) for g in groups], dtype=float)
    d = xa - xb
    sums = np.array([d[g].sum() for g in groups])
    if clusters is None and weighting == "row":
        boot = np.array([sums[idx].sum() / counts[idx].sum() for idx in draws])
        ma, mb, md = xa.mean(), xb.mean(), d.mean()
    elif clusters is None:  # mean over units of the within-unit mean
        means = sums / counts
        boot = means[draws].mean(axis=1)
        ma = np.mean([xa[g].mean() for g in groups])
        mb = np.mean([xb[g].mean() for g in groups])
        md = means.mean()
    else:  # clustered: ratio estimate over the drawn clusters of the same row- or context-weighted mean
        if weighting == "row":
            num, den = sums, counts
            ma, mb, md = xa.mean(), xb.mean(), d.mean()
        else:
            num, den = sums / counts, np.ones(len(groups))
            ma = np.mean([xa[g].mean() for g in groups])
            mb = np.mean([xb[g].mean() for g in groups])
            md = num.mean()
        cnum = np.array([num[c].sum() for c in clusters])
        cden = np.array([den[c].sum() for c in clusters])
        boot = cnum[draws].sum(axis=1) / cden[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"a": float(ma), "b": float(mb), "diff": float(md), "ci_low": float(lo), "ci_high": float(hi),
            "p_le_0": float((boot <= 0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", action="append", required=True)
    ap.add_argument("--b", action="append", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--catalog", default=str(RG / "reddit_catalog/gt_catalog.pkl"))
    ap.add_argument("--unit", choices=["context", "conversation", "submission"], default="context",
                    help="bootstrap unit; conversation uses <data>/../meta/<split>.jsonl conv_id (ReDial, INSPIRED); "
                         "submission groups Reddit-v2 contexts by normalised opening post")
    ap.add_argument("--weighting", choices=["row", "unit"], default="row",
                    help="row: per-row mean (Rank-GRPO protocol, primary); unit: each context/dialogue weighted once "
                         "(request-level secondary metric,  amendment; with --unit submission each context is "
                         "still weighted once)")
    a = ap.parse_args()
    assert len(a.a) == len(a.b)
    ds = load_from_disk(str(Path(a.data) / a.split))
    catalog = set(tuple(x) for x in pickle.load(open(a.catalog, "rb")))
    if a.unit == "conversation":
        ctx = defaultdict(list)
        meta = [json.loads(line) for line in open(Path(a.data).parent / "meta" / f"{a.split}.jsonl")]
        assert len(meta) == len(ds)
        for i, m in enumerate(meta):
            ctx[m["conv_id"]].append(i)
    else:
        ctx = context_groups([p[0]["content"] for p in ds["prompt"]])
    groups = list(ctx.values())
    clusters = submission_clusters(list(ctx)) if a.unit == "submission" else None
    n_units = len(clusters) if clusters is not None else len(groups)
    rng = np.random.default_rng(a.seed)
    draws = rng.integers(0, n_units, size=(a.n_boot, n_units))
    # per-context sums and counts let each resample be a weighted sum instead of a Python loop
    cache, results = {}, []
    for ra, rb in zip(a.a, a.b):
        for r in (ra, rb):
            if r not in cache:
                cache[r] = per_row(Path(r), ds, catalog)
        res = {"a": ra, "b": rb}
        for key in cache[ra]:
            res[key] = paired_stats(cache[ra][key], cache[rb][key], groups, draws, a.weighting, clusters)
        results.append(res)
        print(json.dumps({"a": Path(ra).name, "b": Path(rb).name, "R@20": res["R@20"], "N@20": res["N@20"]}))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    head = {"data": a.data, "split": a.split, "unit": a.unit, "weighting": a.weighting, "n_rows": len(ds),
            "n_units": n_units}
    if clusters is not None:
        head.update(n_contexts=len(groups), submission_key=SUBMISSION_KEY)
    Path(a.out).write_text(json.dumps({**head, "n_boot": a.n_boot, "seed": a.seed, "pairs": results}, indent=2))


if __name__ == "__main__":
    main()
