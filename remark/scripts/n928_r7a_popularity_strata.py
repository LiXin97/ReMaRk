"""Popularity-stratified test recall, ReMaRk vs Rank-GRPO (Reddit-v2 2B/4B/9B) and ReMaRk vs per-responder
batch (the main-table comparison arm) and vs Rank-GRPO (Reddit-Books 2B/4B). CPU only, existing outputs.

Inputs
  runs:        remark/outputs/crs/test/<run>/per_instance.jsonl (+ summary.json), run names from
               make_paper_stats.TEST (arms "remark", "rankgrpo", "perresp")
  data:        Reddit-v2 data/raw/reddit_v2/processed_datasets/sft_dataset/test, catalogue third_party/Rank-GRPO/
               reddit_catalog/gt_catalog.pkl; Reddit-Books remark/data/crs/reddit_books/sft_dataset/test, catalog.pkl
  popularity:  remark/outputs/crs/train_gt_popularity.json (Reddit-v2), train_gt_popularity_reddit_books.json (Books):
               count of each (title, year) in the training ground truth (every row's items); unseen items = 0
Definitions
  Buckets copy crs_val_table.py (Table 5): q50, q80 = np.quantile(popv, [0.5, 0.8]) where popv is the training
  popularity of every ground-truth occurrence (row x GT item) of the evaluated split. Table 5 evaluated the validation
  split; here the evaluated split is test, so the quantiles are taken over the test GT occurrences (the validation
  quantiles are recorded for comparison). tail: pop <= q50; torso: q50 < pop <= q80; head: pop > q80.
  Per row and bucket b: R@20_b = (# of the row's GT items in b matched within the top 20) / (# of the row's GT items in
  b); matching = crs_breakdown.per_row (crs_eval.hits_for, truncate mode: catalogue items only, seen titles skipped,
  exact title + |year diff| <= 2). Rows with no GT item in b are excluded from b.
  Paired bootstrap (per bucket): units = contexts (rows sharing a prompt) that keep >= 1 row in b; 2,000 resamples,
  numpy default_rng(42) drawn per bucket; statistic = crs_compare.paired_stats (weighting "row": row-weighted mean over
  the drawn contexts, primary; "unit": mean over drawn contexts of the within-context mean, per request); 95%
  percentile CI. Bucket "all" (no restriction) reproduces the main-table paired difference (same draws as crs_compare).
  Also reported per arm: pooled item-level recall@20 per bucket (Table 5's own statistic: hits / GT occurrences).
  Sensitivity ("validation_cutoffs"): the same per-row comparison with the validation quantiles as cutoffs (Table 5's
  numerical thresholds), per row only.
Output: remark/outputs/crs/analysis/n928_r7a_popularity_strata.json
Usage:  env/bin/python remark/scripts/n928_r7a_popularity_strata.py
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_breakdown import per_row  # noqa: E402
from crs_compare import paired_stats  # noqa: E402
from crs_eval import DATA, REPO, RG  # noqa: E402
from make_paper_stats import TEST  # noqa: E402

C = REPO / "remark/outputs/crs"
OUT = C / "analysis/n928_r7a_popularity_strata.json"
BOOKS = REPO / "remark/data/crs/reddit_books"
DATASETS = {
    "reddit": {"data": DATA, "catalog": RG / "reddit_catalog/gt_catalog.pkl", "pop": C / "train_gt_popularity.json"},
    "books": {"data": BOOKS / "sft_dataset", "catalog": BOOKS / "catalog.pkl",
              "pop": C / "train_gt_popularity_reddit_books.json"},
}
# setting -> (dataset, [(arm a, arm b), ...]); differences are a - b
SETTINGS = {"2B": ("reddit", [("remark", "rankgrpo")]), "4B": ("reddit", [("remark", "rankgrpo")]),
            "9B": ("reddit", [("remark", "rankgrpo")]),
            "books": ("books", [("remark", "perresp"), ("remark", "rankgrpo")]),
            "books4B": ("books", [("remark", "perresp"), ("remark", "rankgrpo")])}
BUCKETS = ("all", "tail", "torso", "head")
N_BOOT, SEED = 2000, 42


def load_pop(path: Path) -> dict[tuple[str, int], int]:
    return {tuple(json.loads(k)): v for k, v in json.load(open(path)).items()}


def thresholds(ds, pop: dict) -> tuple[float, float]:
    popv = np.array([pop.get((g[0], int(g[1])), 0) for gts in ds["groundtruth_with_release_year"] for g in gts])
    q50, q80 = np.quantile(popv, [0.5, 0.8])
    return float(q50), float(q80)


def bucket_of(p: int, q50: float, q80: float) -> str:
    return "tail" if p <= q50 else ("head" if p > q80 else "torso")


def per_bucket_recall(gt_hit: list, q50: float, q80: float, pop: dict) -> dict[str, np.ndarray]:
    """Per row: recall restricted to each bucket (NaN if the row has no GT item in it)."""
    out = {b: np.full(len(gt_hit), np.nan) for b in BUCKETS}
    for i, row in enumerate(gt_hit):
        num, den = defaultdict(int), defaultdict(int)
        for g, h in row:
            b = bucket_of(pop.get(g, 0), q50, q80)
            num[b] += h; den[b] += 1
            num["all"] += h; den["all"] += 1
        for b in den:
            out[b][i] = num[b] / den[b]
    return out


def item_recall(gt_hit: list, q50: float, q80: float, pop: dict) -> dict[str, dict]:
    hits, n = defaultdict(int), defaultdict(int)
    for row in gt_hit:
        for g, h in row:
            b = bucket_of(pop.get(g, 0), q50, q80)
            hits[b] += h; n[b] += 1
    return {b: {"n_gt": n[b], "recall": hits[b] / n[b]} for b in ("tail", "torso", "head")}


def compare(xa_full: np.ndarray, xb_full: np.ndarray, groups: list[list[int]], with_request: bool) -> dict:
    """Paired bootstrap over the contexts that keep >= 1 row (rows with NaN = no GT item in the bucket are dropped)."""
    keep = ~np.isnan(xa_full)
    assert (keep == ~np.isnan(xb_full)).all()
    new = -np.ones(len(keep), int)
    new[keep] = np.arange(keep.sum())
    gb = [g for g in ([int(new[i]) for i in g if keep[i]] for g in groups) if g]
    draws = np.random.default_rng(SEED).integers(0, len(gb), size=(N_BOOT, len(gb)))
    xa, xb = xa_full[keep], xb_full[keep]
    out = {"n_rows": int(keep.sum()), "n_contexts": len(gb), "per_row": paired_stats(xa, xb, gb, draws, "row")}
    if with_request:
        out["per_request"] = paired_stats(xa, xb, gb, draws, "unit")
    return out


def main() -> None:
    res = {"definition": __doc__.split("Definitions", 1)[1].split("Output:", 1)[0].strip(),
           "n_boot": N_BOOT, "seed": SEED, "datasets": {}, "settings": {}}
    try:
        res["git"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()
    except OSError:
        pass
    loaded = {}
    for name, d in DATASETS.items():
        ds = load_from_disk(str(Path(d["data"]) / "test"))
        val = load_from_disk(str(Path(d["data"]) / "validation"))
        catalog = set(tuple(x) for x in pickle.load(open(d["catalog"], "rb")))
        pop = load_pop(d["pop"])
        q50, q80 = thresholds(ds, pop)
        vq50, vq80 = thresholds(val, pop)
        prompts = [p[0]["content"] for p in ds["prompt"]]
        ctx = defaultdict(list)
        for i, p in enumerate(prompts):
            ctx[p].append(i)
        loaded[name] = (ds, catalog, pop, q50, q80, list(ctx.values()), vq50, vq80)
        res["datasets"][name] = {"test": str(Path(d["data"]) / "test"), "popularity": str(Path(d["pop"]).relative_to(REPO)),
                                 "rows": len(ds), "contexts": len(ctx),
                                 "test_quantiles_q50_q80": [q50, q80], "validation_quantiles_q50_q80": [vq50, vq80],
                                 "same_cutoffs_as_validation": (q50, q80) == (vq50, vq80)}
        print(name, res["datasets"][name], flush=True)
    for setting, (dname, pairs) in SETTINGS.items():
        ds, catalog, pop, q50, q80, groups, vq50, vq80 = loaded[dname]
        arms = sorted({a for p in pairs for a in p})
        per, per_v, out_arms = {}, {}, {}
        for arm in arms:
            run = C / "test" / TEST[setting][arm]
            r20, gt_hit = per_row(run, ds, catalog)
            summ = json.load(open(run / "summary.json"))
            assert abs(r20.mean() - summ["truncate_R@20"]) < 1e-9, (run, r20.mean(), summ["truncate_R@20"])
            per[arm] = per_bucket_recall(gt_hit, q50, q80, pop)
            per_v[arm] = per_bucket_recall(gt_hit, vq50, vq80, pop)
            assert np.allclose(per[arm]["all"], r20)
            out_arms[arm] = {"run": TEST[setting][arm], "R@20": float(r20.mean()),
                             "item_recall@20": item_recall(gt_hit, q50, q80, pop)}
        comps = []
        for a, b in pairs:
            comp = {"a": a, "b": b, "run_a": TEST[setting][a], "run_b": TEST[setting][b], "buckets": {},
                    "validation_cutoffs": {}}
            for bk in BUCKETS:
                comp["buckets"][bk] = compare(per[a][bk], per[b][bk], groups, with_request=True)
                s = comp["buckets"][bk]["per_row"]
                print(f"{setting} {a}-{b} {bk:5s} n={comp['buckets'][bk]['n_rows']:5d} a={s['a']:.4f} b={s['b']:.4f} "
                      f"d={s['diff']:+.4f} [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]", flush=True)
                if bk != "all":
                    comp["validation_cutoffs"][bk] = compare(per_v[a][bk], per_v[b][bk], groups, with_request=False)
            comps.append(comp)
        res["settings"][setting] = {"dataset": dname, "arms": out_arms, "comparisons": comps}
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
