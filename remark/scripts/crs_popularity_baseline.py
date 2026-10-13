"""Reddit-v2 popularity baseline (non-personalized): the most frequent training ground-truth items.

Definition (fixed): item popularity = number of occurrences of (title, year) in the ground truth of the Reddit-v2
training split (sft_dataset/train, every row's items; identical to remark/outputs/crs/train_gt_popularity.json, which
crs_eval.py uses for its popularity buckets; asserted). Items are ranked by count, ties broken by title then year.
For each request (row), the list is the top of that ranking with every item whose title is in the row's seen_titles
removed (the exact-title test crs_eval.py applies to seen items), truncated to 20.

Scoring is crs_eval.py's (hits_for / recall_ndcg; truncate and filter variants, per-row means; item recall@20 by
train popularity) plus request-level metrics (mean over contexts of the within-context mean of truncate R@k / N@k).
Writes per_instance.jsonl + summary.json in crs_eval.py's format to remark/outputs/crs/{val,test}/<name>/ and a
two-split summary to remark/outputs/crs/analysis/popularity_baseline.json.

Usage: env/bin/python remark/scripts/crs_popularity_baseline.py [--splits validation test] [--name popularity_train_top20]
       [--dataset reddit|books|music]   (Table 2: the same definition on data/crs/{reddit_books,music}; the name
       gets a _books/_music suffix; popularity cache outputs/crs/train_gt_popularity_<data dir>.json as crs_eval.py writes it)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import DATA, KS, REPO, RG, hits_for, recall_ndcg  # noqa: E402

OUT = REPO / "remark/outputs/crs"
SPLIT_DIR = {"validation": "val", "test": "test"}
TOP = 20


def train_popularity(pop_json: Path, data: Path = DATA) -> Counter:
    """Stream the training GT column once; must equal the cached file crs_eval.py wrote."""
    tr = load_from_disk(str(data / "train")).select_columns(["groundtruth_with_release_year"])
    pop: Counter = Counter()
    for batch in tr.iter(batch_size=10_000):
        for gts in batch["groundtruth_with_release_year"]:
            for g in gts:
                pop[(g[0], int(g[1]))] += 1
    if pop_json.exists():
        cached = Counter({tuple(json.loads(k)): v for k, v in json.load(open(pop_json)).items()})
        assert cached == pop, "train_gt_popularity.json differs from the training split"
    return pop


def recommend(ranking: list[tuple[str, int]], seen: set[str]) -> list[tuple[str, int]]:
    out = []
    for item in ranking:
        if item[0] in seen:
            continue
        out.append(item)
        if len(out) == TOP:
            break
    return out


def evaluate(split: str, ranking: list[tuple[str, int]], pop: Counter, catalog: set, name: str, data: Path = DATA) -> dict:
    """crs_eval.py's scoring loop, with the popularity list in place of the parsed generation."""
    ds = load_from_disk(str(data / split))
    ctx_idx: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx_idx[p[0]["content"]].append(i)
    res = {m: {k: [[], []] for k in KS} for m in ("truncate", "filter")}
    item_level, per = [], []
    n_parse_err, n_in_cat, n_recs, n_excl = 0, 0, 0, 0
    top_global = ranking[:TOP]
    for i, it in enumerate(ds):
        seen = set(it["seen_titles"])
        recs = recommend(ranking, seen)
        n_excl += recs != top_global
        n_recs += len(recs)
        n_in_cat += sum(r in catalog for r in recs)
        if not recs:
            n_parse_err += 1
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        for m in ("truncate", "filter"):
            hits, matched = hits_for(recs, gts, seen, catalog, m)
            for k in KS:
                r, n = recall_ndcg(hits, len(gts), k)
                res[m][k][0].append(r)
                res[m][k][1].append(n)
            if m == "truncate":
                hit_gts = {gi for gi, pos in matched if pos < 20}
                for gi, g in enumerate(gts):
                    item_level.append((pop.get(g, 0), int(gi in hit_gts)))
        per.append({"i": i, "recs": recs, "gts": gts})
    assert all(len(r["recs"]) == TOP for r in per)

    summary = {"model": name, "data": str(data.relative_to(REPO)), "split": split, "n": len(ds),
               "unique_contexts": len(ctx_idx), "parse_empty": n_parse_err, "in_catalog_rate": n_in_cat / max(1, n_recs)}
    for m in res:
        for k in KS:
            summary[f"{m}_R@{k}"] = float(np.mean(res[m][k][0]))
            summary[f"{m}_N@{k}"] = float(np.mean(res[m][k][1]))
    popv = np.array([p for p, _ in item_level]); hitv = np.array([h for _, h in item_level])
    qs = np.quantile(popv, [0.5, 0.8, 0.9])
    summary["gt_pop_quantiles_50_80_90"] = qs.tolist()
    buckets = {"unseen(0)": popv == 0, "tail(<=median)": popv <= qs[0], "mid": (popv > qs[0]) & (popv <= qs[1]),
               "head(top20%)": popv > qs[1], "top10%": popv > qs[2]}
    summary["item_recall@20_by_pop"] = {b: {"n": int(msk.sum()), "recall": float(hitv[msk].mean()) if msk.any() else None}
                                        for b, msk in buckets.items()}
    # additions beyond crs_eval.py's summary
    for k in KS:
        for j, met in enumerate(("R", "N")):
            v = np.array(res["truncate"][k][j])
            summary[f"request_truncate_{met}@{k}"] = float(np.mean([v[g].mean() for g in ctx_idx.values()]))
    rec_sets = [{tuple(map(tuple, per[i]["recs"])) for i in g} for g in ctx_idx.values()]
    summary["rows_with_seen_exclusion"] = int(n_excl)
    summary["contexts_with_row_varying_lists"] = int(sum(len(s) > 1 for s in rec_sets))
    summary["definition"] = ("top-20 (title, year) by training-GT frequency (sft_dataset/train, every row's items; ties "
                             "by title then year), minus the row's seen_titles (exact title), truncated to 20")
    out = OUT / SPLIT_DIR[split] / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "per_instance.jsonl", "w") as f:
        for r in per:
            f.write(json.dumps(r) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["validation", "test"])
    ap.add_argument("--name", default="popularity_train_top20")
    ap.add_argument("--summary_out", default=None)
    ap.add_argument("--dataset", choices=["reddit", "books", "music"], default="reddit")
    a = ap.parse_args()
    if a.dataset == "reddit":
        data, cat_path, tag, suffix = DATA, RG / "reddit_catalog/gt_catalog.pkl", "", ""
    else:
        data = REPO / ("remark/data/crs/" + ("reddit_books" if a.dataset == "books" else "music")) / "sft_dataset"
        cat_path, tag, suffix = data.parent / "catalog.pkl", "_" + data.parent.name, "_" + a.dataset
        a.name += suffix
    a.summary_out = a.summary_out or str(OUT / f"analysis/popularity_baseline{suffix}.json")
    pop_json = OUT / f"train_gt_popularity{tag}.json"  # crs_eval.py's cache name
    pop = train_popularity(pop_json, data)
    ranking = sorted(pop, key=lambda k: (-pop[k], k[0], k[1]))
    catalog = set(tuple(x) for x in pickle.load(open(cat_path, "rb")))
    keys = [f"truncate_{m}@{k}" for k in (10, 20) for m in ("R", "N")] + \
           [f"request_truncate_{m}@{k}" for k in (10, 20) for m in ("R", "N")]
    report = {"name": a.name, "definition": ("popularity = occurrences of (title, year) in the " + {"reddit": "Reddit-v2", "books": "Reddit-Books", "music": "music"}[a.dataset] + " training "
                                             "ground truth (sft_dataset/train, every row's items); rank by count, ties by "
                                             "title then year; per row drop titles in seen_titles (exact title match, as "
                                             "crs_eval.py), keep 20"),
              "popularity_source": {"split": "sft_dataset/train", "file": str(pop_json.relative_to(REPO)),
                                    "equals_recomputed": True, "n_items": len(pop),
                                    "n_occurrences": int(sum(pop.values()))},
              "global_top20": [{"title": t, "year": y, "count": pop[(t, y)]} for t, y in ranking[:TOP]],
              "protocol": "crs_eval.py truncate scoring; per-row = mean over rows; request-level = mean over contexts "
                          "of the within-context mean", "splits": {}}
    for sp in a.splits:
        s = evaluate(sp, ranking, pop, catalog, a.name, data)
        report["splits"][sp] = {"dir": str((OUT / SPLIT_DIR[sp] / a.name).relative_to(REPO)), "n_rows": s["n"],
                                "n_requests": s["unique_contexts"], **{k: s[k] for k in keys},
                                "filter_R@20": s["filter_R@20"], "in_catalog_rate": s["in_catalog_rate"],
                                "rows_with_seen_exclusion": s["rows_with_seen_exclusion"],
                                "contexts_with_row_varying_lists": s["contexts_with_row_varying_lists"]}
        print(sp, json.dumps(report["splits"][sp]))
    Path(a.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary_out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
