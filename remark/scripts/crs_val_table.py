"""Paper tables from crs_eval.py outputs: per-arm columns and paired differences with bootstrap CIs, for any dataset.

Per arm (each at the checkpoint named in the spec): per-row R@20 / N@20 (from summary.json), request-level R@20,
per-row R@20 on n = 1 and n >= 5 contexts, item recall@20 of tail (<= median train popularity) and head (top 20%)
ground-truth items, exact-20 (share of lists with 20 parsed items), pooled R@20 (the context's list against the
union of all its responders' items), and exposure: coverage (distinct recommended items), mean train popularity of
recommended items, and the share of recommendations at or below the median ground-truth popularity (tail) or unseen. "trunc": cut every list to that many items
first (Reddit-Books robustness check for the one-word-20th-line exploit).
Per pair (a - b): per-row and request-level differences overall, on n = 1 and on n >= 5, with paired bootstrap CIs
over contexts (2,000 resamples, seed 42; per-row = row-weighted mean over drawn contexts, request-level = mean over
drawn contexts of the within-context mean).

Usage: env/bin/python remark/scripts/crs_val_table.py --spec remark/configs/paper_val_tables.json \
           --out remark/outputs/crs/analysis/paper_val_tables.json
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
from crs_eval import DATA, REPO, RG, hits_for, recall_ndcg  # noqa: E402

BUCKETS = {"all": (1, 10**9), "n=1": (1, 1), "n>=5": (5, 10**9), "n=2-4": (2, 4), "n=5-19": (5, 19), "n>=20": (20, 10**9)}


def row_stats(run: Path, ds, catalog, trunc: int) -> tuple[np.ndarray, list, np.ndarray, list]:
    """Per-row R@20, per-GT-item hit@20, whether the list has 20 parsed items, and the (truncated) list per row."""
    r20, gt_hit, full, lists = [], [], [], []
    for r, it in zip(map(json.loads, open(run / "per_instance.jsonl")), ds):
        recs = [(t, int(y)) for t, y in r["recs"]]
        full.append(len(recs) >= 20)
        recs = recs[:trunc]
        lists.append(recs)
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        hits, matched = hits_for(recs, gts, set(it["seen_titles"]), catalog, "truncate")
        r20.append(recall_ndcg(hits, len(gts), 20)[0])
        got = {gi for gi, pos in matched if pos < 20}
        gt_hit.append([(g, int(gi in got)) for gi, g in enumerate(gts)])
    return np.array(r20), gt_hit, np.array(full), lists


def pooled_recall(lists: list, gts_all: list, seen_all: list, groups: list, catalog) -> np.ndarray:
    """Per-context R@20 against the union of all its responders' items (pooled evaluation; one list per context)."""
    out = []
    for g in groups:
        union = list(dict.fromkeys((x[0], int(x[1])) for i in g for x in gts_all[i]))
        hits, _ = hits_for(lists[g[0]], union, set(seen_all[g[0]]), catalog, "truncate")
        out.append(recall_ndcg(hits, len(union), 20)[0])
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    out = {"spec": a.spec, "datasets": {}}
    for dname, d in spec["datasets"].items():
        data = Path(d.get("data", DATA))
        data = data if data.is_absolute() else REPO / data
        catalog_path = Path(d.get("catalog", RG / "reddit_catalog/gt_catalog.pkl"))
        catalog_path = catalog_path if catalog_path.is_absolute() else REPO / catalog_path
        ds = load_from_disk(str(data / d.get("split", "validation")))
        catalog = set(tuple(x) for x in pickle.load(open(catalog_path, "rb")))
        pop = {tuple(json.loads(k)): v for k, v in json.load(open(REPO / d["popularity"])).items()}
        ctx = defaultdict(list)
        for i, p in enumerate(ds["prompt"]):
            ctx[p[0]["content"]].append(i)
        groups = list(ctx.values())
        sizes = np.array([len(g) for g in groups])
        n_row = np.zeros(len(ds), int)
        for g in groups:
            n_row[g] = len(g)
        val_dir = REPO / d.get("val_dir", "remark/outputs/crs/val")
        arms, cache, pooled = {}, {}, {}
        gts_all, seen_all = ds["groundtruth_with_release_year"], ds["seen_titles"]
        for name, arm in d["arms"].items():
            run = val_dir / arm["run"]
            r20, gt_hit, full, lists = row_stats(run, ds, catalog, arm.get("trunc", 20))
            cache[name] = r20
            pooled[name] = pooled_recall(lists, gts_all, seen_all, groups, catalog)
            recs_ctx = [lists[g[0]] for g in groups]
            rec_pop = np.array([pop.get(x, 0) for L in recs_ctx for x in L])
            summ = json.load(open(run / "summary.json"))
            popv = np.array([pop.get(g, 0) for row in gt_hit for g, _ in row])
            hitv = np.array([h for row in gt_hit for _, h in row])
            q50, q80 = np.quantile(popv, [0.5, 0.8])
            arms[name] = {"run": arm["run"], "trunc": arm.get("trunc", 20),
                          "R@20": float(r20.mean()), "N@20": summ["truncate_N@20"] if arm.get("trunc", 20) == 20 else None,
                          "R@10": summ["truncate_R@10"], "N@10": summ["truncate_N@10"],
                          "req_R@20": float(np.mean([r20[g].mean() for g in groups])),
                          "R@20_n1": float(r20[n_row == 1].mean()), "R@20_n5": float(r20[n_row >= 5].mean()),
                          "tail": float(hitv[popv <= q50].mean()), "head": float(hitv[popv > q80].mean()),
                          "exact20": float(full.mean()), "pooled_R@20": float(pooled[name].mean()),
                          "coverage": len({x for L in recs_ctx for x in L}),
                          "rec_mean_pop": float(rec_pop.mean()), "rec_tail_share": float((rec_pop <= q50).mean()),
                          "rec_unseen_share": float((rec_pop == 0).mean())}
            print(dname, name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in arms[name].items()}))
        pairs = []
        for pa, pb in d.get("pairs", []):
            dlt = cache[pa] - cache[pb]
            sums = np.array([dlt[g].sum() for g in groups])
            means = sums / sizes
            res = {"a": pa, "b": pb}
            for bname, (lo, hi) in BUCKETS.items():
                sel = np.where((sizes >= lo) & (sizes <= hi))[0]
                rng = np.random.default_rng(42)
                draws = rng.integers(0, len(sel), size=(2000, len(sel)))
                row_boot = np.array([sums[sel[i]].sum() / sizes[sel[i]].sum() for i in draws])
                req_boot = means[sel][draws].mean(axis=1)
                res[bname] = {"row": [float(sums[sel].sum() / sizes[sel].sum()), *map(float, np.percentile(row_boot, [2.5, 97.5]))],
                              "req": [float(means[sel].mean()), *map(float, np.percentile(req_boot, [2.5, 97.5]))]}
            pd = pooled[pa] - pooled[pb]
            rng = np.random.default_rng(42)
            pboot = pd[rng.integers(0, len(pd), size=(2000, len(pd)))].mean(axis=1)
            res["pooled"] = [float(pd.mean()), *map(float, np.percentile(pboot, [2.5, 97.5]))]
            pairs.append(res)
            print(dname, f"{pa} - {pb}", " ".join(f"{b}: row {v['row'][0]:+.4f} [{v['row'][1]:+.4f}, {v['row'][2]:+.4f}] "
                                                  f"req {v['req'][0]:+.4f} [{v['req'][1]:+.4f}, {v['req'][2]:+.4f}]"
                                                  for b, v in res.items() if isinstance(v, dict)),
                  "pooled {:+.4f} [{:+.4f}, {:+.4f}]".format(*res["pooled"]))
        out["datasets"][dname] = {"data": str(data), "split": d.get("split", "validation"), "rows": len(ds),
                                  "contexts": len(groups), "arms": arms, "pairs": pairs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
