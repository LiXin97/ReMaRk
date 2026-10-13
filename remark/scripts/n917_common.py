"""Shared scoring for baselines: turn raw generations per test context into crs_eval-format outputs
(per_instance.jsonl + summary.json), with exactly crs_eval.py's parser (Rank-GRPO process_rec_raw) and matching.

score_and_write(ds, raw_by_ctx, catalog_path, data_path, out_dir, extra) -> summary dict
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import DATA, KS, REPO, hits_for, process_rec_raw, recall_ndcg  # noqa: E402


def parse(raw: str) -> list[tuple[str, int]]:
    item = {"raw": raw}
    err_item = process_rec_raw(item, "raw", "rec")
    recs = err_item[1]["rec"] if isinstance(err_item, tuple) else item["rec"]
    return [(t, int(y)) for t, y in recs][:20]


def score_and_write(ds, raw_by_ctx: dict[str, str], catalog_path, data_path, out_dir, extra: dict | None = None) -> dict:
    catalog = set(tuple(x) for x in pickle.load(open(catalog_path, "rb")))
    tag = "" if Path(data_path) == DATA else "_" + Path(data_path).parent.name
    pop_path = REPO / f"remark/outputs/crs/train_gt_popularity{tag}.json"
    pop = Counter({tuple(json.loads(k)): v for k, v in json.load(open(pop_path)).items()}) if pop_path.exists() else Counter()
    parsed = {c: parse(r) for c, r in raw_by_ctx.items()}
    res = {m: {k: [[], []] for k in KS} for m in ("truncate", "filter")}
    item_level, per = [], []
    n_parse_err, n_in_cat, n_recs, n_missing = 0, 0, 0, 0
    for i, it in enumerate(ds):
        c = it["prompt"][0]["content"]
        if c not in parsed:
            n_missing += 1
        recs = parsed.get(c, [])
        n_recs += len(recs)
        n_in_cat += sum(r in catalog for r in recs)
        if not recs:
            n_parse_err += 1
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        seen = set(it["seen_titles"])
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
    summary = {"data": str(data_path), "split": "test", "n": len(ds), "unique_contexts": len(raw_by_ctx),
               "rows_without_output": n_missing, "parse_empty": n_parse_err,
               "mean_parsed_items": n_recs / max(1, len(ds)), "in_catalog_rate": n_in_cat / max(1, n_recs)}
    summary.update(extra or {})
    for m in res:
        for k in KS:
            summary[f"{m}_R@{k}"] = float(np.mean(res[m][k][0]))
            summary[f"{m}_N@{k}"] = float(np.mean(res[m][k][1]))
    if item_level:
        popv = np.array([p for p, _ in item_level]); hitv = np.array([h for _, h in item_level])
        qs = np.quantile(popv, [0.5, 0.8, 0.9])
        buckets = {"unseen(0)": popv == 0, "tail(<=median)": popv <= qs[0], "mid": (popv > qs[0]) & (popv <= qs[1]),
                   "head(top20%)": popv > qs[1], "top10%": popv > qs[2]}
        summary["item_recall@20_by_pop"] = {b: {"n": int(msk.sum()), "recall": float(hitv[msk].mean()) if msk.any() else None}
                                            for b, msk in buckets.items()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "per_instance.jsonl", "w") as f:
        for r in per:
            f.write(json.dumps(r) + "\n")
    return summary
