"""R@5, R@10, R@20, N@20 and Hit@20 (per row and per request) for the main test arms, recomputed from
existing crs_eval outputs (per_instance.jsonl) with crs_eval's truncate matching (seen or out-of-catalog items are misses,
title exact, year within 2). Paired bootstrap over requests (2,000, seed 42) for ReMaRk - Rank-GRPO at each size.

Usage: env/bin/python remark/scripts/n917_e6_metrics.py
Output: remark/outputs/crs/analysis/n917_e6_topk_metrics.json (+ manifest)
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
from crs_eval import DATA, REPO, RG, hits_for, recall_ndcg  # noqa: E402

T = REPO / "remark/outputs/crs/test"
OUT = REPO / "remark/outputs/crs/analysis/n917_e6_topk_metrics.json"
BOOKS = REPO / "remark/data/crs/reddit_books"
SPECS = {
    "reddit": {"data": DATA, "catalog": RG / "reddit_catalog/gt_catalog.pkl", "arms": {
        "released_rankgrpo_llama": "../rankgrpo_llama3b_test", "popularity": "popularity_train_top20",
        "2B_sft": "test_2B_sft_step1500", "2B_rankgrpo": "test_2B_single_group_s3407_step2000",
        "2B_perresp_batch": "test_2B_single_batch_s3407_step1750", "2B_remark": "test_2B_union_s3407_step1500",
        "2B_union_group": "test_2B_union_group_step2000",
        "4B_sft": "test_4B_sft_step1000", "4B_rankgrpo": "test_4B_single_group_s3407_step2000",
        "4B_perresp_batch": "test_4B_single_batch_s3407_step2000", "4B_remark": "test_4B_union_s3407_step2000",
        "4B_union_group": "test_4B_union_group_s3407_step2000",
        "9B_sft": "test_9B_sft_step1000", "9B_rankgrpo": "test_9B_single_group_s3407_step2000",
        "9B_perresp_batch": "test_9B_single_batch_s3407_step2000", "9B_remark": "test_9B_union_s3407_step2000",
        "9B_union_group": "test_9B_union_group_s3407_p0c_step2000",
        "knn_prior_k200": "test_knn_prior_k200", "zeroshot_gpt56": "test_zeroshot_gpt56",
        "zeroshot_qwen35_27B": "test_zeroshot_qwen35_27B"},
        "pairs": [("2B_remark", "2B_rankgrpo"), ("4B_remark", "4B_rankgrpo"), ("9B_remark", "9B_rankgrpo"),
                  # final configuration (union + group scaling); "remark" above = the registered union + batch arm
                  ("2B_union_group", "2B_rankgrpo"), ("4B_union_group", "4B_rankgrpo"), ("9B_union_group", "9B_rankgrpo")]},
    "books": {"data": BOOKS / "sft_dataset", "catalog": BOOKS / "catalog.pkl", "arms": {
        "books_sft": "test_books_sft_step1000", "books_rankgrpo": "test_books_single_group_p0d_step2000",
        "books_perresp_batch": "test_books_single_batch_p0d_step2000", "books_remark": "test_books_union_p0d_step2000",
        "books_union_group": "test_books_union_group_p0d_step1500", "books_knn_prior_k200": "test_knn_prior_k200_books",
        "books_zeroshot_gpt56": "test_zeroshot_gpt56_books", "books_zeroshot_qwen35_27B": "test_zeroshot_qwen35_27B_books"},
        "pairs": [("books_remark", "books_perresp_batch"), ("books_remark", "books_rankgrpo"),
                  ("books_union_group", "books_perresp_batch")]},
}
KS = (5, 10, 20)


def row_metrics(run: Path, ds, catalog) -> dict[str, np.ndarray]:
    out = defaultdict(list)
    for r, it in zip(map(json.loads, open(run / "per_instance.jsonl")), ds):
        recs = [(t, int(y)) for t, y in r["recs"]][:20]
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        hits, _ = hits_for(recs, gts, set(it["seen_titles"]), catalog, "truncate")
        for k in KS:
            rk, nk = recall_ndcg(hits, len(gts), k)
            out[f"R@{k}"].append(rk)
            out[f"N@{k}"].append(nk)
        out["Hit@20"].append(float(hits[:20].sum() > 0))
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    res = {"manifest": {"script": __file__, "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                                    text=True, cwd=REPO).stdout.strip()}}
    for dname, spec in SPECS.items():
        ds = load_from_disk(str(Path(spec["data"]) / "test"))
        catalog = set(tuple(x) for x in pickle.load(open(spec["catalog"], "rb")))
        ctx = defaultdict(list)
        for i, p in enumerate(ds["prompt"]):
            ctx[p[0]["content"]].append(i)
        groups = list(ctx.values())
        sizes = np.array([len(g) for g in groups])
        cache, arms = {}, {}
        for name, run in spec["arms"].items():
            if not (T / run / "per_instance.jsonl").exists():
                print("missing", run)
                continue
            m = row_metrics(T / run, ds, catalog)
            cache[name] = m
            arms[name] = {"run": run, **{f"{k}_row": float(v.mean()) for k, v in m.items()},
                          **{f"{k}_req": float(np.mean([v[g].mean() for g in groups])) for k, v in m.items()}}
            print(dname, name, {k: round(v, 4) for k, v in arms[name].items() if k != "run"})
        pairs = []
        rng_draws = np.random.default_rng(42).integers(0, len(groups), size=(2000, len(groups)))
        for a, b in spec["pairs"]:
            if a not in cache or b not in cache:
                continue
            res_p = {"a": a, "b": b}
            for k in cache[a]:
                d = cache[a][k] - cache[b][k]
                sums = np.array([d[g].sum() for g in groups])
                means = sums / sizes
                row_boot = sums[rng_draws].sum(1) / sizes[rng_draws].sum(1)
                req_boot = means[rng_draws].mean(1)
                res_p[k] = {"row": [float(sums.sum() / sizes.sum()), *map(float, np.percentile(row_boot, [2.5, 97.5]))],
                            "req": [float(means.mean()), *map(float, np.percentile(req_boot, [2.5, 97.5]))]}
            pairs.append(res_p)
            print(dname, a, "-", b, {k: [round(x, 4) for x in v["row"]] for k, v in res_p.items() if isinstance(v, dict)})
        res[dname] = {"rows": len(ds), "requests": len(groups), "arms": arms, "pairs": pairs}
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
