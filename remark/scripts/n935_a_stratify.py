"""Memorization stratification on Reddit-Books / Reddit-Music, the method
(n917_e8_stratify.py) with that dataset's test split, catalog and matching. Probe outputs: outputs/crs/test/
memo_probe_{books,music}_<size> (n935_a_memo_probe.py). A test request is "recalled" if the base-model probe lists >= 1
item of any responder; per stratum ReMaRk - Rank-GRPO per-row R@20 of the Table 2 seed-3407 pair of that column, paired
bootstrap over requests (2,000, seed 42). Books also reports ReMaRk - the per-responder batch-scaled arm (Table 2's Delta
baseline for Books).
Usage: env/bin/python remark/scripts/n935_a_stratify.py
Output: outputs/crs/analysis/n935_a_memorization.json
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import REPO, hits_for, recall_ndcg  # noqa: E402

T = REPO / "remark/outputs/crs/test"
# setting -> (dataset, probe, {comparison: (ReMaRk test dir, baseline test dir)}); pairs = make_paper_stats TEST
SETTINGS = {
    "books_2B": ("reddit_books", "memo_probe_books_2B",
                 {"vs_rankgrpo": ("test_books_union_group_p0d_step1500", "test_books_single_group_p0d_step2000"),
                  "vs_perresp_batch": ("test_books_union_group_p0d_step1500", "test_books_single_batch_p0d_step2000")}),
    "books_4B": ("reddit_books", "memo_probe_books_4B",
                 {"vs_rankgrpo": ("test_reddit_books_4B_union_group_s3407_p0c_cap384_ex_step2000",
                                  "test_reddit_books_4B_single_group_s3407_p0c_cap384_ex_step2000"),
                  "vs_perresp_batch": ("test_reddit_books_4B_union_group_s3407_p0c_cap384_ex_step2000",
                                       "test_reddit_books_4B_single_batch_s3407_p0c_cap384_ex_step2000")}),
    "music_2B": ("music", "memo_probe_music_2B",
                 {"vs_rankgrpo": ("test_music_2B_union_group_s3407_p0c_ex_step1250",
                                  "test_music_2B_single_group_s3407_p0c_ex_step2000")}),
}


def row_r20(run: Path, ds, catalog) -> tuple[np.ndarray, np.ndarray]:
    r20, anyhit = [], []
    for r, it in zip(map(json.loads, open(run / "per_instance.jsonl")), ds):
        recs = [(t, int(y)) for t, y in r["recs"]][:20]
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        hits, _ = hits_for(recs, gts, set(it["seen_titles"]), catalog, "truncate")
        r20.append(recall_ndcg(hits, len(gts), 20)[0])
        anyhit.append(hits[:20].sum() > 0)
    return np.array(r20), np.array(anyhit)


def main() -> None:
    out = {}
    for setting, (dsname, probe_name, comps) in SETTINGS.items():
        D = REPO / f"remark/data/crs/{dsname}"
        ds = load_from_disk(str(D / "sft_dataset" / "test"))
        catalog = set(tuple(x) for x in pickle.load(open(D / "catalog.pkl", "rb")))
        ctx = defaultdict(list)
        for i, p in enumerate(ds["prompt"]):
            ctx[p[0]["content"]].append(i)
        groups = list(ctx.values())
        sizes = np.array([len(g) for g in groups])
        probe = T / probe_name
        if not (probe / "per_instance.jsonl").exists():
            print("missing", probe)
            continue
        pr20, phit = row_r20(probe, ds, catalog)
        recalled = np.array([phit[g].any() for g in groups])
        res = {"probe": str(probe.relative_to(REPO)), "probe_R@20_row": float(pr20.mean()),
               "share_requests_recalled": float(recalled.mean()), "requests": int(len(groups)),
               "rows_in_recalled_requests": float(sizes[recalled].sum() / sizes.sum())}
        for comp, (a, b) in comps.items():
            ra, _ = row_r20(T / a, ds, catalog)
            rb, _ = row_r20(T / b, ds, catalog)
            d = ra - rb
            sums = np.array([d[g].sum() for g in groups])
            res[comp] = {"a": a, "b": b}
            for name, mask in (("recalled", recalled), ("not_recalled", ~recalled), ("all", np.ones_like(recalled))):
                sel = np.where(mask)[0]
                draws = np.random.default_rng(42).integers(0, len(sel), size=(2000, len(sel)))
                boot = np.array([sums[sel[i]].sum() / sizes[sel[i]].sum() for i in draws])
                res[comp][name] = {"requests": int(len(sel)),
                                   "remark_minus_baseline_row": float(sums[sel].sum() / sizes[sel].sum()),
                                   "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])]}
        out[setting] = res
        print(setting, json.dumps(res))
    (REPO / "remark/outputs/crs/analysis/n935_a_memorization.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
