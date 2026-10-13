"""Memorization stratification. Per size (2B/4B/9B), a test request is "recalled from the request
alone" if the base-model probe (n917_e8_memo_probe.py, outputs/crs/test/memo_probe_<size>) lists >= 1 item of any of the
request's responders (crs_eval truncate matching). Reports the share of recalled requests, the probe's own R@20, and
ReMaRk - Rank-GRPO per-row R@20 (test, seed 3407 primary pair) within each stratum, with paired bootstrap CIs over
requests (2,000, seed 42).
Usage: env/bin/python remark/scripts/n917_e8_stratify.py
Output: outputs/crs/analysis/n917_e8_memorization.json
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
from crs_eval import DATA, REPO, RG, hits_for, recall_ndcg  # noqa: E402

T = REPO / "remark/outputs/crs/test"
# ReMaRk = union reward + per-group scaling (final method); the registered union+batch pairs are kept as "<size>_union_batch".
PAIRS = {"2B": ("test_2B_union_group_step2000", "test_2B_single_group_s3407_step2000"),
         "4B": ("test_4B_union_group_s3407_step2000", "test_4B_single_group_s3407_step2000"),
         "9B": ("test_9B_union_group_s3407_p0c_step2000", "test_9B_single_group_s3407_step2000"),
         "2B_union_batch": ("test_2B_union_s3407_step1500", "test_2B_single_group_s3407_step2000"),
         "4B_union_batch": ("test_4B_union_s3407_step2000", "test_4B_single_group_s3407_step2000"),
         "9B_union_batch": ("test_9B_union_s3407_step2000", "test_9B_single_group_s3407_step2000")}


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
    ds = load_from_disk(str(DATA / "test"))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    ctx = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx[p[0]["content"]].append(i)
    groups = list(ctx.values())
    sizes = np.array([len(g) for g in groups])
    out = {}
    for size, (a, b) in PAIRS.items():
        probe = T / f"memo_probe_{size.split('_')[0]}"
        if not (probe / "per_instance.jsonl").exists():
            print("missing", probe)
            continue
        pr20, phit = row_r20(probe, ds, catalog)
        recalled = np.array([phit[g].any() for g in groups])
        ra, _ = row_r20(T / a, ds, catalog)
        rb, _ = row_r20(T / b, ds, catalog)
        d = ra - rb
        sums = np.array([d[g].sum() for g in groups])
        res = {"probe_R@20_row": float(pr20.mean()), "share_requests_recalled": float(recalled.mean()),
               "rows_in_recalled_requests": float(sizes[recalled].sum() / sizes.sum())}
        for name, mask in (("recalled", recalled), ("not_recalled", ~recalled), ("all", np.ones_like(recalled))):
            sel = np.where(mask)[0]
            draws = np.random.default_rng(42).integers(0, len(sel), size=(2000, len(sel)))
            boot = np.array([sums[sel[i]].sum() / sizes[sel[i]].sum() for i in draws])
            res[name] = {"requests": int(len(sel)), "remark_minus_rankgrpo_row": float(sums[sel].sum() / sizes[sel].sum()),
                         "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])]}
        out[size] = res
        print(size, json.dumps(res))
    (REPO / "remark/outputs/crs/analysis/n917_e8_memorization.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
