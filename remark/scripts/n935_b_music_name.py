"""Reddit-Music artist-name-only matching.

Re-scores the Table 2 Music pair test outputs (crs_eval per_instance.jsonl, no regeneration) with the registered truncate
protocol except that a recommendation matches a ground-truth artist on the name alone (the |year diff| <= 2 test is
dropped). Seen and out-of-catalog filtering also use names: a recommended name that the user already mentioned, or that
is no catalog artist under any year, counts as a miss. Ground-truth artists are deduplicated by name per row. Per-row and
per-request (mean over contexts of the within-context mean) R@20 / N@20 with the crs_compare paired bootstrap (contexts,
2,000 resamples, seed 42). The registered year-matching numbers are recomputed alongside as a check.
Usage: env/bin/python remark/scripts/n935_b_music_name.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_compare import context_groups, paired_stats  # noqa: E402
from crs_eval import hits_for, recall_ndcg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
D = REPO / "remark/data/crs/music"
T = REPO / "remark/outputs/crs/test"
RUNS = {"remark": T / "test_music_2B_union_group_s3407_p0c_ex_step1250",
        "rankgrpo": T / "test_music_2B_single_group_s3407_p0c_ex_step2000"}
OUT = REPO / "remark/outputs/crs/analysis/n935_b_music_name.json"
K = 20


def hits_name(recs, gt_names, seen, cat_names):
    hits = np.zeros(len(recs), int)
    for g in gt_names:
        for i, r in enumerate(recs):
            if r[0] in seen or r[0] not in cat_names:
                continue
            if hits[i] == 0 and r[0] == g:
                hits[i] = 1
                break
    return hits


def main() -> None:
    ds = load_from_disk(str(D / "sft_dataset" / "test"))
    catalog = set(tuple(x) for x in pickle.load(open(D / "catalog.pkl", "rb")))
    cat_names = {t for t, _ in catalog}
    groups = list(context_groups([p[0]["content"] for p in ds["prompt"]]).values())
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    per, n_dup = {}, 0
    for arm, run in RUNS.items():
        rows = [json.loads(line) for line in open(run / "per_instance.jsonl")]
        assert len(rows) == len(ds)
        m = {f"{v}_{x}@{K}": np.zeros(len(ds)) for v in ("name", "year") for x in ("R", "N")}
        for r, it in zip(rows, ds):
            recs = [(t, int(y)) for t, y in r["recs"]][:K]
            gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
            seen = set(it["seen_titles"])
            names = list(dict.fromkeys(g[0] for g in gts))
            n_dup += arm == "remark" and len(names) < len(gts)
            rk, nk = recall_ndcg(hits_name(recs, names, seen, cat_names), len(names), K)
            m[f"name_R@{K}"][r["i"]], m[f"name_N@{K}"][r["i"]] = rk, nk
            h, _ = hits_for(recs, gts, seen, catalog, "truncate")
            rk, nk = recall_ndcg(h, len(gts), K)
            m[f"year_R@{K}"][r["i"]], m[f"year_N@{K}"][r["i"]] = rk, nk
        per[arm] = m
        reg = json.load(open(run / "summary.json"))["truncate_R@20"]
        assert abs(m[f"year_R@{K}"].mean() - reg) < 1e-9, (arm, m[f"year_R@{K}"].mean(), reg)
    res = {}
    for key in per["remark"]:
        for w in ("row", "unit"):
            st = paired_stats(per["remark"][key], per["rankgrpo"][key], groups, draws, w)
            res[f"{key}_{w}"] = {k: (100 * v if k != "p_le_0" else v) for k, v in st.items()}
    out = {"note": __doc__.split("Usage")[0].strip(), "runs": {k: str(v.relative_to(REPO)) for k, v in RUNS.items()},
           "n_rows": len(ds), "n_contexts": len(groups), "rows_with_duplicate_gt_names": int(n_dup),
           "units": "points (x100); a = ReMaRk, b = Rank-GRPO, diff = a - b, 95% paired bootstrap CI", "results": res}
    OUT.write_text(json.dumps(out, indent=2))
    for k, v in res.items():
        print(f"{k:16s} ReMaRk {v['a']:.2f}  Rank-GRPO {v['b']:.2f}  diff {v['diff']:+.2f} [{v['ci_low']:+.2f},{v['ci_high']:+.2f}]")


if __name__ == "__main__":
    main()
