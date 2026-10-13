"""readouts for Table 5 (4B stronger-baseline controls) and the 35B-A3B batch-scaling cells.

Table 5 (4B Reddit-v2, s3407, Table 2 protocol): ReMaRk (test_4B_union_group_s3407_step2000) minus Rank-GRPO with
DAPO-style dynamic sampling (4B_single_group_s3407_p0c_dyn, --dyn_sample 3) and minus Rank-GRPO with request-balanced
sampling (4B_single_group_s3407_p0c_rb); with G = 16 on both sides: union G16 minus single G16. Each strengthened
Rank-GRPO arm is also compared with plain Rank-GRPO 4B (test_4B_single_group_s3407_step2000).
35B-A3B (new SFT qwen35_35BA3B_r2 step 1500, batch scaling, s3407): per-responder batch (35BA3B_single_batch_s3407_p0c),
union batch (35BA3B_union_batch_s3407_p0c, once locked): "Delta batch scaling" = union batch - single batch, and union batch
minus the old Rank-GRPO 35B (test_35BA3B_single_group_s3407_p0c_step2000, old SFT; appendix). New vs old SFT test.
27B (10-09, same recipe): new SFT qwen35_27B_r2 step 1500 (old SFT qwen35_27B step 1000 was deleted 10-05), 27B_{single,union}_batch_s3407_p0c;
Delta batch scaling, each batch arm minus the old Rank-GRPO / ReMaRk 27B (old SFT), new minus old SFT test.
All test deltas per row and per request (crs_compare paired bootstrap over contexts, 2,000 resamples, seed 42).
Usage: env/bin/python remark/scripts/n936_tab5_35b.py
Output: outputs/crs/analysis/n936_tab5_35b.json
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_compare import DATA, RG, context_groups, paired_stats, per_row  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
OUT = C / "analysis/n936_tab5_35b.json"
RG4, REMARK4 = "test_4B_single_group_s3407_step2000", "test_4B_union_group_s3407_step2000"
OLD_RG27, OLD_REMARK27 = "test_27B_single_group_s3407_p0c_step2000", "test_27B_union_group_s3407_p0c_step2000"
NEW_SFT27 = "test_27B_r2_sft_step1500"
OLD_RG35, OLD_REMARK35 = "test_35BA3B_single_group_s3407_p0c_step2000", "test_35BA3B_union_group_s3407_p0c_step2000"


def curve(run: str) -> dict[int, float]:
    out = {}
    for p in (C / "val").glob(f"grpo_qwen_{run}_step*/summary.json"):
        m = re.search(r"_step(\d+)$", p.parent.name)
        if m:
            out[int(m.group(1))] = 100 * json.load(open(p))["truncate_R@20"]
    return dict(sorted(out.items()))


def locked(run: str) -> str | None:
    lk = C / "locks" / f"{run}.json"
    return f"test_{run}_step{json.load(open(lk))['best_step']}" if lk.exists() else None


def main() -> None:
    ds = load_from_disk(str(Path(DATA) / "test"))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    groups = list(context_groups([p[0]["content"] for p in ds["prompt"]]).values())
    draws = np.random.default_rng(42).integers(0, len(groups), size=(2000, len(groups)))
    cache: dict[str, np.ndarray] = {}

    def r20(name: str) -> np.ndarray:
        if name not in cache:
            cache[name] = per_row(C / "test" / name, ds, catalog)["R@20"]
        return cache[name]

    def cmp(a: str, b: str) -> dict:
        return {w: {k: (100 * v if k != "p_le_0" else v) for k, v in paired_stats(r20(a), r20(b), groups, draws, w).items()}
                for w in ("row", "unit")}

    res: dict = {"note": __doc__.split("Usage")[0].strip(), "units": "points (x100)", "pairs": {}, "runs": {}}
    pairs = {"tab5/remark_minus_dyn": (REMARK4, locked("4B_single_group_s3407_p0c_dyn")),
             "tab5/remark_minus_rb": (REMARK4, locked("4B_single_group_s3407_p0c_rb")),
             "tab5/union_G16_minus_single_G16": (locked("4B_union_group_s3407_p0c_G16"), locked("4B_single_group_s3407_p0c_G16")),
             "tab5/dyn_minus_rankgrpo": (locked("4B_single_group_s3407_p0c_dyn"), RG4),
             "tab5/rb_minus_rankgrpo": (locked("4B_single_group_s3407_p0c_rb"), RG4),
             "tab5/single_G16_minus_rankgrpo": (locked("4B_single_group_s3407_p0c_G16"), RG4),
             "35B/single_batch_minus_old_rankgrpo": (locked("35BA3B_single_batch_s3407_p0c"), OLD_RG35),
             "35B/union_batch_minus_single_batch": (locked("35BA3B_union_batch_s3407_p0c"), locked("35BA3B_single_batch_s3407_p0c")),
             "35B/union_batch_minus_old_rankgrpo": (locked("35BA3B_union_batch_s3407_p0c"), OLD_RG35),
             "35B/new_sft_minus_old_sft": ("test_35BA3B_r2_sft_step1500", "test_35BA3B_sft_step1500"),
             "27B/union_batch_minus_single_batch": (locked("27B_union_batch_s3407_p0c"), locked("27B_single_batch_s3407_p0c")),
             "27B/single_batch_minus_old_rankgrpo": (locked("27B_single_batch_s3407_p0c"), OLD_RG27),
             "27B/union_batch_minus_old_rankgrpo": (locked("27B_union_batch_s3407_p0c"), OLD_RG27),
             "27B/union_batch_minus_old_remark": (locked("27B_union_batch_s3407_p0c"), OLD_REMARK27),
             "27B/new_sft_minus_old_sft": (NEW_SFT27 if (C / "test" / NEW_SFT27 / "summary.json").exists() else None,
                                           "test_27B_sft_step1000")}
    for run in ("4B_single_group_s3407_p0c_dyn", "4B_single_group_s3407_p0c_rb", "4B_union_group_s3407_p0c_G16",
                "4B_single_group_s3407_p0c_G16", "35BA3B_single_batch_s3407_p0c", "35BA3B_union_batch_s3407_p0c",
                "27B_single_batch_s3407_p0c", "27B_union_batch_s3407_p0c"):
        cv = curve(run)
        res["runs"][run] = {"test": locked(run), "best_val": max(cv.values()) if cv else None, "val_curve": cv}
    for k, (a, b) in pairs.items():
        if a is None or b is None:
            res["pairs"][k] = None
            print(f"{k}: pending")
            continue
        r = res["pairs"][k] = {"a": a, "b": b, **cmp(a, b)}
        print(f"{k}: {r['row']['a']:.2f} - {r['row']['b']:.2f} = row {r['row']['diff']:+.2f} [{r['row']['ci_low']:+.2f},"
              f"{r['row']['ci_high']:+.2f}] unit {r['unit']['diff']:+.2f} [{r['unit']['ci_low']:+.2f},{r['unit']['ci_high']:+.2f}]")
    OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
