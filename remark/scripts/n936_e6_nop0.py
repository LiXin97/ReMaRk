"""E6 readout (internal check, not pre-registered): ReMaRk without P0/P0c.

E6 = ReMaRk 2B union group s3407 trained without --p0/--p0c (runner7k "--nop0", run 2B_union_group_s3407; Table 2 protocol
otherwise: 2,000 steps, validation every 250, best-validation lock, tested once). Pairing:
- validation (no-P0 on both sides): E6 vs the 09-28 no-P0 Rank-GRPO run 2B_single_group_s3407 (pre-P0 trainer; only its
  validation summaries survive: no checkpoint, never tested), and the same gap under P0c (main-table runs).
- clean pair : 2B_single_group_s3407_lr1e-6 = Rank-GRPO with the current trainer and --nop0 ("--lr 1e-6" is the
  default lr, only a new name); E6 test minus its locked test, its test minus Rank-GRPO main (P0c), validation gap.
- test: E6 minus ReMaRk main (P0c, test_2B_union_group_step2000) and minus Rank-GRPO main (P0c,
  test_2B_single_group_s3407_step2000), per row and per request (crs_compare paired bootstrap over contexts, 2,000
  resamples, seed 42). Both main-table tests come from the _p0c checkpoints (summary.json "model").
Output: outputs/crs/analysis/n936_e6_nop0.json
Usage: env/bin/python remark/scripts/n936_e6_nop0.py
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
OUT = C / "analysis/n936_e6_nop0.json"
E6 = "2B_union_group_s3407"
NOP0_SINGLE = "2B_single_group_s3407_lr1e-6"  # clean pair : current trainer, single group --nop0; "--lr 1e-6" = default
VAL_RUNS = {"remark_nop0": E6, "rankgrpo_nop0": "2B_single_group_s3407", "rankgrpo_nop0_clean": NOP0_SINGLE,
            "remark_p0c": "2B_union_group_s3407_p0c", "rankgrpo_p0c": "2B_single_group_s3407_p0c"}
REMARK_TEST, RANKGRPO_TEST = "test_2B_union_group_step2000", "test_2B_single_group_s3407_step2000"


def curve(run: str) -> dict[int, float]:
    out = {}
    for p in (C / "val").glob(f"grpo_qwen_{run}_step*/summary.json"):
        m = re.search(r"_step(\d+)$", p.parent.name)
        if m:
            out[int(m.group(1))] = json.load(open(p))["truncate_R@20"]
    return dict(sorted(out.items()))


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

    lk = json.load(open(C / "locks" / f"{E6}.json"))
    test = f"test_{E6}_step{lk['best_step']}"
    val = {}
    for k, run in VAL_RUNS.items():
        cv = curve(run)
        b = max(cv, key=lambda s: (cv[s], -s))
        val[k] = {"run": run, "best_val": 100 * cv[b], "best_step": b, "curve": {s: 100 * v for s, v in cv.items()}}
    res = {"note": __doc__.split("Usage")[0].strip(), "units": "points (x100)",
           "model_of_main_tests": {t: json.load(open(C / "test" / t / "summary.json"))["model"] for t in (REMARK_TEST, RANKGRPO_TEST)},
           "e6": {"run": E6, "lock_step": lk["best_step"], "test": test, "test_R@20": 100 * float(r20(test).mean())},
           "val": val,
           "val_gap_union_minus_single": {"nop0": val["remark_nop0"]["best_val"] - val["rankgrpo_nop0"]["best_val"],
                                          "p0c": val["remark_p0c"]["best_val"] - val["rankgrpo_p0c"]["best_val"]},
           "test_e6_minus_remark_p0c": cmp(test, REMARK_TEST),
           "test_e6_minus_rankgrpo_p0c": cmp(test, RANKGRPO_TEST)}
    if (C / "locks" / f"{NOP0_SINGLE}.json").exists():
        tn = f"test_{NOP0_SINGLE}_step{json.load(open(C / 'locks' / f'{NOP0_SINGLE}.json'))['best_step']}"
        res["clean_nop0"] = {"rankgrpo_nop0_test": tn, "rankgrpo_nop0_test_R@20": 100 * float(r20(tn).mean()),
                             "val_gap_union_minus_single": val["remark_nop0"]["best_val"] - val["rankgrpo_nop0_clean"]["best_val"],
                             "test_e6_minus_rankgrpo_nop0": cmp(test, tn), "test_rankgrpo_nop0_minus_rankgrpo_p0c": cmp(tn, RANKGRPO_TEST)}
    OUT.write_text(json.dumps(res, indent=2))
    for k, v in val.items():
        print(f"val {k:14s} {v['run']:28s} best {v['best_val']:.2f} @ {v['best_step']}")
    g = res["val_gap_union_minus_single"]
    print(f"val gap union-single: no-P0 {g['nop0']:+.2f}, P0c {g['p0c']:+.2f}")
    print(f"E6 test R@20 {res['e6']['test_R@20']:.2f} ({test})")
    for k, t in [(k, res[k]) for k in ("test_e6_minus_remark_p0c", "test_e6_minus_rankgrpo_p0c")] + \
            [(k, res.get("clean_nop0", {}).get(k)) for k in ("test_e6_minus_rankgrpo_nop0", "test_rankgrpo_nop0_minus_rankgrpo_p0c")]:
        if t is None:
            continue
        print(f"{k}: row {t['row']['diff']:+.2f} [{t['row']['ci_low']:+.2f},{t['row']['ci_high']:+.2f}] "
              f"unit {t['unit']['diff']:+.2f} [{t['unit']['ci_low']:+.2f},{t['unit']['ci_high']:+.2f}]")


if __name__ == "__main__":
    main()
