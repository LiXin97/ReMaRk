"""E4 readout (internal check, not pre-registered): ordinary sequence-level GRPO (--seq_level), 2B Reddit-v2, s3407.

One scalar reward per list (recall@20), group-scaled, applied to every completion token; per-responder arm
(2B_single_group_s3407_p0c_seq, rows sampled per responder) and union arm (2B_union_group_s3407_p0c_seq, union of the
responders' items, requests sampled). Training curves (seq/recall, seq/frac_group_zero_std, rank/frac_exact_len every 10
steps) from the trainer_state.json log history of each locked checkpoint (the node training log of the per-responder arm
was lost with its node). Test R@20: union - per-responder, and each arm minus the main-table Rank-GRPO 2B and ReMaRk 2B
(per row and per request, crs_compare paired bootstrap over contexts, 2,000 resamples, seed 42).
Output: outputs/crs/analysis/n936_e4_seq.json
Usage: env/bin/python remark/scripts/n936_e4_seq.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_compare import DATA, RG, context_groups, paired_stats, per_row  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
OUT = C / "analysis/n936_e4_seq.json"
ARMS = {"per_responder": "2B_single_group_s3407_p0c_seq", "union": "2B_union_group_s3407_p0c_seq"}
RANKGRPO, REMARK = "test_2B_single_group_s3407_step2000", "test_2B_union_group_step2000"
KEYS = ("seq/recall", "seq/frac_group_zero_std", "rank/frac_exact_len", "kl", "completions/mean_length")


def main() -> None:
    ds = load_from_disk(str(Path(DATA) / "test"))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    groups = list(context_groups([p[0]["content"] for p in ds["prompt"]]).values())
    draws = np.random.default_rng(42).integers(0, len(groups), size=(2000, len(groups)))
    cache: dict[str, np.ndarray] = {}

    def r20(n: str) -> np.ndarray:
        if n not in cache:
            cache[n] = per_row(C / "test" / n, ds, catalog)["R@20"]
        return cache[n]

    def cmp(a: str, b: str) -> dict:
        return {w: {k: (100 * v if k != "p_le_0" else v) for k, v in paired_stats(r20(a), r20(b), groups, draws, w).items()}
                for w in ("row", "unit")}

    res = {"note": __doc__.split("Usage")[0].strip(), "units": "test values in points (x100)", "arms": {}, "test": {}}
    tests = {}
    for arm, run in ARMS.items():
        lk = json.load(open(C / "locks" / f"{run}.json"))
        st = lk["best_step"]
        tests[arm] = f"test_{run}_step{st}"
        hist = json.load(open(C / "grpo_qwen_selected" / run / f"checkpoint-{st}" / "trainer_state.json"))["log_history"]
        curves = {k: {int(h["step"]): float(h[k]) for h in hist if k in h} for k in KEYS}
        def window(k, lo, hi):
            v = [x for s, x in curves[k].items() if lo < s <= hi]
            return float(np.mean(v)) if v else None
        res["arms"][arm] = {"run": run, "lock_step": st, "val_R@20": lk["val_R@20"], "test": tests[arm],
                            "curve_points": len(curves["seq/recall"]), "curves": curves,
                            "mean_first_100": {k: window(k, 0, 100) for k in KEYS},
                            "mean_last_100": {k: window(k, st - 100, st) for k in KEYS}}
    res["test"]["union - per_responder"] = cmp(tests["union"], tests["per_responder"])
    for arm in ARMS:
        res["test"][f"{arm} - Rank-GRPO"] = cmp(tests[arm], RANKGRPO)
        res["test"][f"{arm} - ReMaRk"] = cmp(tests[arm], REMARK)
    OUT.write_text(json.dumps(res, indent=2))
    fmt = lambda d: f"{d['diff']:+.2f} [{d['ci_low']:+.2f},{d['ci_high']:+.2f}]"
    for arm, e in res["arms"].items():
        print(arm, "lock", e["lock_step"], "best val", round(max(e["val_R@20"].values()), 4), "points", e["curve_points"],
              "first100", {k: round(v, 4) for k, v in e["mean_first_100"].items() if v is not None},
              "last100", {k: round(v, 4) for k, v in e["mean_last_100"].items() if v is not None})
    for k, v in res["test"].items():
        print(f"{k}: {v['row']['a']:.2f} vs {v['row']['b']:.2f}; row {fmt(v['row'])} unit {fmt(v['unit'])}")


if __name__ == "__main__":
    main()
