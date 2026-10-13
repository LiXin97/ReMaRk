"""E5 readout (internal check, not pre-registered): ReMaRk 2B learning-rate control.

ReMaRk (2B_union_group_s3407_p0c) at lr 5e-7 and 2e-6 (_lr5e-7, _lr2e-6; Table 2 protocol: 2,000 steps, P0c, validation
every 250 steps, best-validation lock, tested once) next to the main run at 1e-6, and Rank-GRPO's validation-only lr
selection (locks/2B_rankgrpo_lr_selected.json). Test: each lr arm minus ReMaRk at 1e-6 and minus Rank-GRPO at its selected
lr (1e-6), per row and per request (crs_compare paired bootstrap over contexts, 2,000 resamples, seed 42).
Output: outputs/crs/analysis/n936_e5_lr.json
Usage: env/bin/python remark/scripts/n936_e5_lr.py
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
OUT = C / "analysis/n936_e5_lr.json"
REMARK = {"1e-6": ("2B_union_group_s3407_p0c", "test_2B_union_group_step2000"),
          "5e-7": ("2B_union_group_s3407_p0c_lr5e-7", None), "2e-6": ("2B_union_group_s3407_p0c_lr2e-6", None)}
RANKGRPO_TEST = "test_2B_single_group_s3407_step2000"


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

    sel = json.load(open(C / "locks/2B_rankgrpo_lr_selected.json"))
    res = {"note": __doc__.split("Usage")[0].strip(), "units": "test values in points (x100)",
           "rankgrpo_lr_selection": [{k: c[k] for k in ("lr", "run", "best_step", "best_val")} for c in sel["candidates"]],
           "remark": {}}
    for lr, (run, test) in REMARK.items():
        cv = curve(run)
        lk = json.load(open(C / "locks" / f"{run}.json")) if (C / "locks" / f"{run}.json").exists() else None
        test = test or f"test_{run}_step{lk['best_step']}"
        e = {"run": run, "test": test, "best_val": max(cv.values()), "best_val_step": max(cv, key=lambda k: (cv[k], -k)),
             "val_curve": cv, "lock_step": lk["best_step"] if lk else None}
        if lr != "1e-6":
            e["test_minus_remark_1e-6"] = cmp(test, REMARK["1e-6"][1])
        e["test_minus_rankgrpo"] = cmp(test, RANKGRPO_TEST)
        res["remark"][lr] = e
    OUT.write_text(json.dumps(res, indent=2))
    for c in res["rankgrpo_lr_selection"]:
        print(f"Rank-GRPO lr {c['lr']:g}: best val {c['best_val']:.4f} @ {c['best_step']}")
    for lr, e in res["remark"].items():
        t = e["test_minus_rankgrpo"]
        s = f"ReMaRk lr {lr}: best val {e['best_val']:.4f} @ {e['best_val_step']} (lock {e['lock_step']}); test {t['row']['a']:.2f}; " \
            f"- RankGRPO row {t['row']['diff']:+.2f} [{t['row']['ci_low']:+.2f},{t['row']['ci_high']:+.2f}] unit {t['unit']['diff']:+.2f} " \
            f"[{t['unit']['ci_low']:+.2f},{t['unit']['ci_high']:+.2f}]"
        if "test_minus_remark_1e-6" in e:
            u = e["test_minus_remark_1e-6"]
            s += f"; - ReMaRk 1e-6 row {u['row']['diff']:+.2f} [{u['row']['ci_low']:+.2f},{u['row']['ci_high']:+.2f}] unit " \
                 f"{u['unit']['diff']:+.2f} [{u['unit']['ci_low']:+.2f},{u['unit']['ci_high']:+.2f}]"
        print(s)


if __name__ == "__main__":
    main()
