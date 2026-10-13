"""E1 isolation readout (internal check, not pre-registered).

Question: does the dose-response of Fig 4b vanish under group scaling because P0's shaping_scale grows with m?
(b) fixed advantage scale (--adv_norm fixed --adv_scale 0.130) in every arm, s3407: m = 1 / 2 / 4 / all =
    2B_{single,meanm2rw,meanm4rw,consensusrw}_fixed_s3407_p0c. Fig 4b readout via crs_dose_response.py (validation per-row
    R@20 at best step and at step 2,000, OLS slope on x = 1 - rho(m), paired bootstrap over validation contexts, 2,000
    resamples, seed 42; n=1-only slope) + each arm's locked test minus Rank-GRPO's main test (per row / per request).
(a) constant shaping scale (--shaping_scale_const 2.47) for m = all: 2B_consensusrw_group_s3407_p0c_shconst minus
    2B_consensusrw_group_s3407_p0c (validation best, test per row / per request); reported once its lock exists.
Usage: env/bin/python remark/scripts/n936_e1_iso.py [s3407 s42]
Output: outputs/crs/analysis/n936_e1_iso_fixed_<seeds>.json (+ _test.json)
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
import crs_dose_response as cdr  # noqa: E402
from crs_compare import DATA, RG, context_groups, paired_stats, per_row  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
RANKGRPO_TEST = {"s3407": "test_2B_single_group_s3407_step2000", "s42": "test_2B_single_group_s42_step2000"}
FIXED = {"1": "2B_single_fixed_{s}_p0c", "2": "2B_meanm2rw_fixed_{s}_p0c", "4": "2B_meanm4rw_fixed_{s}_p0c",
         "all": "2B_consensusrw_fixed_{s}_p0c"}
SHCONST, SHBASE = "2B_consensusrw_group_s3407_p0c_shconst", "2B_consensusrw_group_s3407_p0c"


def curve(run: str) -> dict[int, float]:
    out = {}
    for p in (C / "val").glob(f"grpo_qwen_{run}_step*/summary.json"):
        m = re.search(r"_step(\d+)$", p.parent.name)
        if m:
            out[int(m.group(1))] = 100 * json.load(open(p))["truncate_R@20"]
    return dict(sorted(out.items()))


def locked_test(run: str) -> str:
    return f"test_{run}_step{json.load(open(C / 'locks' / f'{run}.json'))['best_step']}"


def main() -> None:
    seeds = sys.argv[1:] or ["s3407"]  # 10-08: s42 added; one seed keeps the original key layout
    cdr.RUNS, cdr.SEEDS = FIXED, seeds
    out = C / f"analysis/n936_e1_iso_fixed_{'_'.join(seeds)}.json"
    argv, sys.argv = sys.argv, [sys.argv[0], "--out", str(out)]
    cdr.main()
    sys.argv = argv
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

    res: dict = {"note": __doc__.split("Usage")[0].strip(), "units": "points (x100)", "b_fixed": {}, "a_shconst": None}
    for sd in seeds:
        key = (lambda m: m) if len(seeds) == 1 else (lambda m, sd=sd: f"{m}/{sd}")
        for m, pat in FIXED.items():
            run = pat.format(s=sd)
            t = locked_test(run)
            cv = curve(run)
            res["b_fixed"][key(m)] = {"run": run, "test": t, "best_val": max(cv.values()), "val_curve": cv,
                                      "rankgrpo_test": RANKGRPO_TEST[sd], "test_minus_rankgrpo": cmp(t, RANKGRPO_TEST[sd])}
            r = res["b_fixed"][key(m)]["test_minus_rankgrpo"]
            print(f"(b) m={m} {sd}: best val {max(cv.values()):.2f}; test {r['row']['a']:.2f} - RankGRPO {r['row']['b']:.2f} = row "
                  f"{r['row']['diff']:+.2f} [{r['row']['ci_low']:+.2f},{r['row']['ci_high']:+.2f}] unit {r['unit']['diff']:+.2f} "
                  f"[{r['unit']['ci_low']:+.2f},{r['unit']['ci_high']:+.2f}]")
        # same-scale pair : m = all fixed minus m = 1 fixed
        r = cmp(res["b_fixed"][key("all")]["test"], res["b_fixed"][key("1")]["test"])
        res["b_all_minus_single_fixed" + ("" if len(seeds) == 1 else f"/{sd}")] = r
        print(f"(b) all fixed - single fixed {sd}: row {r['row']['diff']:+.2f} [{r['row']['ci_low']:+.2f},{r['row']['ci_high']:+.2f}] "
              f"unit {r['unit']['diff']:+.2f} [{r['unit']['ci_low']:+.2f},{r['unit']['ci_high']:+.2f}]")
    if (C / "locks" / f"{SHCONST}.json").exists():
        ta, tb = locked_test(SHCONST), locked_test(SHBASE)
        ca, cb = curve(SHCONST), curve(SHBASE)
        res["a_shconst"] = {"run": SHCONST, "base": SHBASE, "test": ta, "base_test": tb, "best_val": max(ca.values()),
                            "base_best_val": max(cb.values()), "val_curve": ca, "test_minus_base": cmp(ta, tb),
                            "test_minus_rankgrpo": cmp(ta, RANKGRPO_TEST["s3407"])}
        r = res["a_shconst"]["test_minus_base"]
        print(f"(a) shconst: best val {max(ca.values()):.2f} vs {max(cb.values()):.2f}; test {r['row']['a']:.2f} - "
              f"{r['row']['b']:.2f} = row {r['row']['diff']:+.2f} [{r['row']['ci_low']:+.2f},{r['row']['ci_high']:+.2f}] "
              f"unit {r['unit']['diff']:+.2f} [{r['unit']['ci_low']:+.2f},{r['unit']['ci_high']:+.2f}]")
    out.with_name(out.stem + "_test.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
