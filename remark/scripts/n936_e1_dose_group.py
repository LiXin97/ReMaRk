"""E1 readout (internal check, not pre-registered): the dose-response of Fig 4b under per-group advantage scaling.

Same algorithm as crs_dose_response.py (validation per-row R@20 at each run's best step and at step 2,000, OLS slope on
the removed responder noise x = 1 - rho(m), paired bootstrap over validation contexts, 2,000 resamples, seed 42), with
group scaling in every arm: m = 1 is Rank-GRPO (2B_single_group_<seed>_p0c), m = 2 / 4 / all are
2B_{meanm2rw,meanm4rw,consensusrw}_group_<seed>_p0c (requests sampled proportionally to n, as in the batch arms).
Seeds given on the command line (s3407 first; s42 / s1234 when they land). Also the test R@20 of each arm's locked
checkpoint minus Rank-GRPO's main test (per row and per request, crs_compare bootstrap).
Usage: env/bin/python remark/scripts/n936_e1_dose_group.py s3407 [s42 s1234]
Output: outputs/crs/analysis/n936_e1_dose_group_<seeds>.json (+ _test.json)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crs_dose_response as cdr  # noqa: E402
from crs_compare import DATA, RG, context_groups, paired_stats, per_row  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
RANKGRPO_TEST = {"s3407": "test_2B_single_group_s3407_step2000", "s42": "test_2B_single_group_s42_step2000",
                 "s1234": "test_2B_single_group_s1234_step2000"}  # = make_paper_stats rankgrpo_s1234 (_p0c ckpt-2000)


def main() -> None:
    seeds = sys.argv[1:] or ["s3407"]
    cdr.RUNS = {"1": "2B_single_group_{s}_p0c", "2": "2B_meanm2rw_group_{s}_p0c", "4": "2B_meanm4rw_group_{s}_p0c",
                "all": "2B_consensusrw_group_{s}_p0c"}
    cdr.SEEDS = seeds
    tag = "_".join(seeds)
    out = C / f"analysis/n936_e1_dose_group_{tag}.json"
    sys.argv = [sys.argv[0], "--out", str(out)]
    cdr.main()
    # test readout
    ds = load_from_disk(str(Path(DATA) / "test"))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    groups = list(context_groups([p[0]["content"] for p in ds["prompt"]]).values())
    draws = np.random.default_rng(42).integers(0, len(groups), size=(2000, len(groups)))
    res = {}
    for s in seeds:
        base = RANKGRPO_TEST[s]
        if "{}" in base:
            base = base.format(json.load(open(C / "locks/2B_single_group_s1234_p0c.json"))["best_step"])
        xb = per_row(C / "test" / base, ds, catalog)["R@20"]
        for m in ("2", "4", "all"):
            run = cdr.RUNS[m].format(s=s)
            t = f"test_{run}_step{json.load(open(C / 'locks' / f'{run}.json'))['best_step']}"
            xa = per_row(C / "test" / t, ds, catalog)["R@20"]
            res[f"{m}/{s}"] = {"test": t, "base": base, **{w: {k: (100 * v if k != "p_le_0" else v) for k, v in
                               paired_stats(xa, xb, groups, draws, w).items()} for w in ("row", "unit")}}
            r = res[f"{m}/{s}"]
            print(f"test m={m} {s}: {r['row']['a']:.2f} - RankGRPO {r['row']['b']:.2f} = row {r['row']['diff']:+.2f} "
                  f"[{r['row']['ci_low']:+.2f},{r['row']['ci_high']:+.2f}] unit {r['unit']['diff']:+.2f} "
                  f"[{r['unit']['ci_low']:+.2f},{r['unit']['ci_high']:+.2f}]")
    out.with_name(out.stem + "_test.json").write_text(json.dumps({"note": __doc__.split("Usage")[0].strip(),
                                                                  "units": "points (x100)", "pairs": res}, indent=2))


if __name__ == "__main__":
    main()
