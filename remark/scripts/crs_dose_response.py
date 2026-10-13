"""Dose-response readout, registered before any run.

Reward = mean over a responder subset of size m (m in {1, 2, 4, all}); x = removed responder noise 1 - rho(m) with
rho(1) = 1, rho(2) = 0.433, rho(4) = 0.164, rho(all) = 0 (measured on the training rows, crs_build_mean_m.py).
Runs: 2B, P0c, batch scaling, rows in Rank-GRPO order, seeds 3407 / 42 / 1234 (m = 1 is the per-responder batch run and
m = all the consensusrw run for seed 3407, as registered).
Readout on validation, per-row R@20 at each run's best step and at step 2,000: OLS slope of R@20 on x over the 12 runs
(seeds as replicates), CI from a paired bootstrap over validation contexts (2,000, seed 42).
PASS: slope > 0 with CI > 0, the m = all - m = 1 seed-mean gain >= +0.008, and the m = 2 and m = 4 seed means within
+-0.004 of the fitted line. PRUNE: m = 2 and m = 4 seed means both <= m = 1 + 0.002. Otherwise inconclusive.
Also reported: the same on single-responder (n = 1) contexts.

Usage: env/bin/python remark/scripts/crs_dose_response.py --out remark/outputs/crs/analysis/dose_response.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import DATA, REPO, RG  # noqa: E402
from crs_val_table import row_stats  # noqa: E402

RHO = {"1": 1.0, "2": 0.433, "4": 0.164, "all": 0.0}
RUNS = {"1": "2B_single_batch_{s}_p0c", "2": "2B_meanm2rw_batch_{s}_p0c", "4": "2B_meanm4rw_batch_{s}_p0c",
        "all": "2B_consensusrw_batch_{s}_p0c"}
SEEDS = ["s3407", "s42", "s1234"]
STEPS = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]


def verdict(means: dict, slope_ci: tuple) -> str:
    x = np.array([1 - RHO[m] for m in means]); y = np.array(list(means.values()))
    b, a = np.polyfit(x, y, 1)
    resid = {m: means[m] - (a + b * (1 - RHO[m])) for m in means}
    passed = slope_ci[0] > 0 and means["all"] - means["1"] >= 0.008 and abs(resid["2"]) <= 0.004 and abs(resid["4"]) <= 0.004
    pruned = means["2"] <= means["1"] + 0.002 and means["4"] <= means["1"] + 0.002
    return "PASS" if passed else "PRUNE" if pruned else "INCONCLUSIVE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_dir", default=str(REPO / "remark/outputs/crs/val"))
    a = ap.parse_args()
    ds = load_from_disk(str(DATA / "validation"))
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    ctx = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx[p[0]["content"]].append(i)
    groups = list(ctx.values())
    n_row = np.zeros(len(ds), int)
    for g in groups:
        n_row[g] = len(g)
    ctx_of = np.zeros(len(ds), int)
    for c, g in enumerate(groups):
        ctx_of[g] = c
    V = Path(a.val_dir)
    out = {"rho": RHO, "runs": {}, "readouts": {}}
    per_row = {}
    for m, pat in RUNS.items():
        for s in SEEDS:
            run = pat.format(s=s)
            val = {st: json.load(open(V / f"grpo_qwen_{run}_step{st}" / "summary.json"))["truncate_R@20"] for st in STEPS}
            best = max(val, key=val.get)
            out["runs"][f"{m}/{s}"] = {"run": run, "val": val, "best_step": best}
            for which, st in (("best", best), ("2000", 2000)):
                r20, *_ = row_stats(V / f"grpo_qwen_{run}_step{st}", ds, catalog, 20)
                per_row[(which, m, s)] = r20
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    for which in ("best", "2000"):
        for subset, mask in (("all", np.ones(len(ds), bool)), ("n=1", n_row == 1)):
            # per-context sums and row counts, so a context resample gives row-weighted means
            cnt = np.bincount(ctx_of[mask], minlength=len(groups)).astype(float)
            sums = {k: np.bincount(ctx_of[mask], weights=v[mask], minlength=len(groups))
                    for k, v in per_row.items() if k[0] == which}
            def means_for(idx):
                tot = cnt[idx].sum()
                return {k: sums[k][idx].sum() / tot for k in sums}
            full = means_for(np.arange(len(groups)))
            xs = np.array([1 - RHO[k[1]] for k in full]); ys = np.array(list(full.values()))
            slope = float(np.polyfit(xs, ys, 1)[0])
            boots = []
            for d in draws:
                mb = means_for(d)
                boots.append(np.polyfit(np.array([1 - RHO[k[1]] for k in mb]), np.array(list(mb.values())), 1)[0])
            ci = tuple(float(v) for v in np.percentile(boots, [2.5, 97.5]))
            seed_means = {m: float(np.mean([full[(which, m, s)] for s in SEEDS])) for m in RUNS}
            per_seed = {f"{m}/{s}": float(full[(which, m, s)]) for m in RUNS for s in SEEDS}
            res = {"slope": slope, "slope_ci": ci, "seed_means": seed_means, "per_seed": per_seed,
                   "gain_all_minus_1": seed_means["all"] - seed_means["1"],
                   "verdict": verdict(seed_means, ci) if subset == "all" else None}
            out["readouts"][f"{which}/{subset}"] = res
            print(which, subset, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
                                             if k != "per_seed"}))
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
