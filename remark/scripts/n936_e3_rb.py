"""E3 readout (internal check, not pre-registered): exact Rao-Blackwellization (RB) of Rank-GRPO, 2B Reddit-v2.

Arms (Table 2 protocol, best-validation lock, tested once):
  RB_row  = 2B_single_group_<seed>_p0c_rb (rows sampled per responder, as Rank-GRPO), seeds 3407 / 42 / 1234;
  RB_req  = 2B_singlethr_group_s3407_p0c_rb (one random responder per request = request-uniform sampling; the RB table
            holds every responder of the request, --rb_table_path);
  TR      = 2B_singlethr_group_s3407_p0c (Rank-GRPO with request-uniform sampling, no RB);
  ReMaRk  = union reward + group scaling + request sampling; Rank-GRPO = single + group + row sampling.
Splits the s3407 gap ReMaRk - RB_row into sampling (RB_req - RB_row) and reward (ReMaRk - RB_req); also RB_req - TR (RB
under request sampling) and, over three seeds, RB_row - Rank-GRPO and ReMaRk - RB_row (per-seed paired bootstrap over
contexts, 2,000 resamples, seed 42; Welch 95% CI on the seed means). Per row and per request.
Output: outputs/crs/analysis/n936_e3_rb.json
Usage: env/bin/python remark/scripts/n936_e3_rb.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_compare import DATA, RG, context_groups, paired_stats, per_row  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
OUT = C / "analysis/n936_e3_rb.json"
SEEDS = ("3407", "42", "1234")
REMARK = {"3407": "test_2B_union_group_step2000", "42": "test_2B_union_group_s42_p0c_step2000",
          "1234": "test_2B_union_group_s1234_p0c_step2000"}
RANKGRPO = {s: f"test_2B_single_group_s{s}_step2000" for s in SEEDS}


def lock_test(run: str) -> str:
    return f"test_{run}_step{json.load(open(C / 'locks' / f'{run}.json'))['best_step']}"


RB_ROW = {s: lock_test(f"2B_single_group_s{s}_p0c_rb") for s in SEEDS}
RB_REQ = lock_test("2B_singlethr_group_s3407_p0c_rb")
TR = "test_2B_singlethr_group_s3407_p0c_step2000"


def welch(a, b) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    d = a.mean() - b.mean()
    return {"diff": 100 * float(d), "ci_low": 100 * float(d - h), "ci_high": 100 * float(d + h), "df": float(df)}


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

    def unit_mean(n: str) -> float:
        x = r20(n)
        return float(np.mean([x[g].mean() for g in groups]))

    res = {"note": __doc__.split("Usage")[0].strip(), "units": "points (x100)",
           "tests": {"RB_row": RB_ROW, "RB_req": RB_REQ, "TR_req": TR, "ReMaRk": REMARK, "Rank-GRPO": RANKGRPO},
           "s3407": {"ReMaRk - RB_row": cmp(REMARK["3407"], RB_ROW["3407"]),
                     "RB_req - RB_row (sampling)": cmp(RB_REQ, RB_ROW["3407"]),
                     "ReMaRk - RB_req (reward)": cmp(REMARK["3407"], RB_REQ),
                     "RB_req - TR_req (RB under request sampling)": cmp(RB_REQ, TR),
                     "RB_row - Rank-GRPO": cmp(RB_ROW["3407"], RANKGRPO["3407"]),
                     "RB_req - Rank-GRPO": cmp(RB_REQ, RANKGRPO["3407"])},
           "per_seed": {s: {"RB_row - Rank-GRPO": cmp(RB_ROW[s], RANKGRPO[s]), "ReMaRk - RB_row": cmp(REMARK[s], RB_ROW[s])}
                        for s in SEEDS},
           "welch_3_seeds": {}}
    for w, f in (("row", lambda n: float(r20(n).mean())), ("unit", unit_mean)):
        rb, rg, rm = [f(RB_ROW[s]) for s in SEEDS], [f(RANKGRPO[s]) for s in SEEDS], [f(REMARK[s]) for s in SEEDS]
        res["welch_3_seeds"][w] = {"means": {"RB_row": 100 * np.mean(rb), "Rank-GRPO": 100 * np.mean(rg),
                                             "ReMaRk": 100 * np.mean(rm)},
                                   "RB_row - Rank-GRPO": welch(rb, rg), "ReMaRk - RB_row": welch(rm, rb)}
    OUT.write_text(json.dumps(res, indent=2, default=float))
    fmt = lambda d: f"{d['diff']:+.2f} [{d['ci_low']:+.2f},{d['ci_high']:+.2f}]"
    for k, v in res["s3407"].items():
        print(f"s3407 {k}: {v['row']['a']:.2f} vs {v['row']['b']:.2f}; row {fmt(v['row'])} unit {fmt(v['unit'])}")
    for s, v in res["per_seed"].items():
        print(f"s{s} " + "; ".join(f"{k} row {fmt(x['row'])} unit {fmt(x['unit'])}" for k, x in v.items()))
    for w, v in res["welch_3_seeds"].items():
        print(w, {k: round(x, 2) for k, x in v["means"].items()}, "RB-RG", fmt(v["RB_row - Rank-GRPO"]),
              "ReMaRk-RB", fmt(v["ReMaRk - RB_row"]))


if __name__ == "__main__":
    main()
