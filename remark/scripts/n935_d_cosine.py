"""readout (internal check, not pre-registered): cosine learning-rate schedule, 2B Reddit-v2, 4,000 steps.

Runs 2B_{single,union}_group_s{3407,42}_p0c_ms4000_cos (Rank-GRPO / ReMaRk, --lr_scheduler cosine, peak 1e-6), locked on
the best validation per-row R@20 and tested once.
(i) Steps for ReMaRk's validation curve to reach Rank-GRPO's best validation score and its step-2000 score, both from the
    cosine runs of the same seed (first crossing, linear interpolation between validated steps; step 0 = the SFT start,
    sft_qwen35_2B_step1500 validation R@20).
(ii) Test ReMaRk - Rank-GRPO per seed (crs_compare paired bootstrap over contexts, 2,000 resamples, seed 42; per row and
    per request), and over the two seeds the difference of the arms' mean test scores with a Welch 95% CI.
(iii) Each cosine arm against the constant-lr runs of the same arm and seed: test paired differences (cosine - constant)
    and the best validation score within 4,000 steps. s3407: the main 2,000-step runs and the 4,000-step constant runs
    (2B_single_group_s3407_p0c_ms4000_lr1e-6, 2B_union_group_s3407_p0c_ms4000); s42: the main runs (test) and the first
    4,000 steps of the 8,000-step runs 2B_*_group_s42_p0c_ms8000 (validation only).
Output: outputs/crs/analysis/n935_d_cosine.json
Usage: env/bin/python remark/scripts/n935_d_cosine.py
"""
from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_compare import DATA, RG, context_groups, paired_stats, per_row  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
OUT = C / "analysis/n935_d_cosine.json"
SFT_VAL = "sft_qwen35_2B_step1500"
SEEDS = (3407, 42)
# constant-lr comparison runs: (test dir or None, validation run or None, max step of the curve)
CONST = {
    ("single", 3407): [("main_2k", "test_2B_single_group_s3407_step2000", "2B_single_group_s3407_p0c", 2000),
                       ("const_4k", "test_2B_single_group_s3407_p0c_ms4000_lr1e-6_step4000",
                        "2B_single_group_s3407_p0c_ms4000_lr1e-6", 4000)],
    ("union", 3407): [("main_2k", "test_2B_union_group_step2000", "2B_union_group_s3407_p0c", 2000),
                      ("const_4k", "test_2B_union_group_s3407_p0c_ms4000_step3250", "2B_union_group_s3407_p0c_ms4000", 4000)],
    ("single", 42): [("main_2k", "test_2B_single_group_s42_step2000", "2B_single_group_s42_p0c", 2000),
                     ("const_8k_first4k", None, "2B_single_group_s42_p0c_ms8000", 4000)],
    ("union", 42): [("main_2k", "test_2B_union_group_s42_p0c_step2000", "2B_union_group_s42_p0c", 2000),
                    ("const_8k_first4k", None, "2B_union_group_s42_p0c_ms8000", 4000)],
}


def curve(run: str, max_step: int = 10**9) -> dict[int, float]:
    out = {}
    for p in (C / "val").glob(f"grpo_qwen_{run}_step*/summary.json"):
        m = re.search(r"_step(\d+)$", p.parent.name)
        if m and int(m.group(1)) <= max_step:
            out[int(m.group(1))] = json.load(open(p))["truncate_R@20"]
    return dict(sorted(out.items()))


def first_reach(cur: dict[int, float], target: float, v0: float) -> float | None:
    pts = [(0, v0)] + list(cur.items())
    for (s0, a), (s1, b) in zip(pts, pts[1:]):
        if b >= target:
            return float(s1) if b == a or a >= target else float(s0 + (target - a) / (b - a) * (s1 - s0))
    return None


def lock(run: str) -> dict:
    return json.load(open(C / "locks" / f"{run}.json"))


def test_name(run: str) -> str:
    return f"test_{run}_step{lock(run)['best_step']}"


def welch(a: list[float], b: list[float]) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    h = stats.t.ppf(0.975, df) * se
    return {"diff": 100 * float(d), "ci_low": 100 * float(d - h), "ci_high": 100 * float(d + h), "df": float(df)}


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

    def req_mean(name: str) -> float:
        x = r20(name)
        return float(np.mean([x[g].mean() for g in groups]))

    v0 = json.load(open(C / "val" / SFT_VAL / "summary.json"))["truncate_R@20"]
    res: dict = {"note": __doc__.split("Usage")[0].strip(), "sft_val_R@20": v0, "units": "test/diff values in points (x100)",
                 "per_seed": {}, "welch_two_seeds": {}, "vs_constant": {}}
    means = {(arm, w): [] for arm in ("single", "union") for w in ("row", "unit")}
    for s in SEEDS:
        runs = {arm: f"2B_{arm}_group_s{s}_p0c_ms4000_cos" for arm in ("single", "union")}
        cu, cs = curve(runs["union"]), curve(runs["single"])
        best_s = max(cs.values())
        tests = {arm: test_name(r) for arm, r in runs.items()}
        res["per_seed"][str(s)] = {
            "runs": runs, "locks": {arm: lock(r)["best_step"] for arm, r in runs.items()}, "tests": tests,
            "val_curves": {"union": cu, "single": cs},
            "steps_to_rankgrpo_best_val": first_reach(cu, best_s, v0), "rankgrpo_best_val": best_s,
            "rankgrpo_best_val_step": max(cs, key=lambda k: (cs[k], -k)),
            "steps_to_rankgrpo_step2000_val": first_reach(cu, cs[2000], v0), "rankgrpo_step2000_val": cs[2000],
            "test_remark_minus_rankgrpo": cmp(tests["union"], tests["single"]),
        }
        for arm in ("single", "union"):
            means[(arm, "row")].append(float(r20(tests[arm]).mean()))
            means[(arm, "unit")].append(req_mean(tests[arm]))
            out = {}
            for tag, tname, vrun, ms in CONST[(arm, s)]:
                cv = curve(vrun, ms)
                e = {"val_run": vrun, "best_val_within": ms, "const_best_val": max(cv.values()) if cv else None,
                     "cos_best_val": max((curve(runs[arm])).values())}
                if tname and (C / "test" / tname / "per_instance.jsonl").exists():
                    e["test"] = tname
                    e["test_cos_minus_const"] = cmp(tests[arm], tname)
                out[tag] = e
            res["vs_constant"][f"{arm}_s{s}"] = out
    for w in ("row", "unit"):
        res["welch_two_seeds"][w] = {"remark_mean": 100 * float(np.mean(means[("union", w)])),
                                     "rankgrpo_mean": 100 * float(np.mean(means[("single", w)])),
                                     **welch(means[("union", w)], means[("single", w)])}
    OUT.write_text(json.dumps(res, indent=2))
    for s, r in res["per_seed"].items():
        t = r["test_remark_minus_rankgrpo"]
        print(f"s{s}: locks {r['locks']} steps-to-best {r['steps_to_rankgrpo_best_val']} steps-to-2k "
              f"{r['steps_to_rankgrpo_step2000_val']}; test diff row {t['row']['diff']:+.2f} [{t['row']['ci_low']:+.2f},"
              f"{t['row']['ci_high']:+.2f}] unit {t['unit']['diff']:+.2f} [{t['unit']['ci_low']:+.2f},{t['unit']['ci_high']:+.2f}]")
    print(json.dumps(res["welch_two_seeds"]))
    for k, v in res["vs_constant"].items():
        for tag, e in v.items():
            t = e.get("test_cos_minus_const", {}).get("row")
            print(k, tag, f"val best cos {e['cos_best_val']:.4f} const {e['const_best_val']}",
                  f"test cos-const row {t['diff']:+.2f} [{t['ci_low']:+.2f},{t['ci_high']:+.2f}]" if t else "")


if __name__ == "__main__":
    main()
