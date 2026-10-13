"""A readout: mean reward (consensus, phat = c/n) + per-group scaling + request sampling, 2B, seeds 3407/42/1234.

Registered readout: per seed A - ReMaRk (same seed, main-table lock) and A - mean+batch (Table 4 row 5, seed 3407 only),
paired bootstrap over requests (crs_compare, 2,000 resamples, seed 42), per row (primary) and per request; three-seed
Welch for A - ReMaRk (make_paper_stats.welch, seeds as replicates). Stability verdicts come from n933_a_stability.py
(run it first). Output: outputs/crs/analysis/n933_a_readout.json
Usage: env/bin/python remark/scripts/n933_a_readout.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_paper_stats import welch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
T, A = REPO / "remark/outputs/crs/test", REPO / "remark/outputs/crs/analysis"
ARM = {s: f"test_2B_consensus_group_s{s}_p0c_step2000" for s in (3407, 42, 1234)}
# s42: unstable at step 1,800 (criterion i, one KL window in the resumed segment); registered lock = best validated
# checkpoint up to the event = 1,500 (primary 1,500, the autotest 2,000 test in the appendix)
ARM_S42_2000 = ARM[42]
ARM[42] = "test_2B_consensus_group_s42_p0c_step1500"
REMARK = {3407: "test_2B_union_group_step2000", 42: "test_2B_union_group_s42_p0c_step2000",
          1234: "test_2B_union_group_s1234_p0c_step2000"}
MEAN_BATCH = "test_2B_consensus_rerun_step2000"  # Table 4 row 5 (mean reward, batch scaling, request sampling)


def main() -> None:
    a_args = [str(T / ARM[s]) for s in (3407, 42, 1234)] + [str(T / ARM[3407]), str(T / ARM_S42_2000)]
    b_args = [str(T / REMARK[s]) for s in (3407, 42, 1234)] + [str(T / MEAN_BATCH), str(T / REMARK[42])]
    res = {"pairs": {}, "arm_R@20": {}, "remark_R@20": {}}
    for w in ("row", "unit"):
        cmd = [sys.executable, str(REPO / "remark/scripts/crs_compare.py"), "--split", "test", "--weighting", w,
               "--out", str(A / f"n933_a_pair_{w}.json")]
        for a, b in zip(a_args, b_args):
            cmd += ["--a", a, "--b", b]
        subprocess.run(cmd, check=True, capture_output=True)
        for p in json.load(open(A / f"n933_a_pair_{w}.json"))["pairs"]:
            key = f"{Path(p['a']).name} - {Path(p['b']).name}"
            res["pairs"].setdefault(key, {})[w] = {m: [p[m]["diff"], p[m]["ci_low"], p[m]["ci_high"]] for m in ("R@20", "N@20")}
            if w == "row":
                res["arm_R@20"][Path(p["a"]).name], res["remark_R@20"][Path(p["b"]).name] = p["R@20"]["a"], p["R@20"]["b"]
    a = [res["arm_R@20"][ARM[s]] for s in (3407, 42, 1234)]
    b = [res["remark_R@20"][REMARK[s]] for s in (3407, 42, 1234)]
    res["welch_A_minus_ReMaRk"] = welch(a, b)[:3]
    a2 = [res["arm_R@20"][ARM[3407]], res["arm_R@20"][ARM_S42_2000], res["arm_R@20"][ARM[1234]]]
    res["welch_A_minus_ReMaRk_s42_step2000_appendix"] = welch(a2, b)[:3]
    res["stability"] = json.load(open(A / "n933_a_stability.json"))
    (A / "n933_a_readout.json").write_text(json.dumps(res, indent=1))
    for k, v in res["pairs"].items():
        r, u = v["row"]["R@20"], v["unit"]["R@20"]
        print(f"{k}: row {100 * r[0]:+.2f} [{100 * r[1]:+.2f}, {100 * r[2]:+.2f}]  request {100 * u[0]:+.2f} "
              f"[{100 * u[1]:+.2f}, {100 * u[2]:+.2f}]")
    print("A", [round(100 * x, 2) for x in a], "ReMaRk", [round(100 * x, 2) for x in b],
          "Welch", [round(100 * x, 2) for x in res["welch_A_minus_ReMaRk"]],
          "| s42 @2000 (appendix) Welch", [round(100 * x, 2) for x in res["welch_A_minus_ReMaRk_s42_step2000_appendix"]])
    for s, r in res["stability"].items():
        print(s, "UNSTABLE" if r["unstable"] else "stable", r["unstable"] or "")


if __name__ == "__main__":
    main()
