"""Vote025 reward (weight 1 + 0.25 * 1{named by >= 2 responders}) vs ReMaRk and vs Rank-GRPO, 2B s3407.

Runs the registered analyses unchanged on the E7 arm:
  (1) paired test difference per row and per request (crs_compare, 2,000 resamples, seed 42):
      E7 - ReMaRk (test_2B_union_group_step2000) and E7 - Rank-GRPO (test_2B_single_group_s3407_step2000);
  (2) popularity strata (n928_r7a_popularity_strata definitions; Reddit-v2 test quantiles);
  (3) single-responder-item recall (n931_c3_minority definitions; U1 = named once, U2 = named by >= 2).
The two analysis modules are imported and their pair tables replaced; their own output files are not touched.
Output: outputs/crs/analysis/n932_e7_{pair_row,pair_unit}.json, n932_e7_popularity_strata.json, n932_e7_minority.json
Usage: env/bin/python remark/scripts/n932_e7_readout.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import n928_r7a_popularity_strata as r7a  # noqa: E402
import n931_c3_minority as c3  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
A = REPO / "remark/outputs/crs/analysis"
T = REPO / "remark/outputs/crs/test"
E7 = "test_2B_vote025_group_s3407_p0c_step2000"
REMARK, RANKGRPO = "test_2B_union_group_step2000", "test_2B_single_group_s3407_step2000"


def main() -> None:
    for w in ("row", "unit"):
        subprocess.run([sys.executable, str(REPO / "remark/scripts/crs_compare.py"), "--a", str(T / E7), "--a", str(T / E7),
                        "--b", str(T / REMARK), "--b", str(T / RANKGRPO), "--split", "test", "--weighting", w,
                        "--out", str(A / f"n932_e7_pair_{w}.json")], check=True, capture_output=True)
    # (2) popularity strata, Reddit-v2 only
    r7a.TEST["2B"]["vote025"] = E7
    r7a.DATASETS = {k: v for k, v in r7a.DATASETS.items() if k == "reddit"}
    r7a.SETTINGS = {"2B": ("reddit", [("vote025", "remark"), ("vote025", "rankgrpo")])}
    r7a.OUT = A / "n932_e7_popularity_strata.json"
    r7a.main()
    # (3) minority recall; c3 writes to <REPO>/remark/outputs/crs/analysis/n931_c3_minority.json -> redirect to a temp root
    c3.PAIRS = {"2B": ("reddit", E7, {"vs_remark": REMARK, "vs_rankgrpo": RANKGRPO})}  # c3 stores arm a under "remark"
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "remark/outputs/crs/analysis").mkdir(parents=True)
        c3.REPO = Path(tmp)
        c3.main()
        (A / "n932_e7_minority.json").write_text((Path(tmp) / "remark/outputs/crs/analysis/n931_c3_minority.json").read_text())
    for w in ("row", "unit"):
        for p in json.load(open(A / f"n932_e7_pair_{w}.json"))["pairs"]:
            r = p["R@20"]
            print(w, Path(p["b"]).name, f"{100 * r['a']:.2f} vs {100 * r['b']:.2f}: {100 * r['diff']:+.2f} "
                  f"[{100 * r['ci_low']:+.2f}, {100 * r['ci_high']:+.2f}]")


if __name__ == "__main__":
    main()
