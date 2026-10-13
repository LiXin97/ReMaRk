"""Independent audit E2(c): KL to the reference policy when ReMaRk and Rank-GRPO are at the same validation R@20.

For each 2,000-step pair, s* = the (interpolated) step at which ReMaRk reaches Rank-GRPO's step-2,000 validation R@20
(outputs/crs/analysis/n931_c1c2.json, c2.s_star). KL is TRL's logged 'kl' (mean over the 10 optimizer steps of each
record; scripts/n928_r7c_batch_sigma.parse_log handles resumed segments). Reported: ReMaRk's mean KL over the records
within +-100 steps of s*, Rank-GRPO's over steps 1,900-2,000, and both arms at steps 1,900-2,000.
Output: outputs/crs/analysis/e2_kl_matched.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n928_r7c_batch_sigma import parse_log  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "outputs/crs/train_logs"
C1C2 = ROOT / "outputs/crs/analysis/n931_c1c2.json"
OUT = ROOT / "outputs/crs/analysis/e2_kl_matched.json"
PAIRS = {"2B": "2B", "4B": "4B", "9B": "9B", "llama3b": "llama3b", "llama8b": "llama8b"}


def kl_series(run: str) -> dict[int, float]:
    files = sorted(LOGS.glob(f"*/grpo_{run}.log"), key=lambda p: p.stat().st_mtime)
    out: dict[int, float] = {}
    for f in files:
        r = parse_log(f, "kl")
        out.update(dict(zip(r["steps"], r["values"])))
    return dict(sorted(out.items()))


def window(s: dict[int, float], lo: float, hi: float) -> float | None:
    v = [x for k, x in s.items() if lo <= k <= hi]
    return float(np.mean(v)) if v else None


def main() -> None:
    c = json.loads(C1C2.read_text())["settings"]
    res = {"note": __doc__.split("\n")[0]}
    for col, key in PAIRS.items():
        st = c[key]
        rm, rg = st["remark"], st["rankgrpo"]
        s_star = st["rankgrpo"]["c2"]["s_star"]
        rg_run = rm.replace("union_group", "single_group")
        R, B = kl_series(rm), kl_series(rg_run)
        res[col] = {"remark_run": rm, "rankgrpo_run": rg_run, "s_star": round(s_star, 1),
                    "remark_kl_at_s_star": window(R, s_star - 100, s_star + 100),
                    "rankgrpo_kl_at_2000": window(B, 1900, 2000),
                    "remark_kl_at_2000": window(R, 1900, 2000),
                    "records": [len(R), len(B)]}
        print(col, res[col], flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
