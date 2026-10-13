"""A: registered instability check for the mean-reward + per-group-scaling runs (2B_consensus_group_s<seed>_p0c).

A run is unstable at the first logging window after step 100 in which any of
  (i)   KL > 5x the maximum KL of the same-seed ReMaRk run (2B_union_group_s<seed>_p0c, all its logged windows),
  (ii)  mean items per list outside [19.5, 20.5] or exact-20 share < 90%, over a 50-step window (5 log entries),
  (iii) truncated-completion share (completions/clipped_ratio) > 10% over a 50-step window,
  (iv)  validation R@20 more than 3 points below the run's running best.
Log entries are every 10 steps; a _seg<k> log starts at step k. Prints one line per run and writes
outputs/crs/analysis/n933_a_stability.json. Usage: env/bin/python remark/scripts/n933_a_stability.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n928_step_cost import entries_from_log, L  # noqa: E402

C = Path(__file__).resolve().parents[2] / "remark/outputs/crs"
SEEDS = (3407, 42, 1234)


def logged(run: str) -> dict[int, dict]:
    out = {}
    for p in sorted(L.glob(f"*/grpo_{run}.log")) + sorted(L.glob(f"*/grpo_{run}_seg*.log")):
        m = re.search(r"_seg(\d+)\.log$", p.name)
        off = int(m.group(1)) if m else 0
        for i, x in enumerate(entries_from_log(p)):
            out[off + 10 * (i + 1)] = x
    return dict(sorted(out.items()))


def val_curve(run: str) -> dict[int, float]:
    out = {}
    for p in (C / "val").glob(f"grpo_qwen_{run}_step*/summary.json"):
        out[int(p.parent.name.rsplit("_step", 1)[1])] = 100 * json.load(open(p))["truncate_R@20"]
    return dict(sorted(out.items()))


def check(seed: int) -> dict:
    run, ref = f"2B_consensus_group_s{seed}_p0c", f"2B_union_group_s{seed}_p0c"
    ref_kl = max((x.get("kl", 0.0) for p in list(L.glob(f"*/grpo_{ref}.log")) + list(L.glob(f"*/grpo_{ref}_seg*.log")) for x in entries_from_log(p)), default=None)
    lg = logged(run)
    res = {"run": run, "ref_run": ref, "ref_max_kl": ref_kl, "kl_limit": 5 * ref_kl if ref_kl else None,
           "last_step": max(lg) if lg else 0, "unstable": None}
    steps = list(lg)
    for j, s in enumerate(steps):
        if s <= 100:
            continue
        x = lg[s]
        if ref_kl and x.get("kl", 0) > 5 * ref_kl:
            res["unstable"] = {"step": s, "criterion": "i", "value": x["kl"]}
            break
        win = [lg[t] for t in steps[max(0, j - 4): j + 1]]
        if len(win) == 5:
            items = float(np.mean([w["rank/items_emitted"] for w in win]))
            exact = float(np.mean([w["rank/frac_exact_len"] for w in win]))
            trunc = float(np.mean([w["completions/clipped_ratio"] for w in win]))
            if not 19.5 <= items <= 20.5 or exact < 0.90:
                res["unstable"] = {"step": s, "criterion": "ii", "items": items, "exact": exact}
                break
            if trunc > 0.10:
                res["unstable"] = {"step": s, "criterion": "iii", "truncated": trunc}
                break
    cv, best = val_curve(run), -1.0
    for s, v in cv.items():
        best = max(best, v)
        if s > 100 and v < best - 3 and (res["unstable"] is None or s < res["unstable"]["step"]):
            res["unstable"] = {"step": s, "criterion": "iv", "val": v, "running_best": best}
            break
    res["val"] = cv
    if lg:
        last = [lg[t] for t in steps[-5:]]
        res["last_window"] = {k: float(np.mean([w[k] for w in last])) for k in
                              ("kl", "rank/items_emitted", "rank/frac_exact_len", "completions/clipped_ratio")}
    return res


def main() -> None:
    out = {s: check(s) for s in SEEDS}
    (C / "analysis/n933_a_stability.json").write_text(json.dumps(out, indent=1))
    for s, r in out.items():
        print(r["run"], "last step", r["last_step"], "KL limit", r["kl_limit"] and round(r["kl_limit"], 3),
              "UNSTABLE" if r["unstable"] else "stable so far", r["unstable"] or r.get("last_window"))


if __name__ == "__main__":
    main()
