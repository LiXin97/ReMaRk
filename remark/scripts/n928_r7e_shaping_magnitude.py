"""Effective length-shaping magnitude per arm under the P0c hygiene, from the training logs. CPU only.

Inputs: remark/outputs/crs/train_logs/<node>/grpo_<run>.log (inventoried below; parsed with
        n928_r7c_batch_sigma.parse_log). Under P0/P0c (lib/rank_grpo_trl.py, shaping_units="reward") the end bonus
        (+0.1 on EOS of an exact-20 list), the early-stop penalty (-0.1 on EOS of a shorter list) and the overflow
        penalty (-0.1 per token after the 20th item) are added to the token advantages multiplied by shaping_scale =
        1 / (mean of the nonzero per-position stds used for the item advantages + 1e-4); every log record (each 10
        optimizer steps) holds the mean of that per-step scale over its 10 steps (rank/shaping_scale).
Definitions
  Effective magnitude = 0.1 x shaping_scale: the size of each shaping term in units of the item advantage (item
  advantages are (reward - group mean) / (std + 1e-4), so a nonzero item advantage at a position with std sigma has
  magnitude ~ 1 per unit of reward / sigma). Per run: mean and 10 / 50 / 90th percentiles over the available log
  records, with the steps the records cover. Resumed segments ('[resume] ... checkpoint-k') are offset by k;
  segments whose log is not on shared storage are reported as missing, never filled in.
  Pairs: Llama-3.2-3B (seed 3407 and 42), Llama-3.1-8B (3407), Qwen3.5-2B (3407, plus seeds 42 / 1234 where logs exist).
  For the 2B ReMaRk seeds 42 / 1234 only the final attempt's resumed segment belongs to the reported run
  (the third attempt); the logs of the earlier, lost attempt are not used.
Output: remark/outputs/crs/analysis/n928_r7e_shaping_magnitude.json
Usage:  env/bin/python remark/scripts/n928_r7e_shaping_magnitude.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import REPO  # noqa: E402
from n928_r7c_batch_sigma import LOGS, parse_log, summarize  # noqa: E402

OUT = REPO / "remark/outputs/crs/analysis/n928_r7e_shaping_magnitude.json"
TOTAL = 2000
# pair -> arm -> (run name, [log files relative to train_logs], note)
PAIRS = {
    "llama3b_s3407": {
        "remark": ("llama3b_union_group_s3407_p0c", ["run_logs/grpo_llama3b_union_group_s3407_p0c.log"],
                   "resumed from checkpoint-250; the log of steps 1-250 is not on shared storage"),
        "rankgrpo": ("llama3b_single_group_s3407_p0c", ["run_logs/grpo_llama3b_single_group_s3407_p0c.log"],
                     "resumed from checkpoint-250; the log of steps 1-250 is not on shared storage")},
    "llama3b_s42": {
        "remark": ("llama3b_union_group_s42_p0c", ["run_logs/grpo_llama3b_union_group_s42_p0c.log"],
                   "log copy ends before step 2,000"),
        "rankgrpo": ("llama3b_single_group_s42_p0c", ["run_logs/grpo_llama3b_single_group_s42_p0c.log"],
                     "log copy ends before step 2,000")},
    "llama8b_s3407": {
        "remark": ("llama8b_union_group_s3407_p0c", ["run_logs/grpo_llama8b_union_group_s3407_p0c.log"], ""),
        "rankgrpo": ("llama8b_single_group_s3407_p0c", ["run_logs/grpo_llama8b_single_group_s3407_p0c.log"], "")},
    "qwen2B_s3407": {
        "remark": ("2B_union_group_s3407_p0c", ["run_logs/grpo_2B_union_group_s3407_p0c.log"], ""),
        "rankgrpo": ("2B_single_group_s3407_p0c", ["run_logs/grpo_2B_single_group_s3407_p0c.log"], "")},
    "qwen2B_s42": {
        "remark": ("2B_union_group_s42_p0c", ["run_logs/grpo_2B_union_group_s42_p0c.log"],
                   "resumed from checkpoint-500; the log of steps 1-500 of this attempt is not on shared storage"),
        "rankgrpo": ("2B_single_group_s42_p0c", ["run_logs/grpo_2B_single_group_s42_p0c.log"], "")},
    "qwen2B_s1234": {
        "remark": ("2B_union_group_s1234_p0c", ["run_logs/grpo_2B_union_group_s1234_p0c.log"],
                   "resumed from checkpoint-250; the log of steps 1-250 of this attempt is not on shared storage"),
        "rankgrpo": ("2B_single_group_s1234_p0c", ["run_logs/grpo_2B_single_group_s1234_p0c.log"], "")},
}


def bins(v: np.ndarray, steps: np.ndarray) -> dict[str, float]:
    out = {}
    for lo in range(0, TOTAL, 500):
        m = (steps > lo) & (steps <= lo + 500)
        if m.any():
            out[f"{steps[m].min() - 9}-{steps[m].max()}"] = float(v[m].mean())
    return out


def main() -> None:
    inventory = sorted(str(p.relative_to(LOGS)) for p in LOGS.rglob("grpo_*.log")
                       if any(k in p.name for k in ("llama3b", "llama8b", "2B_union_group", "2B_single_group")))
    res = {"definition": __doc__.split("Definitions", 1)[1].split("Output:", 1)[0].strip(),
           "log_inventory_considered": inventory, "pairs": {}}
    try:
        res["git"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()
    except OSError:
        pass
    for pair, arms in PAIRS.items():
        out = {}
        for arm, (run, files, note) in arms.items():
            segs, vals, steps = [], [], []
            for f in files:
                p = LOGS / f
                if not p.exists():
                    segs.append({"file": f, "missing": True})
                    continue
                lg = parse_log(p)
                segs.append({k: lg[k] for k in ("file", "resumed_from", "n_records", "steps_covered")})
                vals += lg["values"]
                steps += lg["steps"]
            if not vals:
                out[arm] = {"run": run, "segments": segs, "note": note, "missing": True}
                continue
            v = 0.1 * np.array(vals)
            covered = len(vals) * 10
            out[arm] = {"run": run, "segments": segs, "note": note,
                        "steps_covered": covered, "share_of_2000_steps_covered": covered / TOTAL,
                        "effective_shaping_0.1x_scale": summarize(v.tolist()),
                        "shaping_scale": summarize(vals),
                        "by_500_steps_mean_0.1x_scale (label = steps actually covered)": bins(v, np.array(steps))}
        if all("effective_shaping_0.1x_scale" in out.get(a, {}) for a in ("remark", "rankgrpo")):
            out["ratio_remark_over_rankgrpo_mean"] = (out["remark"]["effective_shaping_0.1x_scale"]["mean"]
                                                      / out["rankgrpo"]["effective_shaping_0.1x_scale"]["mean"])
        res["pairs"][pair] = out
        print(pair, {a: (round(x["effective_shaping_0.1x_scale"]["mean"], 4), x["steps_covered"])
                     for a, x in out.items() if isinstance(x, dict) and "effective_shaping_0.1x_scale" in x}, flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
