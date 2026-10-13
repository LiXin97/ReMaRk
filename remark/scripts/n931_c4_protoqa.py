"""C4 (registered 2026-10-04 15:06): ProtoQA steps-to-match from the ProtoQA arms' validation curves.

Curves: validation per-respondent recall@10 (400 questions) at steps 25..250 of arms (a) single, (b) rb, (c) mean,
(d) union, step 0 = the selected SFT (@360). For (b), (c) and (d) against (a): s* = first crossing of the target by the
arm's curve (linear interpolation), targets (i) (a)'s value at 250 and (ii) (a)'s best; bootstrap over questions (2,000,
seed 42) recomputing all curves and targets; resamples without a crossing are reported as a share. Also the logged share
of groups whose 8 lists all get the same reward (TRL frac_reward_zero_std) and the per-position share
(rank/frac_group_zero_std), from the log history in each arm's kept checkpoint (steps up to that checkpoint).
Output: outputs/protoqa/analysis/n931_c4_steps.json
Usage: env/bin/python remark/scripts/n931_c4_protoqa.py
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
O = REPO / "remark/outputs/protoqa"
ARMS = {"a": "single", "b": "rb", "c": "mean", "d": "union"}
STEPS = list(range(25, 251, 25))


def per_q(name: str) -> dict[str, float]:
    return {r["qid"]: r["resp_recall@10"] for r in map(json.loads, open(O / "eval" / name / "per_instance.jsonl"))}


def crossing(steps, vals, target) -> float:
    if vals[0] >= target:
        return float(steps[0])
    for i in range(1, len(steps)):
        if vals[i] >= target:
            v0, v1 = vals[i - 1], vals[i]
            return float(steps[i - 1] + (target - v0) / (v1 - v0) * (steps[i] - steps[i - 1])) if v1 > v0 else float(steps[i])
    return float("inf")


def main() -> None:
    sft = per_q("r9_sft2B_step360")
    qids = sorted(sft)
    M = {}
    for k, arm in ARMS.items():
        rows = [[sft[q] for q in qids]]
        for s in STEPS:
            d = per_q(f"r9_2B_{arm}_step{s}")
            rows.append([d[q] for q in qids])
        M[k] = np.array(rows)  # (11, n_q)
    steps = np.array([0] + STEPS, dtype=float)
    rng = np.random.default_rng(42)
    W = np.stack([np.bincount(rng.integers(0, len(qids), len(qids)), minlength=len(qids)) for _ in range(2000)])
    out = {"note": __doc__.split("\n")[0], "n_questions": len(qids),
           "curves": {k: (100 * M[k].mean(1)).round(2).tolist() for k in M}, "steps": steps.tolist(), "pairs": {}}
    for k in ("b", "c", "d"):
        for tgt in ("a_at_250", "a_best"):
            def est(w):
                ca = M["a"] @ w / w.sum(); ck = M[k] @ w / w.sum()
                t = ca[-1] if tgt == "a_at_250" else ca[1:].max()
                ref = 250.0 if tgt == "a_at_250" else float(steps[1:][int(np.argmax(ca[1:]))])
                return crossing(steps, ck, t), ref
            s0, ref0 = est(np.ones(len(qids)))
            bs = [est(W[i].astype(float)) for i in range(2000)]
            sb = np.array([b[0] for b in bs]); rb = np.array([b[1] for b in bs]); fin = np.isfinite(sb)
            out["pairs"][f"{k}_vs_a_{tgt}"] = {
                "s_star": s0, "ref_step": ref0, "ratio": s0 / ref0,
                "ratio_ci": [float(np.percentile(np.where(fin, sb / rb, np.inf), q)) for q in (2.5, 97.5)],
                "share_no_crossing": float(1 - fin.mean())}
            print(k, tgt, round(s0, 1), ref0, round(s0 / ref0, 3), out["pairs"][f"{k}_vs_a_{tgt}"]["ratio_ci"],
                  out["pairs"][f"{k}_vs_a_{tgt}"]["share_no_crossing"])
    logs = {}
    for k, arm in ARMS.items():
        for f in glob.glob(str(O / f"grpo/2B_{arm}_s3407/checkpoint-*/trainer_state.json")):
            lh = [x for x in json.load(open(f))["log_history"] if "frac_reward_zero_std" in x]
            logs[k] = {"checkpoint": Path(f).parent.name, "logged_intervals": len(lh),
                       "frac_reward_zero_std": statistics.mean(x["frac_reward_zero_std"] for x in lh),
                       "rank_frac_group_zero_std": statistics.mean(x["rank/frac_group_zero_std"] for x in lh)
                       if lh and "rank/frac_group_zero_std" in lh[0] else None}
    out["no_signal_logged"] = logs
    (O / "analysis").mkdir(exist_ok=True)
    (O / "analysis/n931_c4_steps.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["curves"]), json.dumps(logs, indent=1))


if __name__ == "__main__":
    main()
