"""Sample-efficiency framing: per-step compute of ReMaRk vs Rank-GRPO from existing logs.

Sources: the TRL log dicts in outputs/crs/train_logs/**/grpo_<run>[_seg<k>].log (one dict per 10 steps) and, for the
2B 8k plateau pair, the log_history in the selected checkpoints' trainer_state.json (the node logs of earlier segments
were overwritten or lost with the nodes). Per logged interval TRL reports step_time (mean seconds per optimizer step,
generation + scoring + forward/backward), completions/mean_length, and num_tokens (cumulative prompt + completion tokens
over all processes). Per arm:
  s_per_step      mean and median of step_time over intervals, excluding the first 30 steps of every segment (Triton
                  autotuning and vLLM warm-up after a start or resume)
  completion_tok  mean completion length x 384 lists per step
  total_tok       mean per-step increase of num_tokens (prompt + completion, all processes)
  prompt_tok      total_tok - completion_tok
GPU-hours of the 2B 8k pair to a step S = 4 GPUs x sum of step_time over steps <= S where logged, plus steps without a
log at the arm's mean step time (reported separately).
Usage: env/bin/python remark/scripts/n928_step_cost.py -> outputs/crs/analysis/n928_step_cost.json
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
L = C / "train_logs"
LISTS = 384  # 16 per device x 6 accumulation x 4 GPUs = 48 requests x 8 lists (both arms, every size)
PAIRS = {"2B": ("2B_single_group_s3407_p0c", "2B_union_group_s3407_p0c"),
         "4B": ("4B_single_group_s3407_p0c", "4B_union_group_s3407_p0c"),
         "9B": ("9B_single_group_s3407_p0c", "9B_union_group_s3407_p0c"),
         "27B": ("27B_single_group_s3407_p0c", "27B_union_group_s3407_p0c"),
         "35BA3B": ("35BA3B_single_group_s3407_p0c", "35BA3B_union_group_s3407_p0c")}
SKIP = 30


def entries_from_log(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="ignore").replace("\r", "\n").splitlines():
        line = line.strip()
        if line.startswith("{'loss'") and "'step_time'" in line:
            try:
                d = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            out.append({k: float(v) for k, v in d.items() if re.fullmatch(r"[-+0-9.e]+", str(v))})
    return out


def segments(run: str) -> list[list[dict]]:
    """Distinct segment logs of a run (identical copies under several node dirs counted once)."""
    seen, segs = set(), []
    for p in sorted(L.glob(f"*/grpo_{run}.log")) + sorted(L.glob(f"*/grpo_{run}_seg*.log")):
        if "discarded" in str(p) or "aborted" in p.name:
            continue
        e = entries_from_log(p)
        key = (len(e), e[0]["num_tokens"] if e else None, e[-1]["num_tokens"] if e else None)
        if e and key not in seen:
            seen.add(key); segs.append(e)
    return segs


def summarize(segs: list[list[dict]]) -> dict:
    st, ml, tok = [], [], []
    for e in segs:
        for i, x in enumerate(e):
            if i < SKIP // 10:  # entries are every 10 steps within a segment
                continue
            st.append(x["step_time"]); ml.append(x["completions/mean_length"])
            if i > 0:
                tok.append((x["num_tokens"] - e[i - 1]["num_tokens"]) / 10)
    st, ml, tok = map(np.array, (st, ml, tok))
    comp = ml.mean() * LISTS
    return {"intervals": int(len(st)), "s_per_step_mean": float(st.mean()), "s_per_step_median": float(np.median(st)),
            "completion_len_mean": float(ml.mean()), "completion_tok_per_step": float(comp),
            "total_tok_per_step": float(tok.mean()), "prompt_tok_per_step": float(tok.mean() - comp)}


def plateau_pair() -> dict:
    """2B 8k pair: GPU-hours to the best and to the stop step."""
    out = {}
    arms = {"remark": ("2B_union_group_s3407_p0c_ms8000", 4000, 5500, ["checkpoint-4000"]),
            "rankgrpo": ("2B_single_group_s3407_p0c_ms8000", 5250, 6750, ["checkpoint-5250"])}
    for arm, (run, best, stop, ckpts) in arms.items():
        logged = {}  # global step -> step_time (mean over the 10 steps ending there)
        segs = []
        for ck in ckpts:
            ts = json.load(open(C / "grpo_qwen_selected" / run / ck / "trainer_state.json"))
            lh = [x for x in ts["log_history"] if "step_time" in x]
            offset = int(ck.split("-")[1]) - ts["global_step"]  # resumed segments restart the trainer's step count
            segs.append([dict(x, step=x["step"] + offset) for x in lh])
        for p in L.glob(f"*/grpo_{run}.log"):  # the last segment's node log (Rank-GRPO: 6,250 -> 6,750)
            e = entries_from_log(p)
            if e:
                segs.append([dict(x, step=None) for x in e])
        for s in segs:
            for x in s:
                if x.get("step") is not None:
                    logged[int(x["step"])] = x["step_time"]
        # node-log segment without global steps: Rank-GRPO's last segment resumed at 6,250
        if arm == "rankgrpo":
            e = segs[-1]
            for i, x in enumerate(e):
                logged[6250 + 10 * (i + 1)] = x["step_time"]
        mean_st = float(np.mean([v for k, v in logged.items()]))
        res = {"run": run, "best_step": best, "stop_step": stop, "mean_s_per_step": mean_st}
        for name, S in (("best", best), ("stop", stop)):
            have = sorted(k for k in logged if k <= S)
            sec_logged = 10 * sum(logged[k] for k in have)
            missing_steps = S - 10 * len(have)
            res[f"to_{name}"] = {"steps": S, "steps_logged": 10 * len(have), "steps_estimated": missing_steps,
                                 "gpu_hours": 4 * (sec_logged + missing_steps * mean_st) / 3600}
        out[arm] = res
    return out


def main() -> None:
    res = {"note": __doc__.split("\n")[0], "pairs": {}, "plateau_2B": plateau_pair()}
    for size, (rg, rm) in PAIRS.items():
        a, b = segments(rg), segments(rm)
        if not a or not b:
            res["pairs"][size] = None
            continue
        res["pairs"][size] = {"rankgrpo": summarize(a) | {"segments": len(a)}, "remark": summarize(b) | {"segments": len(b)}}
        r = res["pairs"][size]
        print(size, {k: (round(v["s_per_step_mean"], 2), round(v["s_per_step_median"], 2), round(v["completion_tok_per_step"]),
                         round(v["prompt_tok_per_step"]), v["segments"]) for k, v in r.items()})
    print(json.dumps(res["plateau_2B"], indent=1))
    (C / "analysis/n928_step_cost.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
