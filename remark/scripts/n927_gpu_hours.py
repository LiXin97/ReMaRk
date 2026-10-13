"""total GPU-hours of the work reported in the paper (training + evaluation), for the Ethics statement.

Training: every outputs/crs/train_logs/**/grpo_*.log and sft_*.log; the wall-clock of a run is the last elapsed time in
its tqdm bar (HH:MM:SS or H:MM:SS before the "<"), summed over the segments of a resumed run. GPUs per run: 4 for the
2B/4B/9B/Llama/Books slot runs (two runs share a node), 8 for the 27B and 35B-A3B runs and for every SFT.
Evaluation: outputs/crs/{val,test}/*/summary.json, one GPU each. crs_eval.py records no duration and writes both of
its files at the end, so the per-evaluation wall-clock is taken from three 27B evaluations measured end to end on
a GPU node (validator claim file -> summary: 9, 10, 10 min, i.e. model load plus generation) and the generation-phase
time in the node logs for the smaller models (Llama-8B 8-9 min, Books 4B 0.5 min, 35B-A3B 1 min of generation).
EVAL_MIN below is the per-model-size estimate used; it is deliberately generous for the small models, where loading
dominates. API-only evaluations (GPT-5.6 zero-shot, the judge) use no GPU and are excluded.

Numbers are a lower bound on cluster time: they exclude staging, model download, node idle time and preempted segments
whose logs were lost with the node.

Usage: env/bin/python remark/scripts/n927_gpu_hours.py
Output: outputs/crs/analysis/n927_gpu_hours.json
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
BAR = re.compile(r"(\d+)/(\d+) \[(\d+:\d+:\d+|\d+:\d+)<")
# measured minutes per evaluation (load + generation), one GPU each; see the module docstring
EVAL_MIN = {"2B": 4.0, "4B": 5.0, "9B": 7.0, "llama3b": 4.0, "llama8b": 10.0, "27B": 10.0, "35BA3B": 10.0}


def elapsed_hours(path: Path) -> tuple[float, int]:
    """Last tqdm elapsed time in the log, in hours, and the step reached."""
    try:
        txt = path.read_text(errors="ignore").replace("\r", "\n")
    except OSError:
        return 0.0, 0
    last = None
    for m in BAR.finditer(txt):
        last = m
    if not last:
        return 0.0, 0
    parts = [int(x) for x in last.group(3).split(":")]
    h = parts[0] + parts[1] / 60 + (parts[2] / 3600 if len(parts) == 3 else 0) if len(parts) == 3 else parts[0] / 60 + parts[1] / 3600
    return h, int(last.group(1))


def gpus_for(name: str, log: Path | None = None) -> int:
    if log is not None and str(log.relative_to(C / "train_logs")) in EIGHT_GPU_LOGS:
        return 8
    return 8 if ("27B" in name or "35BA3B" in name or name.startswith("sft_")) else 4


# 10-09: slot runs that trained a segment on all 8 GPUs of a node (req2x s42 resume 1250 on node-x2, GA 6)
EIGHT_GPU_LOGS = {"node-x2/grpo_2B_single_group_s42_p0c_req2x.log"}
# 10-09: training whose node logs were lost with the node (platform kills 10-08/09), recovered from other records and
# counted as GPU-hours: run -> (gpus, steps, seconds per step, source)
RECOVERED = {
    "grpo_27B_single_batch_s3407_p0c": (8, 2000, 30.237, "trainer_state step_time mean, grpo_qwen_selected/.../checkpoint-2000"),
    "grpo_27B_union_batch_s3407_p0c": (8, 2000, 31.616, "trainer_state step_time mean, grpo_qwen_selected/.../checkpoint-2000"),
    "grpo_2B_single_group_s42_p0c_req2x": (4, 1440, 13.095, "segments 0..500 (node-p) and 500..~1440 (node-x); 4-GPU tqdm "
                                      "s/step, analysis/n937_req2x_s42_steptime_note.json"),
}


def main() -> None:
    runs: dict[str, dict] = {}
    seen: set[str] = set()  # byte-identical copies of a log (e.g. _final/ and the node dir) are counted once
    for log in sorted((C / "train_logs").rglob("*.log")):
        if not (log.name.startswith("grpo_") or log.name.startswith("sft_")):
            continue
        digest = hashlib.sha1(log.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        h, step = elapsed_hours(log)
        if h <= 0:
            continue
        # a resumed run is split over <run>_seg<k>.log files; group them under the run name
        run = re.sub(r"_seg\d+$", "", log.stem)
        discarded = "discarded" in str(log)
        d = runs.setdefault(run, {"gpu_hours": 0.0, "gpus": gpus_for(run), "segments": 0, "discarded": discarded})
        d["gpu_hours"] += h * gpus_for(run, log)
        d["segments"] += 1
        d["discarded"] = d["discarded"] and discarded
    rec_gpu_h = 0.0
    for run, (g, steps, sps, src) in RECOVERED.items():
        d = runs.setdefault(run, {"gpu_hours": 0.0, "gpus": g, "segments": 0, "discarded": False})
        d["gpu_hours"] += g * steps * sps / 3600
        d["recovered"] = {"gpus": g, "steps": steps, "s_per_step": sps, "source": src}
        rec_gpu_h += g * steps * sps / 3600
    train_gpu_h = sum(d["gpu_hours"] for d in runs.values())
    kept_gpu_h = sum(d["gpu_hours"] for d in runs.values() if not d["discarded"])

    by_size: dict[str, int] = {}
    minutes = 0.0
    for s in list((C / "val").glob("*/summary.json")) + list((C / "test").glob("*/summary.json")):
        n = s.parent.name
        size = ("27B" if "27B" in n else "35BA3B" if "35BA3B" in n else "9B" if "9B" in n else
                "llama8b" if "llama8b" in n else "4B" if "4B" in n else "llama3b" if "llama3b" in n else "2B")
        by_size[size] = by_size.get(size, 0) + 1
        minutes += EVAL_MIN[size]
    evals = sum(by_size.values())
    eval_gpu_h = minutes / 60  # one GPU per evaluation

    res = {
        "training": {"runs": len(runs), "gpu_hours": round(train_gpu_h, 1),
                     "gpu_hours_excluding_discarded": round(kept_gpu_h, 1),
                     "node_hours_8gpu": round(train_gpu_h / 8, 1),
                     "recovered_from_lost_logs_gpu_hours": round(rec_gpu_h, 1)},
        "evaluation": {"runs": evals, "by_size": by_size, "minutes_per_eval": EVAL_MIN,
                       "gpu_hours": round(eval_gpu_h, 1)},
        "total_gpu_hours": round(train_gpu_h + eval_gpu_h, 1),
        "note": "lower bound: excludes staging, downloads, idle time and preempted segments whose logs were lost (except the "
                "RECOVERED runs, counted from trainer_state / measured step times); excludes the 27B/35B-A3B SFT runs (no "
                "logs or timing records kept); "
                "API-only evaluations (GPT-5.6 zero-shot and the judge) use no GPU",
        "by_run": {k: {"gpu_hours": round(v["gpu_hours"], 1), "gpus": v["gpus"], "segments": v["segments"],
                       "discarded": v["discarded"], **({"recovered": v["recovered"]} if "recovered" in v else {})}
                   for k, v in sorted(runs.items())},
    }
    out = C / "analysis/n927_gpu_hours.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "by_run"}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
