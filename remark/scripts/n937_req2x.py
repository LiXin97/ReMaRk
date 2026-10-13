"""(review item 2, author spec 10-08): Rank-GRPO with twice the requests per step ("req2x") vs ReMaRk at equal compute.

req2x = 2B Rank-GRPO (per-responder, row sampling, group scaling, P0c) with gradient_accumulation_steps 6 -> 12, i.e. 96
requests per step instead of 48; G = 8, lr 1e-6, KL unchanged; 2,000 steps, validation every 250, best-validation lock,
tested once. Runs 2B_single_group_<seed>_p0c_req2x. Compared with the same-seed main-table runs:
ReMaRk 2B_union_group_<seed>_p0c and Rank-GRPO 2B_single_group_<seed>_p0c.
1. validation R@20 curves by step and by GPU-h. GPU-h (n927 definition) = 4 GPUs x training wall-clock from the tqdm
   elapsed time in the training log at that step (validation runs outside the trainer, so it is excluded).
2. test: req2x locked test minus ReMaRk and minus Rank-GRPO main tests, per row and per request (crs_compare paired
   bootstrap over contexts, 2,000 resamples, seed 42).
3. main readout: req2x validation at the GPU-h of ReMaRk step 2,000 (linear interpolation on the GPU-h axis) and at
   step 1,000, vs ReMaRk validation at step 2,000.
4. steps and GPU-h to first reach Rank-GRPO's step-2,000 validation (24.19 for s3407; linear interpolation between
   validation points), req2x vs ReMaRk.
5. groups with signal per step: (1 - rank/frac_group_zero_std) x groups per step (requests per step: 96 req2x, 48 the
   main runs), mean over the logged steps.
Usage: env/bin/python remark/scripts/n937_req2x.py [s3407 s42]
Output: outputs/crs/analysis/n937_req2x.json
"""
from __future__ import annotations

import ast
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_compare import DATA, RG, context_groups, paired_stats, per_row  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
OUT = C / "analysis/n937_req2x.json"
BAR = re.compile(r"(\d+)/(\d+) \[(\d+):(\d+)(?::(\d+))?<")
TEST = {"s3407": {"remark": "test_2B_union_group_step2000", "rankgrpo": "test_2B_single_group_s3407_step2000"},
        "s42": {"remark": "test_2B_union_group_s42_p0c_step2000", "rankgrpo": "test_2B_single_group_s42_step2000"}}
LOGS = {"s3407": {"remark": "node-b/grpo_2B_union_group_s3407_p0c.log", "rankgrpo": "node-a/grpo_2B_single_group_s3407_p0c.log"},
        # s42 ReMaRk main = segment 0 on node-e (0..1518, preempted) + resume from 500 on node-h (1,500 steps)
        "s42": {"remark": ["node-e/grpo_2B_union_group_s42_p0c.log", "node-h/grpo_2B_union_group_s42_p0c.log"],
                "rankgrpo": "node-d/grpo_2B_single_group_s42_p0c.log"}}
# 10-09: req2x s42 resumed twice (500 on node-x, 1250 on 8 GPUs of node-x2), each with a fresh optimizer; the node logs of
# 0..~1440 are lost. author rule (10-09): main GPU-h on the 4-GPU basis = measured 4-GPU s/step x steps, extrapolated to 2,000
# (13.095 s/step: tqdm 905 steps in 3:17:31 on node-x, analysis/n937_req2x_s42_steptime_note.json); the 8-GPU segment's
# measured wall-clock goes only into a note.
FIXED_STEP_S = {"s42": 13.095}
REQ_PER_STEP = {"req2x": 96, "remark": 48, "rankgrpo": 48}


def curve(run: str) -> dict[int, float]:
    out = {}
    for p in (C / "val").glob(f"grpo_qwen_{run}_step*/summary.json"):
        m = re.search(r"_step(\d+)$", p.parent.name)
        if m:
            out[int(m.group(1))] = 100 * json.load(open(p))["truncate_R@20"]
    return dict(sorted(out.items()))


def _bar(log: Path, total: int = 2000) -> tuple[int, dict[int, float]]:
    """(offset, {local step: elapsed hours}) of one segment log; a resumed segment trains N = total - offset steps."""
    out, n = {}, None
    for m in BAR.finditer(log.read_text(errors="ignore").replace("\r", "\n")):
        k, nn = int(m.group(1)), int(m.group(2))
        if nn > total:
            continue
        a, b, c = int(m.group(3)), int(m.group(4)), m.group(5)
        out[k] = a + b / 60 + int(c) / 3600 if c is not None else a / 60 + b / 3600
        n = nn
    return (total - n if n else 0), out


def gpu_h_by_step(logs: Path | list[Path], gpus: int = 4) -> dict[int, float]:
    """gpus x tqdm elapsed hours at each global step, chained over resumed segments (each segment is used from its
    offset until the next segment's offset; steps a preempted segment trained beyond that are not counted)."""
    segs = sorted(_bar(p) for p in (logs if isinstance(logs, list) else [logs]))
    out, base = {}, 0.0
    for i, (off, el) in enumerate(segs):
        end = segs[i + 1][0] if i + 1 < len(segs) else None
        for k, h in sorted(el.items()):
            if end is None or off + k <= end:
                out[off + k] = gpus * (base + h)
        if end is not None:
            base += el[max(k for k in el if off + k <= end)]
    return out


def signal(log: Path | list[Path], n_req: int, extra: list[float] | None = None) -> dict:
    def vals(lg: Path) -> list[float]:
        return [float(ast.literal_eval(m.group(0))["rank/frac_group_zero_std"])
                for m in re.finditer(r"\{'loss'[^}]*\}", lg.read_text(errors="ignore")) if "rank/frac_group_zero_std" in m.group(0)]
    # resumed segments: entry j (logging every 10 steps) of a segment at offset o is global step o + 10 j; a preempted segment
    # is used only up to the next segment's offset (as in gpu_h_by_step)
    segs = sorted((_bar(lg)[0], vals(lg)) for lg in (log if isinstance(log, list) else [log]))
    fr = []
    for i, (off, v) in enumerate(segs):
        end = segs[i + 1][0] if i + 1 < len(segs) else None
        fr += [x for j, x in enumerate(v, 1) if end is None or off + 10 * j <= end]
    fr += extra or []
    z = float(np.mean(fr)) if fr else None
    return {"logged_steps": len(fr), "frac_group_zero_std": z, "requests_per_step": n_req,
            "signal_groups_per_step": None if z is None else (1 - z) * n_req}


def trainer_state_zero_std(run: str) -> list[float]:
    """frac_group_zero_std from trainer_state copies of segments whose node log is lost (train_logs/*/trainer_state_<run>_*)."""
    out = []
    for p in (C / "train_logs").glob(f"*/trainer_state_{run}_seg*.json"):
        out += [float(x["rank/frac_group_zero_std"]) for x in json.load(open(p))["log_history"] if "rank/frac_group_zero_std" in x]
    return out


def first_reach(cv: dict[int, float], gh: dict[int, float], target: float) -> dict | None:
    pts = [(0, -1e9)] + list(cv.items())
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if v1 >= target:
            s = s1 if v0 < -1e8 else s0 + (target - v0) / (v1 - v0) * (s1 - s0)
            return {"step": s, "gpu_h": float(np.interp(s, sorted(gh), [gh[k] for k in sorted(gh)]))}
    return None


def main() -> None:
    seeds = sys.argv[1:] or ["s3407"]
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

    res: dict = {"note": __doc__.split("Usage")[0].strip(), "units": "points (x100); GPU-h", "seeds": {}}
    for sd in seeds:
        runs = {"req2x": f"2B_single_group_{sd}_p0c_req2x", "remark": f"2B_union_group_{sd}_p0c",
                "rankgrpo": f"2B_single_group_{sd}_p0c"}
        req_logs = sorted((C / "train_logs").rglob(f"grpo_{runs['req2x']}.log"))
        logs = {"req2x": req_logs[-1] if len(req_logs) == 1 or sd not in FIXED_STEP_S else req_logs,
                **{k: ([C / "train_logs" / x for x in v] if isinstance(v, list) else C / "train_logs" / v)
                   for k, v in LOGS[sd].items()}}
        rel = lambda v: [str(x.relative_to(C)) for x in v] if isinstance(v, list) else str(v.relative_to(C))
        r: dict = {"runs": runs, "logs": {k: rel(v) for k, v in logs.items()}, "curves": {}}
        gh = {k: gpu_h_by_step(v) for k, v in logs.items() if k != "req2x" or sd not in FIXED_STEP_S}
        if sd in FIXED_STEP_S:
            gh["req2x"] = {k: 4 * FIXED_STEP_S[sd] * k / 3600 for k in range(1, 2001)}
            segs = [_bar(p) for p in (logs["req2x"] if isinstance(logs["req2x"], list) else [logs["req2x"]])]
            r["gpu_h_note"] = {"rule": "main GPU-h = 4 GPUs x %.3f s/step x steps (4-GPU measurement, extrapolated to 2,000)"
                               % FIXED_STEP_S[sd],
                               "resumes": "resumed twice (500, 1250), each with a fresh optimizer (seed 42+k)",
                               "lost_logs": "segments 0..500 (node-p) and 500..~1440 (node-x) lost with the nodes",
                               "measured_segments": [{"offset": o, "steps": max(e) if e else 0,
                                                      "elapsed_h": round(e[max(e)], 3) if e else None} for o, e in segs],
                               "8gpu_segment": "offset 1250 on node-x2 (8 GPUs, GA 6, same 96 requests/step): GPU-h = 8 x elapsed"}
            for sg in r["gpu_h_note"]["measured_segments"]:
                if sg["offset"] == 1250 and sg["elapsed_h"]:
                    sg["gpu_h_8gpu"] = round(8 * sg["elapsed_h"], 2)
                    sg["s_per_step"] = round(3600 * sg["elapsed_h"] / sg["steps"], 2)
        for k, run in runs.items():
            cv = curve(run)
            r["curves"][k] = {"by_step": cv, "gpu_h_at_val_step": {s: gh[k].get(s) for s in cv},
                              "total_gpu_h_2000": gh[k].get(2000),
                              "signal": signal(logs[k], REQ_PER_STEP[k], trainer_state_zero_std(run))}
        # 2. test
        lk = json.load(open(C / "locks" / f"{runs['req2x']}.json"))
        t = f"test_{runs['req2x']}_step{lk['best_step']}"
        r["test"] = {"req2x": t, "lock_step": lk["best_step"], "req2x_R@20": 100 * float(r20(t).mean()),
                     "minus_remark": cmp(t, TEST[sd]["remark"]), "minus_rankgrpo": cmp(t, TEST[sd]["rankgrpo"])}
        # 3. GPU-h aligned
        cq, ce = r["curves"]["req2x"]["by_step"], r["curves"]["remark"]["by_step"]
        g_e2000 = gh["remark"][2000]
        xs = [gh["req2x"][s] for s in cq]
        r["gpu_h_aligned"] = {"remark_gpu_h_at_2000": g_e2000, "remark_val_2000": ce[2000],
                              "req2x_gpu_h_at_1000": gh["req2x"][1000], "req2x_val_1000": cq[1000],
                              "req2x_val_at_remark_2000_gpu_h": float(np.interp(g_e2000, xs, list(cq.values()))),
                              "req2x_step_at_remark_2000_gpu_h": float(np.interp(g_e2000, [gh["req2x"][s] for s in sorted(gh["req2x"])],
                                                                                  sorted(gh["req2x"])))}
        a = r["gpu_h_aligned"]
        a["remark_minus_req2x_at_equal_gpu_h"] = a["remark_val_2000"] - a["req2x_val_at_remark_2000_gpu_h"]
        a["remark2000_minus_req2x1000"] = a["remark_val_2000"] - a["req2x_val_1000"]
        # 4. reach Rank-GRPO's step-2000 validation
        target = r["curves"]["rankgrpo"]["by_step"][2000]
        r["reach_rankgrpo_2000_val"] = {"target": target, **{k: first_reach(r["curves"][k]["by_step"], gh[k], target)
                                                             for k in ("req2x", "remark", "rankgrpo")}}
        res["seeds"][sd] = r
        print(f"== {sd}: val best req2x {max(cq.values()):.2f}, ReMaRk@2000 {ce[2000]:.2f}, RankGRPO@2000 {target:.2f}")
        print(f"GPU-h@2000: req2x {gh['req2x'][2000]:.1f}, ReMaRk {g_e2000:.1f}, RankGRPO {gh['rankgrpo'][2000]:.1f}")
        print(f"aligned: req2x val @ ReMaRk-2000 GPU-h ({a['req2x_step_at_remark_2000_gpu_h']:.0f} steps) "
              f"{a['req2x_val_at_remark_2000_gpu_h']:.2f}; req2x@1000 {cq[1000]:.2f} ({gh['req2x'][1000]:.1f} GPU-h)")
        for k in ("minus_remark", "minus_rankgrpo"):
            x = r["test"][k]
            print(f"test req2x {k}: {x['row']['a']:.2f} - {x['row']['b']:.2f} = row {x['row']['diff']:+.2f} "
                  f"[{x['row']['ci_low']:+.2f},{x['row']['ci_high']:+.2f}] unit {x['unit']['diff']:+.2f} "
                  f"[{x['unit']['ci_low']:+.2f},{x['unit']['ci_high']:+.2f}]")
        print("reach", json.dumps(r["reach_rankgrpo_2000_val"]))
        print("signal", {k: r["curves"][k]["signal"] for k in runs})
    OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
