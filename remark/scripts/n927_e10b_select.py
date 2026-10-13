"""Select the DPO configuration on validation and test the selected checkpoint once.

Candidates for a size: every validated DPO run of that size (the one-epoch run <size>_dpo_union_s3407 and the sweep
runs <size>_dpo_union_lr<lr>_ep<ep>_s3407). For each, its best validation per-row R@20 (ties to the earlier step).
Selected: the highest; ties -> fewer epochs, then the lower lr, then the earlier step. Writes
outputs/crs/locks/<size>_dpo_union_selected.json (all curves, the selection, whether the selected curve is still rising at
its last checkpoint) and tests the selected checkpoint once unless that run/step already has a test.

: --prefix dpo_strong selects among the stronger-DPO runs <size>_dpo_strong_lr<lr>_ep<ep>_s3407 the same way
(lock <size>_dpo_strong_selected.json); the default prefix dpo_union is the original DPO behaviour.
Usage (node, system python3 with vLLM): python3 remark/scripts/n927_e10b_select.py --size 2B --gpu 4 [--prefix dpo_union]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import socket
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"


def curve(run: str) -> dict[int, float]:
    out = {}
    for p in (C / "val").glob(f"grpo_qwen_{run}_step*/summary.json"):
        m = re.search(r"_step(\d+)$", p.parent.name)
        if m:
            out[int(m.group(1))] = json.load(open(p))["truncate_R@20"]
    return dict(sorted(out.items()))


def cfg_of(run: str) -> tuple[float, float]:
    m = re.search(r"_lr([0-9.e-]+)_ep([0-9.]+)_", run)
    return (float(m.group(2)), float(m.group(1))) if m else (1.0, 1e-6)  # (epochs, lr); the one-epoch run is lr 1e-6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--prefix", default="dpo_union")
    a = ap.parse_args()
    runs = sorted({p.parent.name.split("_step")[0][len("grpo_qwen_"):]
                   for p in (C / "val").glob(f"grpo_qwen_{a.size}_{a.prefix}*_step*/summary.json")})
    cands = []
    for r in runs:
        cv = curve(r)
        if not cv:
            continue
        best = max(sorted(cv), key=lambda s: cv[s])
        ep, lr = cfg_of(r)
        last = max(cv)
        cands.append({"run": r, "epochs": ep, "lr": lr, "best_step": best, "best_val": cv[best], "curve": cv,
                      "still_rising_at_end": best == last and len(cv) > 1 and cv[last] > cv[sorted(cv)[-2]]})
    if not cands:
        raise SystemExit(f"no validated DPO run for {a.size}")
    sel = sorted(cands, key=lambda c: (-c["best_val"], c["epochs"], c["lr"], c["best_step"]))[0]
    lock = C / "locks" / f"{a.size}_{a.prefix}_selected.json"
    if not lock.exists():
        lock.write_text(json.dumps({"size": a.size, "selected": {k: sel[k] for k in ("run", "epochs", "lr", "best_step",
                                    "best_val", "still_rising_at_end")}, "candidates": cands,
                                    "rule": "highest best-checkpoint validation per-row R@20; ties: fewer epochs, lower lr, "
                                            "earlier step (amendment); tested once",
                                    "locked_at_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                                    "host": socket.gethostname()}, indent=2, default=str))
    s = json.loads(lock.read_text())["selected"]
    name = f"test_{s['run']}_step{s['best_step']}"
    if (C / "test" / name / "summary.json").exists():
        print("already tested:", name); return
    ck = C / "dpo_mirror" / s["run"] / f"checkpoint-{s['best_step']}"
    out = Path("/tmp/work_local/test"); out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "NCCL_SOCKET_IFNAME", "NCCL_SOCKET_FAMILY",
                                                             "NCCL_NET", "NCCL_NET_PLUGIN")}
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), VLLM_WORKER_MULTIPROC_METHOD="spawn")
    subprocess.run(["python3", str(REPO / "remark/scripts/crs_eval.py"), "--model", str(ck), "--name", name,
                    "--split", "test", "--out", str(out)], env=env, check=True)
    subprocess.run(["rsync", "-a", str(out / name), str(C / "test") + "/"], check=True)
    print("tested", name)


if __name__ == "__main__":
    main()
