"""(review R1 W3b, author request 10-09): does ReMaRk take larger effective steps than Rank-GRPO?

Zero-GPU readout from the existing training logs of the main-table arms (seed 3407, P0c, 2,000 steps):
ReMaRk = <size>_union_group_s3407_p0c, Rank-GRPO = <size>_single_group_s3407_p0c, sizes 2B (main), 4B, 9B.
1. per-step curves of grad_norm (pre-clipping global norm, as logged by the HF trainer) and kl (TRL GRPO, mean
   per-token KL to the reference = SFT), logged every 10 steps;
2. mean and median over the logged steps 1..2,000;
3. at equal validation quality: the first step at which each arm's validation R@20 reaches Rank-GRPO's step-2,000
   validation R@20 (2B: 24.19; linear interpolation between the 250-step validation points, as n937_req2x.py), and the
   grad_norm / kl there (mean of the logged values within +-50 steps, plus the nearest logged value).
Step indices: a log's tqdm bar k/N gives the step of the trainer dict lines that follow it; a resumed segment trains
N = 2000 - k0 steps, so its offset is k0 = 2000 - N. Where segments overlap, the later segment (larger offset) is used
for its steps (the earlier one was preempted after the checkpoint it resumed from). Identical log copies count once.
Usage: env/bin/python remark/scripts/n937_gradnorm_kl.py
Output: outputs/crs/analysis/n937_gradnorm_kl.json
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
OUT = C / "analysis/n937_gradnorm_kl.json"
WANT = 2000
BAR = re.compile(r"(\d+)/(\d+) \[")
SIZES = ["2B", "4B", "9B"]
ARMS = {"remark": "{}_union_group_s3407_p0c", "rankgrpo": "{}_single_group_s3407_p0c"}
KEYS = ["grad_norm", "kl"]


def segment(path: Path) -> tuple[int, dict[int, dict]]:
    """(offset, {global step: entry}) of one log segment."""
    cur, total, out = None, None, {}
    for line in path.read_text(errors="ignore").replace("\r", "\n").splitlines():
        s = line.strip()
        for m in BAR.finditer(s):
            k, n = int(m.group(1)), int(m.group(2))
            if n <= WANT:
                cur, total = k, n
        if s.startswith("{'loss'") and cur is not None:
            try:
                d = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                continue
            e = {k: float(d[k]) for k in KEYS if k in d}
            if e:
                out[cur] = e
    off = WANT - total if total else 0
    return off, {off + k: v for k, v in out.items()}


def series(run: str) -> tuple[dict[int, dict], list[dict]]:
    seen, segs = set(), []
    for p in sorted(C.glob(f"train_logs/*/grpo_{run}.log")):
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        off, d = segment(p)
        if d:
            segs.append((off, d, str(p.relative_to(C))))
    merged: dict[int, dict] = {}
    for off, d, _ in sorted(segs):
        merged = {k: v for k, v in merged.items() if k < off}
        merged.update(d)
    info = [{"log": f, "offset": off, "steps": [min(d), max(d)], "n_logged": len(d)} for off, d, f in sorted(segs)]
    return dict(sorted(merged.items())), info


def val_curve(run: str) -> dict[int, float]:
    out = {}
    for p in (C / "val").glob(f"grpo_qwen_{run}_step*/summary.json"):
        m = re.search(r"_step(\d+)$", p.parent.name)
        if m:
            out[int(m.group(1))] = 100 * json.load(open(p))["truncate_R@20"]
    return dict(sorted(out.items()))


def first_reach(curve: dict[int, float], target: float) -> float | None:
    pts = sorted(curve.items())
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if v0 >= target:
            return float(s0)
        if v1 >= target:
            return s0 + (target - v0) / (v1 - v0) * (s1 - s0)
    return float(pts[0][0]) if pts and pts[0][1] >= target else None


def at_step(ser: dict[int, dict], step: float, key: str) -> dict:
    win = [v[key] for k, v in ser.items() if abs(k - step) <= 50 and key in v]
    near = min(ser, key=lambda k: abs(k - step))
    return {"mean_pm50": round(float(np.mean(win)), 5) if win else None, "n": len(win),
            "nearest_step": near, "nearest": ser[near].get(key)}


def main() -> None:
    res = {"doc": __doc__.split("Usage")[0].strip(), "sizes": {}}
    for size in SIZES:
        runs = {a: t.format(size) for a, t in ARMS.items()}
        vals = {a: val_curve(r) for a, r in runs.items()}
        target = vals["rankgrpo"].get(WANT)
        block = {"runs": runs, "target_val_R20": round(target, 2) if target else None, "arms": {}}
        for arm, run in runs.items():
            ser, info = series(run)
            if not ser:
                block["arms"][arm] = {"run": run, "missing": "no training log"}
                continue
            steps = sorted(ser)
            missing = [s for s in range(10, WANT + 1, 10) if s not in ser]
            a = {"run": run, "segments": info, "n_logged": len(ser),
                 "missing_logged_steps": [missing[0], missing[-1], len(missing)] if missing else [],
                 "curve": {k: [ser[s].get(k) for s in steps] for k in KEYS}, "steps": steps}
            for k in KEYS:
                x = np.array([ser[s][k] for s in steps if k in ser[s] and s <= WANT])
                a[f"{k}_mean"] = round(float(x.mean()), 5)
                a[f"{k}_median"] = round(float(np.median(x)), 5)
                a[f"{k}_max"] = [round(float(x.max()), 4), steps[int(x.argmax())]]
            a["kl_gt1_steps"] = [s for s in steps if ser[s].get("kl", 0) > 1]  # KL spikes (instability)
            kx = np.array([ser[s]["kl"] for s in steps if "kl" in ser[s] and ser[s]["kl"] <= 1])
            a["kl_mean_excl_gt1"] = round(float(kx.mean()), 5)
            if target is not None:
                st = first_reach(vals[arm], target)
                a["reach_target_step"] = round(st, 1) if st is not None else None
                if st is not None:
                    for k in KEYS:
                        a[f"{k}_at_target"] = at_step(ser, st, k)
            a["val_R20"] = {s: round(v, 2) for s, v in vals[arm].items()}
            block["arms"][arm] = a
        r, g = block["arms"].get("remark", {}), block["arms"].get("rankgrpo", {})
        if "grad_norm_mean" in r and "grad_norm_mean" in g:
            block["ratio_remark_over_rankgrpo"] = {f"{k}_{s}": round(r[f"{k}_{s}"] / g[f"{k}_{s}"], 3)
                                                   for k in KEYS for s in ("mean", "median")}
        res["sizes"][size] = block
    OUT.write_text(json.dumps(res, indent=1))
    for size, b in res["sizes"].items():
        print(size, "target", b["target_val_R20"], b.get("ratio_remark_over_rankgrpo"))
        for arm, a in b["arms"].items():
            if "grad_norm_mean" not in a:
                print(" ", arm, a); continue
            print(f"  {arm}: n={a['n_logged']} miss={a['missing_logged_steps']} gn mean/med {a['grad_norm_mean']:.4f}/"
                  f"{a['grad_norm_median']:.4f} kl mean/med {a['kl_mean']:.5f}/{a['kl_median']:.5f} reach@{a.get('reach_target_step')}"
                  f" gn {a.get('grad_norm_at_target', {}).get('mean_pm50')} kl {a.get('kl_at_target', {}).get('mean_pm50')}")
            print("    segs", [(s['log'].split('/')[-2], s['offset'], s['steps']) for s in a['segments']])


if __name__ == "__main__":
    main()
