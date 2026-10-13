"""Reviewer checks K1/K2 (registered 2026-10-05 05:30): steps-to-match with a request bootstrap, as C2.

Check 1: ReMaRk 2B vs the coverage-matched per-responder arm (request sampling + group scale): C1 counts and C2 s*.
K2: Rank-GRPO 2B at lr 2e-6 / 3e-6 reaching Rank-GRPO lr 1e-6's step-2,000 validation (24.19).
Uses n931_c1c2's per-row cache (adds the new runs to it), its crossing() and request groups. Fresh rng (seed 42) per check.
Output: outputs/crs/analysis/n932_reviewer_checks.json
Usage: env/bin/python remark/scripts/n932_reviewer_checks.py [--workers 32]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import n931_c1c2 as c

SFT = "sft_qwen35_2B_step1500"
RUNS = ["2B_union_group_s3407_p0c", "2B_singlethr_group_s3407_p0c", "2B_single_group_s3407_p0c",
        "2B_single_group_s3407_p0c_lr2e-6", "2B_single_group_s3407_p0c_lr3e-6"]


def boot_weights(n_req: int, sizes: np.ndarray):
    rng = np.random.default_rng(42)
    draws = rng.integers(0, n_req, size=(2000, n_req))
    w = np.stack([np.bincount(d, minlength=n_req) for d in draws])
    return w, (w * sizes).sum(1)


def sums(curve: dict, g: np.ndarray, n_req: int):
    ks = np.array(sorted(curve))
    return ks, np.stack([np.bincount(g, weights=curve[k], minlength=n_req) for k in ks])


def steps_to(arm: dict, ref: dict, g, n_req, sizes, ref_step: int = 2000) -> dict:
    """s* at which `arm` first reaches `ref`'s validation at ref_step; bootstrap over requests."""
    ka, sa = sums(arm, g, n_req); kr, sr = sums(ref, g, n_req)
    j = list(kr).index(ref_step)
    def est(wv, rw):
        return c.crossing(ka.astype(float), (sa @ wv) / rw, ((sr @ wv) / rw)[j])
    w, rows_w = boot_weights(n_req, sizes)
    s0 = est(np.ones(n_req), float(sizes.sum()))
    bs = np.array([est(w[i].astype(float), float(rows_w[i])) for i in range(2000)])
    fin = np.isfinite(bs)
    return {"target": float(ref[ref_step].mean()), "s_star": s0, "ratio": s0 / ref_step,
            "ratio_ci": [float(np.percentile(np.where(fin, bs, np.inf), q)) / ref_step for q in (2.5, 97.5)],
            "share_no_crossing": float(1 - fin.mean())}


def c1(a: dict, b: dict, g, n_req, sizes) -> dict:
    w, rows_w = boot_weights(n_req, sizes)
    out = {}
    for k in sorted(set(a) & set(b) - {0}):
        d = a[k] - b[k]
        bt = (w @ np.bincount(g, weights=d, minlength=n_req)) / rows_w
        out[str(k)] = [float(d.mean()), *map(float, np.percentile(bt, [2.5, 97.5]))]
    return {"per_step": out, "ci_above_0": sum(v[1] > 0 for v in out.values()),
            "positive": sum(v[0] > 0 for v in out.values()), "K": len(out)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=32); a = ap.parse_args()
    c.CACHE.mkdir(parents=True, exist_ok=True)
    jobs = sorted({("reddit", str(d)) for r in RUNS for d in c.steps_of(r).values()} | {("reddit", str(c.V / SFT))})
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(c._work, jobs, chunksize=2))
    g = c.groups("reddit"); n_req = g.max() + 1; sizes = np.bincount(g, minlength=n_req)
    cv = {r: c.curve(r, SFT) for r in RUNS}
    pts = {r: {k: round(100 * float(v.mean()), 2) for k, v in cv[r].items()} for r in RUNS}
    remark, r3, rg = cv["2B_union_group_s3407_p0c"], cv["2B_singlethr_group_s3407_p0c"], cv["2B_single_group_s3407_p0c"]
    res = {"note": __doc__.split("\n")[0], "curves_val_R@20": pts,
           "K1": {"c1_remark_vs_r3": c1(remark, r3, g, n_req, sizes), "c2_remark_to_r3_2000": steps_to(remark, r3, g, n_req, sizes),
                  "c2_r3_to_rankgrpo_2000": steps_to(r3, rg, g, n_req, sizes),
                  "c2_remark_to_rankgrpo_2000": steps_to(remark, rg, g, n_req, sizes)},
           "K2": {lr: steps_to(cv[f"2B_single_group_s3407_p0c_lr{lr}"], rg, g, n_req, sizes) for lr in ("2e-6", "3e-6")}}
    (c.A / "n932_reviewer_checks.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "curves_val_R@20"}, indent=1)[:4000])
    for r, p in pts.items():
        print(r, " ".join(f"{k}:{v}" for k, v in p.items()))


if __name__ == "__main__":
    main()
