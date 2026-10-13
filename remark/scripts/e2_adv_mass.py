"""Independent audit E2(b): share of the squared advantage that single-responder requests carry, per arm and scaling.

Input: the training-start reward cache of scripts/n928_r7c_batch_sigma.py (outputs/crs/analysis/n928_cache/
rewards_{single,union}.npz: 8 lists per training request sampled from the selected Qwen3.5-2B SFT checkpoint, per-position
0/1 rewards; single = one row per (request, responder), union = one row per request).
Advantages per position k: group scaling A = (R - mean_g R) / (std_g R + 1e-4) (Bessel, as the trainer; 0 when the group
std is 0); batch scaling divides by the per-position standard deviation over all rows instead (population value).
Sampling: Rank-GRPO draws rows (requests in proportion to n_x); ReMaRk draws requests uniformly.
Reports, per arm: share of sampled groups from n_x = 1 requests, share of groups with any signal, and the share of the
total squared advantage carried by n_x = 1 requests (also by n_x buckets). 95% CIs: bootstrap over requests
(1,000 resamples, seed 42).
Output: outputs/crs/analysis/e2_adv_mass.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/crs/analysis/n928_cache"
OUT = ROOT / "outputs/crs/analysis/e2_adv_mass.json"
EPS = 1e-4
BUCKETS = [(1, 1), (2, 4), (5, 19), (20, 10**9)]


def sq_mass(R: np.ndarray, scale: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-row sum over lists and positions of A^2, and whether the row's group has any signal."""
    R = R.astype(np.float32)
    c = R - R.mean(axis=1, keepdims=True)
    if scale == "group":
        sd = R.std(axis=1, ddof=1, keepdims=True)
        A = np.where(sd > 0, c / (sd + EPS), 0.0)
    else:
        sd = R.reshape(-1, R.shape[2]).std(axis=0, ddof=1)
        A = c / (sd + EPS)
    return (A ** 2).sum(axis=(1, 2)), (R.std(axis=1) > 0).any(axis=1)


def main() -> None:
    s = np.load(CACHE / "rewards_single.npz"); u = np.load(CACHE / "rewards_union.npz")
    req_s, req_u = s["req"], u["req"]
    n_req = len(req_u)
    n_x = np.bincount(req_s, minlength=n_req)
    assert (np.sort(req_u) == np.arange(n_req)).all()
    rng = np.random.default_rng(42)
    draws = rng.integers(0, n_req, size=(1000, n_req))
    W = np.stack([np.bincount(d, minlength=n_req) for d in draws]).astype(np.float64)  # request resample weights
    res = {"note": __doc__.split("\n")[0], "n_requests": int(n_req), "share_requests_n1": float((n_x == 1).mean())}
    for arm, (z, req) in {"rankgrpo_row": (s, req_s), "remark_request": (u, req_u)}.items():
        for scale in ("group", "batch"):
            m, sig = sq_mass(z["R"], scale)
            # per request totals
            mass = np.bincount(req, weights=m, minlength=n_req)
            groups = np.bincount(req, minlength=n_req).astype(np.float64)
            sigs = np.bincount(req, weights=sig.astype(np.float64), minlength=n_req)
            one = (n_x == 1).astype(np.float64)
            def stat(w):
                return {"groups_from_n1": float((w * groups * one).sum() / (w * groups).sum()),
                        "groups_with_signal": float((w * sigs).sum() / (w * groups).sum()),
                        "sq_adv_share_n1": float((w * mass * one).sum() / (w * mass).sum())}
            pt = stat(np.ones(n_req))
            bs = [stat(W[i]) for i in range(len(W))]
            out = {k: [round(pt[k], 4), *[round(float(np.percentile([b[k] for b in bs], q)), 4) for q in (2.5, 97.5)]]
                   for k in pt}
            out["sq_adv_share_by_nx"] = {f"{lo}-{hi if hi < 10**9 else 'inf'}":
                                         round(float(mass[(n_x >= lo) & (n_x <= hi)].sum() / mass.sum()), 4)
                                         for lo, hi in BUCKETS}
            out["mean_sq_adv_per_group_n1_vs_rest"] = [
                round(float(mass[n_x == 1].sum() / groups[n_x == 1].sum()), 3),
                round(float(mass[n_x > 1].sum() / groups[n_x > 1].sum()), 3)]
            res[f"{arm}_{scale}"] = out
            print(arm, scale, out, flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
