"""Share of the per-responder advantage variance that comes from which responder was drawn.

Input: outputs/crs/consensus_check/per_context_{reddit,books,music}_qwen2B.jsonl (crs_consensus_check2.py,
protocol: 500 training requests with >= 2 responders, 8 lists at temperature 1 from the dataset's 2B SFT policy;
S[r][g] = list-level NDCG@20 of list g against responder r).

For one request, the per-responder advantage of list g when responder r is drawn is a[r][g] = S[r][g] - mean_g S[r]
(centred, as GRPO's numerator) or that divided by the group's standard deviation (group-scaled, as Rank-GRPO).
Averaging over responders gives the Rao-Blackwellized advantage abar[g] = mean_r a[r][g]. Over draws of r and g,
E[a^2] = E[abar^2] + E[(a - abar)^2]; the responder share is E[(a - abar)^2] / E[a^2], pooled over requests as a ratio
of sums. Requests are weighted by their number of responders (row sampling, as Rank-GRPO draws them) or equally.
95% CIs: bootstrap over requests (2,000 resamples, seed 42). List-level NDCG stands in for the per-position rewards.
Output: outputs/crs/analysis/adv_variance.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/crs/consensus_check/per_context_{}_qwen2B.jsonl"
OUT = ROOT / "outputs/crs/analysis/adv_variance.json"
EPS = 1e-4
BUCKETS = [(2, 4), (5, 19), (20, 10**9)]


def parts(S: np.ndarray, scaled: bool) -> tuple[float, float]:
    """(responder part, total) of sum over r, g of a[r][g]^2 for one request."""
    a = S - S.mean(axis=1, keepdims=True)
    if scaled:
        sd = S.std(axis=1, keepdims=True)
        a = np.where(sd > 0, a / (sd + EPS), 0.0)
    abar = a.mean(axis=0, keepdims=True)
    return float(((a - abar) ** 2).sum()), float((a**2).sum())


def share(resp: np.ndarray, tot: np.ndarray, w: np.ndarray) -> float:
    # per-request sums are over n_r x G cells; divide by n_r to get a per-draw mean, then weight requests by w
    return float((w * resp).sum() / (w * tot).sum())


def main() -> None:
    rng = np.random.default_rng(42)
    out: dict = {"note": __doc__.split("\n")[0], "datasets": {}}
    for ds in ("reddit", "books", "music"):
        rows = [json.loads(l) for l in open(str(SRC).format(ds))]
        n = np.array([r["n_rows"] for r in rows], dtype=float)
        res: dict = {"requests": len(rows), "mean_responders": round(float(n.mean()), 2)}
        for scaled in (False, True):
            pr = np.array([parts(np.array(r["S"], dtype=float), scaled) for r in rows])
            resp, tot = pr[:, 0] / n, pr[:, 1] / n  # per-draw means
            key = "group_scaled" if scaled else "centred"
            block = {}
            for wname, w in (("row_weighted", n), ("request_weighted", np.ones_like(n))):
                est = share(resp, tot, w)
                boots = []
                for _ in range(2000):
                    i = rng.integers(0, len(rows), len(rows))
                    if (w[i] * tot[i]).sum() > 0:
                        boots.append(share(resp[i], tot[i], w[i]))
                block[wname] = [round(est, 4), round(float(np.percentile(boots, 2.5)), 4),
                                round(float(np.percentile(boots, 97.5)), 4)]
            by = {}
            for lo, hi in BUCKETS:
                m = (n >= lo) & (n <= hi)
                if m.sum() >= 10:
                    by[f"{lo}-{hi if hi < 10**9 else 'inf'}"] = {
                        "requests": int(m.sum()), "row_weighted": round(share(resp[m], tot[m], n[m]), 4)}
            block["by_responders"] = by
            block["zero_total_requests"] = int((tot == 0).sum())
            res[key] = block
        out["datasets"][ds] = res
        print(ds, json.dumps(res))
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
