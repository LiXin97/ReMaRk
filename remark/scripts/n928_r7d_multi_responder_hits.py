"""How often a training group (request x 8 SFT lists) hits, at a rank position, an item named by >= 2
responders. CPU only; reads the per-row reward cache written by n928_r7c_batch_sigma.py (built from the offline-DPO 2B SFT
samples, 8 lists per training request, crs_eval parser, crs_eval.hits_for truncate matching per (request, responder) row
of data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train with the row's own seen titles).

Definitions
  Group-position = (request x, position k), k = 1..20; 72,233 requests x 20 positions.
  S(x, l, k) = number of x's responders whose per-responder 0/1 reward is 1 at position k of list l (the sum reward).
  multi(x, k) = some list l of the 8 has S(x, l, k) >= 2 (it hits an item named by >= 2 responders); any(x, k) = some
  list has S >= 1 (a hit at all).
  Cross-check: the same with the union match (hits_for against the request's union items, union seen set) and the item's
  responder count c_i (rows naming the item): multi_c(x, k) = some l whose matched union item has c_i >= 2.
  Reported: share of group-positions with multi (request-weighted, primary), overall and by n_x bucket (1, 2-4, 5-9,
  >= 10), and among group-positions with any hit; the same weighted by n_x (row-weighted, secondary); the share of hit
  list-positions (S >= 1) whose item has S >= 2.
Output: remark/outputs/crs/analysis/n928_r7d_multi_responder_hits.json
Usage:  env/bin/python remark/scripts/n928_r7d_multi_responder_hits.py   (after n928_r7c_batch_sigma.py)
"""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import REPO  # noqa: E402
from n928_r7c_batch_sigma import CACHE  # noqa: E402

OUT = REPO / "remark/outputs/crs/analysis/n928_r7d_multi_responder_hits.json"
NBUCKETS = {"1": (1, 1), "2-4": (2, 4), "5-9": (5, 9), ">=10": (10, 10**9)}


def shares(multi: np.ndarray, anyhit: np.ndarray, w: np.ndarray) -> dict:
    """multi, anyhit: (requests, 20) bool; w: request weights."""
    W = np.repeat(w[:, None], multi.shape[1], axis=1).astype(np.float64)
    return {"group_positions": int(multi.size), "share_multi": float((multi * W).sum() / W.sum()),
            "share_any_hit": float((anyhit * W).sum() / W.sum()),
            "share_multi_among_any_hit": float((multi * W).sum() / max((anyhit * W).sum(), 1e-12))}


def main() -> None:
    z = np.load(CACHE / "rewards_single.npz")
    R1, req = z["R"].astype(np.uint16), z["req"]
    meta = pickle.load(open(CACHE / "union_item_counts.pkl", "rb"))
    n_x = meta["n_x"]
    n_req = len(n_x)
    assert (np.bincount(req, minlength=n_req) == n_x).all()
    order = np.argsort(req, kind="stable")
    S = np.add.reduceat(R1[order], np.r_[0, np.cumsum(n_x)[:-1]], axis=0)  # (requests, 8, 20)
    multi = (S >= 2).any(axis=1)
    anyhit = (S >= 1).any(axis=1)
    M = np.load(CACHE / "union_match.npz")["M"]
    cmat = np.zeros(M.shape, np.int32)
    for x, cnt in enumerate(meta["union_item_counts"]):
        m = M[x]
        if (m >= 0).any():
            cmat[x][m >= 0] = np.array(cnt)[m[m >= 0]]
    multi_c = (cmat >= 2).any(axis=1)
    any_u = (M >= 0).any(axis=1)
    res = {"definition": __doc__.split("Definitions", 1)[1].split("Output:", 1)[0].strip(),
           "n_requests": int(n_req), "positions": int(S.shape[2]), "max_S": int(S.max()),
           "request_weighted": {"all": shares(multi, anyhit, np.ones(n_req))},
           "row_weighted (weights n_x)": {"all": shares(multi, anyhit, n_x)},
           "by_position_request_weighted": {"share_multi": multi.mean(axis=0).round(4).tolist(),
                                            "share_multi_among_any_hit":
                                                (multi.sum(axis=0) / np.maximum(anyhit.sum(axis=0), 1)).round(4).tolist()},
           "hit_level": {"hit_list_positions": int((S >= 1).sum()),
                         "share_of_hits_with_S_ge_2": float((S >= 2).sum() / max((S >= 1).sum(), 1))},
           "cross_check_union_match_item_counts": {
               "share_multi_c": float(multi_c.mean()), "share_any_union_hit": float(any_u.mean()),
               "disagreement_multi_vs_multi_c": float((multi != multi_c).mean()),
               "disagreement_any_vs_any_union": float((anyhit != any_u).mean())}}
    for b, (lo, hi) in NBUCKETS.items():
        sel = (n_x >= lo) & (n_x <= hi)
        res["request_weighted"][b] = {"requests": int(sel.sum()), **shares(multi[sel], anyhit[sel], np.ones(sel.sum()))}
        res["row_weighted (weights n_x)"][b] = {"requests": int(sel.sum()), **shares(multi[sel], anyhit[sel], n_x[sel])}
    try:
        res["git"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()
    except OSError:
        pass
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
