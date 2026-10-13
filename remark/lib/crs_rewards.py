"""Per-position CRS rewards for Rank-GRPO-style training on Reddit-v2.

single:    Rank-GRPO exp_inf reward (hit at rank position k against one responder's list), from third_party.
consensus: the weight (fraction of the context's responders) of the ground-truth item matched at position k; the
           expectation of the single-responder reward over the context's responders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RG = Path(__file__).resolve().parents[1] / "third_party/Rank-GRPO"
sys.path.insert(0, str(RG))
from libs.metrics_align import _default_title_normalizer  # noqa: E402
from libs.reward_funcs import make_reward_func_individual  # noqa: E402,F401
from libs.utils import process_rec_raw  # noqa: E402


def weighted_hits(recs, gts, weights, seen, catalog, rec_num=20, tol=2):
    """Per-position reward: weight of the first unmatched GT with the same normalized title and |dy| <= tol."""
    out = np.zeros(rec_num)
    remaining = {}
    for (t, y), w in zip(gts, weights):
        try:
            remaining.setdefault(_default_title_normalizer(t), []).append([int(str(y)), float(w)])
        except ValueError:
            continue
    for pos, (rt, ry) in enumerate(recs[:rec_num]):
        if rt in seen or (rt, ry) not in catalog:
            continue
        cands = remaining.get(_default_title_normalizer(rt))
        if not cands:
            continue
        best = min(range(len(cands)), key=lambda k: abs(cands[k][0] - int(ry)))
        if abs(cands[best][0] - int(ry)) <= tol:
            out[pos] = cands[best][1]
            cands.pop(best)
    return out


def make_consensus_reward(catalog, rec_num=20):
    def reward_consensus(completions, groundtruth_with_release_year, groundtruth_weights, seen_titles, **kwargs):
        rewards = []
        for comp, gts, ws, seen in zip(completions, groundtruth_with_release_year, groundtruth_weights, seen_titles):
            item = {"raw": comp[0]["content"]}
            err, item = process_rec_raw(item, "raw", "recs")
            if err:
                rewards.append([0.0] * rec_num)
                continue
            recs = [(t, int(y)) for t, y in item["recs"]]
            rewards.append(weighted_hits(recs, gts, ws, set(seen), catalog, rec_num).tolist())
        return rewards
    return reward_consensus
