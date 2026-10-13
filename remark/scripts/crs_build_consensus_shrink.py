"""Stage 2 data: GRPO training set with shrinkage consensus weights p~_i = (c_i + alpha mu_i) / (n + alpha).

Items = the thread's responder union plus the top-100 neighbour-prior items (k nearest training threads, see
crs_neighbor_prior.py); weights = p~. Same row order and columns as remark/data/crs/grpo_consensus/train, so the
consensus reward (lib/crs_rewards.make_consensus_reward) is used unchanged.

Usage (node): /tmp/qenv/bin/python remark/scripts/crs_build_consensus_shrink.py --k 200 --alpha 8
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

REPO = Path(__file__).resolve().parents[2]
PRIOR = REPO / "remark/data/crs/neighbor_prior"
CONS = REPO / "remark/data/crs/grpo_consensus/train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--top_prior", type=int, default=100)
    a = ap.parse_args()
    threads = [json.loads(line)["responders"] for line in open(PRIOR / "threads.jsonl")]
    # same-submission threads excluded (incident)
    nbr_idx, nbr_sim = np.load(PRIOR / "nbr_idx_xsub.npy"), np.load(PRIOR / "nbr_sim_xsub.npy").astype(np.float32)
    cons = load_from_disk(str(CONS))
    assert len(cons) == len(threads)
    counts, freq = [], []
    for resp in threads:
        c = Counter()
        for r in resp:
            c.update(tuple(x) for x in r)
        counts.append(c)
        freq.append({i: v / len(resp) for i, v in c.items()})
    gts_out, w_out, n_items, prior_mass = [], [], [], []
    for t, resp in enumerate(threads):
        n = len(resp)
        assert n == cons[t]["n_responders"]
        mu: Counter = Counter()
        w_sum = 0.0
        for j, s in zip(nbr_idx[t, :a.k], nbr_sim[t, :a.k]):
            w = max(float(s), 0.0)
            w_sum += w
            for item, f in freq[j].items():
                mu[item] += w * f
        mu_top = {i: v / w_sum for i, v in mu.most_common(a.top_prior)} if w_sum > 0 else {}
        c = counts[t]
        items = list(c) + [i for i in mu_top if i not in c]
        if a.alpha == float("inf"):  # neighbour prior alone (ablation): the thread's own responders are ignored
            items = list(mu_top)
            p = [mu_top[i] for i in items]
        else:
            p = [(c.get(i, 0) + a.alpha * mu_top.get(i, 0.0)) / (n + a.alpha) for i in items]
        gts_out.append([[ti, y] for ti, y in items])
        w_out.append(p)
        n_items.append(len(items))
        prior_mass.append(sum(pi for i, pi in zip(items, p) if i not in c) / max(sum(p), 1e-9))
    out = Dataset.from_dict({"prompt": cons["prompt"], "seen_titles": cons["seen_titles"],
                             "groundtruth_with_release_year": gts_out, "groundtruth_weights": w_out,
                             "n_responders": cons["n_responders"]})
    dst = REPO / f"remark/data/crs/grpo_consensus_shrinkx_k{a.k}_a{a.alpha:g}/train"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(dst))
    stats = {"k": a.k, "alpha": a.alpha, "threads": len(out), "mean_items": float(np.mean(n_items)),
             "mean_prior_share_of_weight": float(np.mean(prior_mass))}
    (dst.parent / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
