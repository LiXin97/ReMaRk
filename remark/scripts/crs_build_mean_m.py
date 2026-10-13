"""Dose-response data: per Rank-GRPO row, the reward is the mean over a responder subset of size m that
contains the row's own responder and m - 1 other responders of the same request drawn uniformly at random (all
responders if n <= m); the item weight is the share of the subset that named the item. Rows keep Rank-GRPO's order
(requests drawn in proportion to n), so runs with the same seed see the same request sequence as the per-responder and
consensusrw runs. Also writes rho(m), the residual responder-noise share on the training rows:
mean over rows of sum_i (w_M(i) - p(i))^2 divided by the same quantity for m = 1 (p = full-request share).

Usage (node): /tmp/qenv/bin/python remark/scripts/crs_build_mean_m.py --m 2 4
Output: remark/data/crs/grpo_mean_m<m>_rw/train (+ stats.json)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

REPO = Path(__file__).resolve().parents[2]
ROWS = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"
CONS = REPO / "remark/data/crs/grpo_consensus/train"
THREADS = REPO / "remark/data/crs/neighbor_prior/threads.jsonl"


def key(items) -> frozenset:
    return frozenset((str(t), str(y)) for t, y in items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    threads = [[key(r) for r in json.loads(line)["responders"]] for line in open(THREADS)]
    cons_prompts = load_from_disk(str(CONS))["prompt"]
    assert len(cons_prompts) == len(threads)
    index = {p[0]["content"]: i for i, p in enumerate(cons_prompts)}
    rows = load_from_disk(str(ROWS))
    prompts, seen, gts = rows["prompt"], rows["seen_titles"], rows["groundtruth_with_release_year"]
    thread_of = np.array([index[p[0]["content"]] for p in prompts])
    # own responder of each row: the thread's responder with the same item set (else the best Jaccard match)
    own, exact = [], 0
    for j, t in enumerate(thread_of):
        g = key(gts[j])
        resp = threads[t]
        hit = [k for k, r in enumerate(resp) if r == g]
        if hit:
            own.append(hit[0]); exact += 1
        else:
            jac = [len(g & r) / max(len(g | r), 1) for r in resp]
            own.append(int(np.argmax(jac)))
    full = []
    for resp in threads:
        c = Counter(i for r in resp for i in r)
        full.append({i: v / len(resp) for i, v in c.items()})
    stats = {"rows": len(rows), "own_responder_exact_match": exact / len(rows), "seed": a.seed}
    noise = {}
    for m in [1] + a.m:
        rng = np.random.default_rng(a.seed)
        items_out, w_out, sq = [], [], []
        for j, t in enumerate(thread_of):
            resp = threads[t]
            n = len(resp)
            if m >= n:
                sub = list(range(n))
            else:
                others = [k for k in range(n) if k != own[j]]
                sub = [own[j]] + list(rng.choice(others, size=m - 1, replace=False))
            c = Counter(i for k in sub for i in resp[k])
            w = {i: v / len(sub) for i, v in c.items()}
            p = full[t]
            sq.append(sum((w.get(i, 0.0) - p[i]) ** 2 for i in p))
            if m != 1:
                items_out.append([[ti, y] for ti, y in w])
                w_out.append([w[i] for i in w])
        noise[m] = float(np.mean(sq))
        if m == 1:
            continue
        ds = Dataset.from_dict({"prompt": prompts, "seen_titles": seen, "groundtruth_with_release_year": items_out,
                                "groundtruth_weights": w_out, "n_responders": [len(threads[t]) for t in thread_of]})
        dst = REPO / f"remark/data/crs/grpo_mean_m{m}_rw/train"
        dst.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(dst))
        print(f"m={m}: saved {len(ds)} rows to {dst}")
    stats["residual_noise_sq"] = noise
    stats["rho"] = {m: noise[m] / noise[1] for m in noise}
    for m in a.m:
        (REPO / f"remark/data/crs/grpo_mean_m{m}_rw/stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
