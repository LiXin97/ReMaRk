"""Reward/sampling ablations of the consensus reward (amendment), built from grpo_consensus/train.

union:        weight 1 for every item any responder named (Mult-DPO-style positives inside Rank-GRPO); threads uniform.
consensus_rw: consensus weights, but one row per (thread, responder) of the Rank-GRPO split, so threads are sampled
              in proportion to n as in Rank-GRPO and the per-row metric; each row carries its thread's consensus.

Usage (node): /tmp/qenv/bin/python remark/scripts/crs_build_consensus_variants.py
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk

REPO = Path(__file__).resolve().parents[2]
CONS = REPO / "remark/data/crs/grpo_consensus/train"
ROWS = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"


def main() -> None:
    cons = load_from_disk(str(CONS))
    union = cons.map(lambda ex: {"groundtruth_weights": [1.0] * len(ex["groundtruth_weights"])}, num_proc=8)
    dst_u = REPO / "remark/data/crs/grpo_union/train"
    union.save_to_disk(str(dst_u))
    index = {p[0]["content"]: i for i, p in enumerate(cons["prompt"])}
    rows = load_from_disk(str(ROWS))
    thread = [index[p[0]["content"]] for p in rows["prompt"]]
    rw = cons.select(thread)
    dst_rw = REPO / "remark/data/crs/grpo_consensus_rw/train"
    rw.flatten_indices().save_to_disk(str(dst_rw))
    stats = {"union_rows": len(union), "consensus_rw_rows": len(rw),
             "rw_mean_n": float(sum(rw["n_responders"]) / len(rw)), "thread_mean_n": float(sum(cons["n_responders"]) / len(cons))}
    for d in (dst_u, dst_rw):
        (d.parent / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
