"""Row-weighted (threads sampled in proportion to n) copy of a thread-level Reddit-v2 GRPO dataset.

Each row of the Rank-GRPO per-responder split is replaced by its thread's row of the given thread-level dataset (same
thread order as grpo_consensus/train), so a thread appears n times, as in Rank-GRPO and the per-row test metric.

Usage (node): /tmp/qenv/bin/python remark/scripts/crs_build_rw.py --src remark/data/crs/grpo_consensus_shrinkx_k200_a8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk

REPO = Path(__file__).resolve().parents[2]
CONS = REPO / "remark/data/crs/grpo_consensus/train"
ROWS = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="thread-level dataset dir (contains train/)")
    a = ap.parse_args()
    src = Path(a.src) if Path(a.src).is_absolute() else REPO / a.src
    thread_ds = load_from_disk(str(src / "train"))
    index = {p[0]["content"]: i for i, p in enumerate(load_from_disk(str(CONS))["prompt"])}
    assert len(index) == len(thread_ds)
    rows = load_from_disk(str(ROWS))
    rw = thread_ds.select([index[p[0]["content"]] for p in rows["prompt"]])
    dst = src.parent / f"{src.name}_rw" / "train"
    rw.flatten_indices().save_to_disk(str(dst))
    stats = {"src": str(src), "rows": len(rw), "row_weighted_mean_n": float(sum(rw["n_responders"]) / len(rw))}
    (dst.parent / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
