"""Union and row-weighted consensus variants for any dataset in our Rank-GRPO format (e.g. Reddit-Books).

<root>/grpo_consensus/train (thread level: groundtruth_with_release_year, groundtruth_weights, n_responders) →
  <root>/grpo_union/train          weights set to 1 (bounded sum), threads uniform
  <root>/grpo_consensus_rw/train   one row per (thread, responder) of <root>/grpo/train carrying its thread's consensus
Usage: python remark/scripts/crs_build_variants_generic.py --root remark/data/crs/reddit_books
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    cons = load_from_disk(str(root / "grpo_consensus/train"))
    union = cons.map(lambda ex: {"groundtruth_weights": [1.0] * len(ex["groundtruth_weights"])})
    union.save_to_disk(str(root / "grpo_union/train"))
    index = {p[0]["content"]: i for i, p in enumerate(cons["prompt"])}
    rows = load_from_disk(str(root / "grpo/train"))
    missing = sum(1 for p in rows["prompt"] if p[0]["content"] not in index)
    assert missing == 0, f"{missing} per-responder rows have no thread in grpo_consensus"
    rw = cons.select([index[p[0]["content"]] for p in rows["prompt"]]).flatten_indices()
    rw.save_to_disk(str(root / "grpo_consensus_rw/train"))
    stats = {"threads": len(cons), "rw_rows": len(rw), "thread_mean_n": sum(cons["n_responders"]) / len(cons),
             "row_weighted_mean_n": sum(rw["n_responders"]) / len(rw)}
    (root / "grpo_consensus_rw" / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
