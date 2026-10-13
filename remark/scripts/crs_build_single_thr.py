"""Single-responder reward with thread-uniform sampling — one row per Reddit-v2 thread, the responder chosen at
random (seed 42), same thread order as grpo_consensus/train. Packaged as dataset `redditthr` so the unchanged launcher
(DATASET=redditthr, reward single) and slot runner can train and validate it on Reddit-v2.

Usage (node): /tmp/qenv/bin/python remark/scripts/crs_build_single_thr.py
"""

from __future__ import annotations

import json
import random
import shutil
from collections import OrderedDict
from pathlib import Path

from datasets import load_from_disk

REPO = Path(__file__).resolve().parents[2]
ROWS = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"
OUT = REPO / "remark/data/crs/redditthr"


def main() -> None:
    rows = load_from_disk(str(ROWS))
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, p in enumerate(rows["prompt"]):
        groups.setdefault(p[0]["content"], []).append(i)
    rng = random.Random(42)
    pick = [rng.choice(idx) for idx in groups.values()]
    thr = rows.select(pick).flatten_indices()
    (OUT / "grpo").mkdir(parents=True, exist_ok=True)
    thr.save_to_disk(str(OUT / "grpo/train"))
    sft = OUT / "sft_dataset"
    if not sft.exists():
        sft.symlink_to(REPO / "data/raw/reddit_v2/processed_datasets/sft_dataset")
    shutil.copy(REPO / "remark/third_party/Rank-GRPO/reddit_catalog/gt_catalog.pkl", OUT / "catalog.pkl")
    stats = {"threads": len(thr), "rows_in": len(rows), "seed": 42}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
