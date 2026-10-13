"""Build the consensus GRPO training set from Rank-GRPO's Reddit-v2 grpo_dataset.

One row per unique conversation context. Ground truth = union of all responders' recommendations;
each item's weight = fraction of the context's responders who recommended it. The consensus reward
(sum of matched weights, per rank position) equals the expectation of Rank-GRPO's single-responder
reward over responders, so the objective is unchanged and only responder-sampling noise is removed.

Usage (node): python3 remark/scripts/crs_build_consensus.py
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path

from datasets import Dataset, load_from_disk

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"
OUT = REPO / "remark/data/crs/grpo_consensus/train"


def main() -> None:
    d = load_from_disk(str(SRC))
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    prompts = d["prompt"]
    for i, p in enumerate(prompts):
        groups.setdefault(p[0]["content"], []).append(i)
    gts_all, seen_all = d["groundtruth_with_release_year"], d["seen_titles"]
    rows = {"prompt": [], "seen_titles": [], "groundtruth_with_release_year": [], "groundtruth_weights": [], "n_responders": []}
    for ctx, idx in groups.items():
        cnt: "OrderedDict[tuple, int]" = OrderedDict()
        seen = set()
        for i in idx:
            seen |= set(seen_all[i])
            for t, y in {(g[0], str(g[1])) for g in gts_all[i]}:  # count each responder once per item
                cnt[(t, y)] = cnt.get((t, y), 0) + 1
        n = len(idx)
        rows["prompt"].append(prompts[idx[0]])
        rows["seen_titles"].append(sorted(seen))
        rows["groundtruth_with_release_year"].append([[t, y] for (t, y) in cnt])
        rows["groundtruth_weights"].append([c / n for c in cnt.values()])
        rows["n_responders"].append(n)
    out = Dataset.from_dict(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(OUT))
    stats = {"rows_in": len(d), "contexts_out": len(out),
             "mean_responders": sum(rows["n_responders"]) / len(out),
             "mean_union_size": sum(len(g) for g in rows["groundtruth_with_release_year"]) / len(out)}
    (OUT.parent / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
