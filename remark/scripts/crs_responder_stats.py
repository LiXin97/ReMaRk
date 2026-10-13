"""Responder statistics of the Reddit-v2 GRPO train split (inputs to the consensus-reward theory).

Groups Rank-GRPO's per-(thread, responder) rows by prompt text and reports: responders per thread (histogram,
quantiles, thread- vs row-weighted mean), items per responder, union size, how often a union item is named by
exactly one responder, and how many threads have rows with different seen-title sets (for those threads the
consensus reward, which uses the union of seen titles, is not exactly the responder average of the per-row reward).

Usage: env/bin/python -s remark/scripts/crs_responder_stats.py   (~3 min, ~3 GB RAM)
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path = [p for p in sys.path if ".local" not in p]  # user-site packages shadow the env's datasets deps
import numpy as np  # noqa: E402
from datasets import load_from_disk  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"
OUT = REPO / "remark/outputs/theory/reddit_responder_stats.json"


def main() -> None:
    d = load_from_disk(str(SRC))
    prompts, seen, gts = d["prompt"], d["seen_titles"], d["groundtruth_with_release_year"]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[p[0]["content"]].append(i)
    ns = np.array([len(v) for v in groups.values()])
    seen_differ = sum(1 for v in groups.values() if len({tuple(sorted(seen[i])) for i in v}) > 1)
    items_per_row = np.array([len({(g[0], str(g[1])) for g in x}) for x in gts])
    union, top_frac, frac_count1, weights = [], [], [], []
    for v in groups.values():
        c: Counter = Counter()
        for i in v:
            for key in {(g[0], str(g[1])) for g in gts[i]}:
                c[key] += 1
        union.append(len(c))
        if len(v) >= 2 and c:
            counts = np.array(list(c.values()))
            top_frac.append(counts.max() / len(v))
            frac_count1.append(float((counts == 1).mean()))
            weights.extend((counts / len(v)).tolist())
    weights = np.array(weights)
    hist = Counter(ns.tolist())
    out = {
        "source": str(SRC.relative_to(REPO)),
        "rows": int(len(d)), "threads": int(len(groups)),
        "n_mean_thread_weighted": float(ns.mean()), "n_mean_row_weighted": float((ns ** 2).sum() / ns.sum()),
        "n_median": float(np.median(ns)), "n_max": int(ns.max()),
        "n_quantiles": {str(q): float(np.quantile(ns, q)) for q in (0.5, 0.75, 0.9, 0.99)},
        "frac_threads_n1": float((ns == 1).mean()), "frac_rows_in_n1_threads": float(ns[ns == 1].sum() / ns.sum()),
        "frac_rows_in_threads_n_ge_20": float(ns[ns >= 20].sum() / ns.sum()),
        "frac_threads_n_ge_20": float((ns >= 20).mean()),
        "threads_with_differing_seen_sets": int(seen_differ),
        "items_per_responder_mean": float(items_per_row.mean()),
        "items_per_responder_median": float(np.median(items_per_row)),
        "union_size_mean": float(np.mean(union)),
        "multi_responder_threads": int((ns >= 2).sum()),
        "multi_top_item_fraction_mean": float(np.mean(top_frac)),
        "multi_frac_union_items_named_once_mean": float(np.mean(frac_count1)),
        "multi_consensus_weight_quantiles": {str(q): float(np.quantile(weights, q)) for q in (0.1, 0.5, 0.9)},
        "multi_frac_weights_le_0.1": float((weights <= 0.1).mean()),
        "n_hist": [[int(k), int(hist[k])] for k in sorted(hist)],
        "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "n_hist"}, indent=1))


if __name__ == "__main__":
    main()
