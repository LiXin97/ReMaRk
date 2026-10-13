"""Stronger offline DPO pairs at 2B (every training request yields a pair).

chosen   = the request's pooled positives (the union of its responders' items) ordered by the number of responders who
           named each item, ties by training popularity, then title (the multi-positive ordering, n927_e10a_multipos_sft.py),
           seen titles and duplicates removed, truncated to 20; if fewer than 20, padded with items from the request's 8
           SFT-sampled lists (the offline-DPO 2B samples), ordered by the number of those lists containing the item, then by
           first appearance (list order, then position); seen titles and duplicates removed.
rejected = the lowest-scoring of the 8 sampled lists under the union reward (first in sampling order on ties).
Training labels only (the union is built from the training split). Output format as n927_e10b_pairs.py.

Usage: env/bin/python remark/scripts/n928_r6_pairs.py
Output: remark/data/crs/r6_dpo_strong/{train, manifest.json}
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from n927_e10b_sample import parse  # noqa: E402  (the parser used to score the samples)

SFT = REPO / "data/raw/reddit_v2/processed_datasets/sft_dataset"
CONS = REPO / "remark/data/crs/grpo_consensus/train"
UNION = REPO / "remark/data/crs/grpo_union/train"
SAMPLES = REPO / "remark/outputs/crs/e10b/2B/samples"
OUT = REPO / "remark/data/crs/r6_dpo_strong"
LINE = re.compile(r"^(.*?)\s*\((\d{4})\)\s*$")


def key(title: str, year) -> str:
    return f"{str(title).strip().casefold()}|{year}"


def main() -> None:
    counts = {}
    for e in load_from_disk(str(CONS)):
        n = max(int(e["n_responders"]), 1)
        counts[e["prompt"][0]["content"]] = [((t, y), w * n) for (t, y), w in
                                             zip(e["groundtruth_with_release_year"], e["groundtruth_weights"])]
    pop: Counter[str] = Counter()
    for e in load_from_disk(str(SFT / "train")):
        for line in e["completion"][0]["content"].splitlines():
            m = LINE.match(line.strip())
            if m:
                pop[key(m.group(1), m.group(2))] += 1
    union = load_from_disk(str(UNION))
    recs = {}
    for f in sorted(SAMPLES.glob("shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[r["idx"]] = r
    rows, st, n_union, n_pad = [], Counter(), Counter(), Counter()
    for i in sorted(recs):
        ex = union[i]
        prompt = ex["prompt"][0]["content"]
        pos = counts.get(prompt)
        if pos is None:
            st["missing_counts"] += 1
            continue
        seen = {t.strip().casefold() for t in ex["seen_titles"]}
        pos = sorted(pos, key=lambda iw: (-iw[1], -pop[key(*iw[0])], str(iw[0][0]).casefold()))
        lines, used = [], set()
        for (t, y), _ in pos:
            k = key(t, y)
            if k in used or str(t).strip().casefold() in seen:
                continue
            used.add(k); lines.append(f"{t} ({y})")
            if len(lines) == 20:
                break
        n_union[min(len(lines), 20)] += 1
        texts, scores = recs[i]["texts"], recs[i]["scores"]
        if len(lines) < 20:
            first, lists_with = {}, Counter()  # first appearance; number of lists containing the item
            for li, t in enumerate(texts):
                items = parse(t)
                for pi, (tt, yy) in enumerate(items):
                    first.setdefault(key(tt, yy), (li, pi, tt, yy))
                for k in {key(tt, yy) for tt, yy in items}:
                    lists_with[k] += 1
            for k in sorted(first, key=lambda k: (-lists_with[k], first[k][0], first[k][1])):
                _, _, tt, yy = first[k]
                if k in used or str(tt).strip().casefold() in seen:
                    continue
                used.add(k); lines.append(f"{tt} ({yy})")
                if len(lines) == 20:
                    break
        n_pad[len(lines)] += 1
        lo = min(scores); r = scores.index(lo)
        st["chosen_len_20"] += len(lines) == 20
        rows.append({"prompt": [{"role": "user", "content": prompt}],
                     "chosen": [{"role": "assistant", "content": "\n".join(lines)}],
                     "rejected": [{"role": "assistant", "content": texts[r]}]})
        st["rejected_score_" + str(lo)] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).shuffle(seed=3407).save_to_disk(str(OUT / "train"))
    man = {"requests_sampled": len(recs), "requests_in_pool": len(union), "pairs": len(rows),
           "chosen_len_hist": dict(sorted(n_pad.items())), "union_items_used_hist": dict(sorted(n_union.items())),
           "stats": dict(st),
           "rule": "chosen = union by responder count (ties: training popularity, title), padded to 20 with the request's "
                   "SFT-sampled items (by number of lists containing them, then first appearance); rejected = lowest "
                   "union-reward score of the 8 SFT samples (first on ties); every request yields a pair",
           "inputs": {"samples": str(SAMPLES.relative_to(REPO)), "consensus": str(CONS.relative_to(REPO))},
           "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps({k: v for k, v in man.items() if k != "rule"}, indent=2))


if __name__ == "__main__":
    main()
