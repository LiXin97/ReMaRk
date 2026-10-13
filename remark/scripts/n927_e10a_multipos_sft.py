"""Build the multi-positive SFT dataset (one example per request, targets from the pooled positives).

Rank-GRPO's SFT dataset has one example per (request, responder) row, whose target is a 20-title list containing that
responder's items. The multi-positive baseline keeps everything else fixed and changes only the target: one example per
request, whose list leads with the union of all the responders' items, ordered by the number of responders who named
each item (ties by the item's training popularity, then by title), and is filled to 20 titles from the request's
existing row targets (in row order, skipping duplicates and the seen titles). Validation and test splits are copied
unchanged, so the selection and evaluation protocol is the one every other arm uses.

Responder counts come from remark/data/crs/grpo_consensus (groundtruth_weights = share of responders, n_responders),
the same source the consensus reward uses; popularity is the number of training rows whose target list contains the
item. A request contributes at most 20 pooled positives; the share of requests whose union exceeds 20 is reported.

Usage: env/bin/python remark/scripts/n927_e10a_multipos_sft.py
Output: remark/data/crs/sft_multipos/{train,validation,test} + manifest.json
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk

REPO = Path(__file__).resolve().parents[2]
SFT = REPO / "data/raw/reddit_v2/processed_datasets/sft_dataset"
CONS = REPO / "remark/data/crs/grpo_consensus"
OUT = REPO / "remark/data/crs/sft_multipos"
LINE = re.compile(r"^(.*?)\s*\((\d{4})\)\s*$")


def key(title: str, year: str) -> str:
    return f"{title.strip().casefold()}|{year}"


def main() -> None:
    cons = load_from_disk(str(CONS / "train"))
    # request -> [(item, n_responders_who_named_it)], keyed by the prompt text (identical in both datasets)
    counts: dict[str, list[tuple[tuple[str, str], float]]] = {}
    for e in cons:
        p = e["prompt"][0]["content"]
        n = max(int(e["n_responders"]), 1)
        counts[p] = [((t, y), w * n) for (t, y), w in zip(e["groundtruth_with_release_year"], e["groundtruth_weights"])]

    train = load_from_disk(str(SFT / "train"))
    pop: Counter[str] = Counter()
    rows_by_prompt: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(train):
        rows_by_prompt[e["prompt"][0]["content"]].append(i)
        for line in e["completion"][0]["content"].splitlines():
            m = LINE.match(line.strip())
            if m:
                pop[key(m.group(1), m.group(2))] += 1

    out_rows, over20, missing = [], 0, 0
    for prompt, idxs in rows_by_prompt.items():
        pos = counts.get(prompt)
        if pos is None:
            missing += 1
            continue
        pos = sorted(pos, key=lambda iw: (-iw[1], -pop[key(*iw[0])], iw[0][0].casefold()))
        if len(pos) > 20:
            over20 += 1
        seen_keys = {key(t, "") .split("|")[0] for t in train[idxs[0]]["seen_titles"]}
        lines, used = [], set()
        for (t, y), _ in pos:
            k = key(t, y)
            if k in used or t.strip().casefold() in seen_keys:
                continue
            used.add(k)
            lines.append(f"{t} ({y})")
            if len(lines) == 20:
                break
        for i in idxs:  # fill from the request's own row targets, in row order
            if len(lines) == 20:
                break
            for line in train[i]["completion"][0]["content"].splitlines():
                m = LINE.match(line.strip())
                if not m:
                    continue
                k = key(m.group(1), m.group(2))
                if k in used or m.group(1).strip().casefold() in seen_keys:
                    continue
                used.add(k)
                lines.append(f"{m.group(1)} ({m.group(2)})")
                if len(lines) == 20:
                    break
        e0 = train[idxs[0]]
        out_rows.append({"prompt": e0["prompt"], "completion": [{"role": "assistant", "content": "\n".join(lines)}],
                         "seen_titles": e0["seen_titles"],
                         "groundtruth_with_release_year": [list(i) for i, _ in pos]})

    from datasets import Dataset
    OUT.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(out_rows)
    ds.save_to_disk(str(OUT / "train"))
    lens = Counter(len(r["completion"][0]["content"].splitlines()) for r in out_rows)
    for split in ("validation", "test"):
        load_from_disk(str(SFT / split)).save_to_disk(str(OUT / split))
    man = {
        "built_from": {"sft": str(SFT.relative_to(REPO)), "responder_counts": str(CONS.relative_to(REPO))},
        "train_rows_in": len(train), "train_requests_out": len(out_rows),
        "requests_without_counts": missing, "requests_with_more_than_20_pooled_positives": over20,
        "target_length_histogram": dict(sorted(lens.items())),
        "ordering": "pooled positives by responder count desc, then training popularity desc, then title; "
                    "filled to 20 from the request's row targets in row order; seen titles excluded",
        "validation_test": "copied unchanged from the row-level SFT dataset",
        "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    }
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
