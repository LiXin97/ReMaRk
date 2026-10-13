"""Turn the scored samples into DPO pairs.

Per request: chosen = the highest-scoring of its G lists, rejected = the lowest, the first in sampling order on ties;
a request whose G lists all score the same is dropped. Writes a conversational DPO dataset
({"prompt": [user turn], "chosen": [assistant turn], "rejected": [assistant turn]}) and a manifest with the drop share
and the score distribution.

Usage: env/bin/python remark/scripts/n927_e10b_pairs.py --samples <dir with shard*.jsonl> --out <dataset dir>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_from_disk

REPO = Path(__file__).resolve().parents[2]
UNION = REPO / "remark/data/crs/grpo_union/train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ds = load_from_disk(str(UNION))
    recs = {}
    for f in sorted(Path(a.samples).glob("shard*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[r["idx"]] = r
    rows, dropped, gap, best, worst = [], 0, Counter(), Counter(), Counter()
    for i in sorted(recs):
        sc = recs[i]["scores"]
        hi, lo = max(sc), min(sc)
        if hi == lo:
            dropped += 1
            continue
        c, r = sc.index(hi), sc.index(lo)
        gap[hi - lo] += 1; best[hi] += 1; worst[lo] += 1
        prompt = ds[i]["prompt"][0]["content"]
        rows.append({"prompt": [{"role": "user", "content": prompt}],
                     "chosen": [{"role": "assistant", "content": recs[i]["texts"][c]}],
                     "rejected": [{"role": "assistant", "content": recs[i]["texts"][r]}]})
    out = Path(a.out)
    Dataset.from_list(rows).shuffle(seed=3407).save_to_disk(str(out / "train"))
    man = {"requests_sampled": len(recs), "requests_in_pool": len(ds), "pairs": len(rows),
           "dropped_all_tie": dropped, "dropped_share": round(dropped / max(1, len(recs)), 4),
           "chosen_score_hist": dict(sorted(best.items())), "rejected_score_hist": dict(sorted(worst.items())),
           "score_gap_hist": dict(sorted(gap.items())),
           "rule": "chosen = max union-reward score, rejected = min, first in sampling order on ties; all-tie requests dropped"}
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))


if __name__ == "__main__":
    main()
