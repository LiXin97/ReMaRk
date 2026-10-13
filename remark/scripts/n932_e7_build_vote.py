"""Vote-weighted binary reward data. From grpo_consensus/train (consensus weight w_i = c_i / n per item),
set groundtruth_weights to 1 + 0.25 * 1{c_i >= 2} with c_i = round(w_i * n_responders); rows, prompts and items unchanged
(ReMaRk's union data has the same rows with weight 1). Usage: env/bin/python remark/scripts/n932_e7_build_vote.py
Output: remark/data/crs/grpo_vote025/train (+ manifest.json)
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from datasets import load_from_disk

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "remark/data/crs/grpo_consensus/train"
DST = REPO / "remark/data/crs/grpo_vote025"
BONUS = 0.25


def main() -> None:
    cons = load_from_disk(str(SRC))

    def vote(ex):
        c = [round(w * ex["n_responders"]) for w in ex["groundtruth_weights"]]
        return {"groundtruth_weights": [1.0 + BONUS * (k >= 2) for k in c]}

    out = cons.map(vote, num_proc=4)
    out.save_to_disk(str(DST / "train"))
    ws = [w for row in out["groundtruth_weights"] for w in row]
    man = {"src": str(SRC.relative_to(REPO)), "rule": "1 + 0.25*1{c>=2}, c = round(w*n)", "rows": len(out),
           "items": len(ws), "share_bonus": sum(w > 1 for w in ws) / len(ws),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()}
    (DST / "manifest.json").write_text(json.dumps(man, indent=2))
    print(man)


if __name__ == "__main__":
    main()
