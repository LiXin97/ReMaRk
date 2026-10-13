"""Questions whose GPT-5.6 teacher list repeats an answer under the ProtoQA matcher
(a pair i > j with matches(a_i, [a_j]), the definition of n928_r9_teacher.py's duplicate count) are left out of the SFT
data. Writes remark/data/protoqa/teacher_dup_qids.json, read by n928_r9_build.py --part sft.
Needs nltk (run with the ProtoQA scratch venv python, see n928_r9_matcher_test.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.r9_common import OUT, REPO  # noqa: E402
from lib.r9_rewards import matches  # noqa: E402


def main() -> None:
    dup = []
    n = 0
    for line in (OUT / "teacher/teacher_lists.jsonl").open():
        t = json.loads(line)
        a = t["answers"]
        n += 1
        if any(matches(a[i], [a[j]]) for i in range(len(a)) for j in range(i)):
            dup.append(t["qid"])
    out = REPO / "remark/data/protoqa/teacher_dup_qids.json"
    out.write_text(json.dumps({"lists": n, "with_duplicates": len(dup), "qids": sorted(dup),
                               "rule": "pair i > j with matches(a_i, [a_j]) under the ProtoQA matcher"}, indent=1))
    print(n, len(dup))


if __name__ == "__main__":
    main()
