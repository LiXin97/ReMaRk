"""ProtoQA datasets in the CRS on-disk format (load_from_disk dirs) under remark/data/crs/protoqa/.

  grpo/train         one row per (question, respondent): the 8,581 training questions' cluster counts expanded into
                     765,701 respondents, each row's target = its respondent's cluster. TRL's RepeatSampler draws rows
                     with torch.randperm (uniform over rows), so questions are drawn in proportion to their respondent
                     count. Serves arms (a) single, (b) rb, (c) mean.
  grpo_union/train   one row per training question (all clusters as targets): questions drawn uniformly. Arm (d) union.
  sft_dataset/train  training questions with a 10-answer teacher list; completion = the teacher list (one answer per line)
  sft_dataset/validation   the 400 validation questions, completion = their teacher list (crs_train_sft.py's eval loss);
                     clusters stored for evaluation / checkpoint selection
  sft_dataset/test   test_scraped (558), clusters only;  sft_dataset/test_crowdsourced (52), clusters only
Column encoding: lib/r9_common.py docstring. Splits and cleaning: remark/data/protoqa/splits (n928_r9_stats.py, C1-C5).

Usage (CPU host, repo env): env/bin/python remark/scripts/n928_r9_build.py --part grpo|sft|all
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

from datasets import Dataset, Features, List, Value

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.r9_common import CRS_DATA, OUT, REC_NUM, REPO, load_split, prompt_messages  # noqa: E402

MSG = List({"content": Value("string"), "role": Value("string")})
BASE = {"prompt": MSG, "seen_titles": List(Value("string")), "qid": Value("string"), "question": Value("string"),
        "all_clusters": List(List(Value("string"))), "cluster_counts": List(Value("int32")),
        "n_respondents": Value("int32"), "resp_cluster": Value("int32"),
        "groundtruth_with_release_year": List(List(Value("string")))}
TEACHER = OUT / "teacher/teacher_lists.jsonl"


def row(q: dict, resp_cluster: int = -1) -> dict:
    p = prompt_messages(q["question"])
    return {"prompt": [{"content": p[0]["content"], "role": "user"}], "seen_titles": [], "qid": q["qid"],
            "question": q["question"], "all_clusters": q["clusters"], "cluster_counts": q["counts"],
            "n_respondents": q["n"], "resp_cluster": resp_cluster,
            "groundtruth_with_release_year": [q["clusters"][resp_cluster]] if resp_cluster >= 0 else q["clusters"]}


def save(rows: list[dict], path: Path, features: Features) -> int:
    tmp = path.with_name(path.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    Dataset.from_list(rows, features=features).save_to_disk(str(tmp))
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)
    return len(rows)


def build_grpo() -> dict:
    train = load_split("train")
    rows = [row(q, c) for q in train for c, n in enumerate(q["counts"]) for _ in range(n)]
    n_rows = save(rows, CRS_DATA / "grpo/train", Features(BASE))
    per_q = Counter(r["qid"] for r in rows)
    assert all(per_q[q["qid"]] == q["n"] for q in train)
    del rows
    n_union = save([row(q) for q in train], CRS_DATA / "grpo_union/train", Features(BASE))
    total = sum(q["n"] for q in train)
    ratio = sorted(q["n"] / total * len(train) for q in train)  # P(question) under row sampling / uniform
    pct = lambda p: round(ratio[min(len(ratio) - 1, int(p * len(ratio)))], 3)  # noqa: E731
    return {"grpo/train": {"rows": n_rows, "questions": len(train), "rows_per_question_eq_n_respondents": True,
                           "question_prob_over_uniform": {"min": round(ratio[0], 3), "p5": pct(0.05),
                                                          "median": pct(0.5), "p95": pct(0.95),
                                                          "max": round(ratio[-1], 3)},
                           "sampler": "TRL RepeatSampler: torch.randperm over rows (trl/trainer/utils.py, TRL 1.14) -> "
                                      "uniform over rows -> questions in proportion to respondent count"},
            "grpo_union/train": {"rows": n_union, "questions": len(train), "sampler": "uniform over questions"}}


def build_sft() -> dict:
    teacher = {}
    for line in TEACHER.open():
        t = json.loads(line)
        teacher[t["qid"]] = t["answers"]
    feats = Features({**BASE, "completion": MSG})
    # (fixed 11:46 before any GPU run): lists that repeat an answer under the matcher are left out of the SFT data
    dup_path = REPO / "remark/data/protoqa/teacher_dup_qids.json"
    dup = set(json.loads(dup_path.read_text())["qids"]) if dup_path.exists() else set()
    out, dropped = {}, {}
    for split, src in (("train", "train"), ("validation", "validation")):
        rows, drop = [], Counter()
        for q in load_split(src):
            a = teacher.get(q["qid"])
            if not a:
                drop["no_list"] += 1
                continue
            if split == "train" and len(a) != REC_NUM:  # SFT targets teach exactly 10 lines
                drop[f"{len(a)}_answers"] += 1
                continue
            if split == "train" and q["qid"] in dup:
                drop["matcher_duplicates"] += 1
                continue
            r = row(q)
            r["completion"] = [{"content": "\n".join(a), "role": "assistant"}]
            rows.append(r)
        out[split] = save(rows, CRS_DATA / f"sft_dataset/{split}", feats)
        dropped[split] = dict(drop)
    for split, src in (("test", "test_scraped"), ("test_crowdsourced", "test_crowdsourced")):
        out[split] = save([row(q) for q in load_split(src)], CRS_DATA / f"sft_dataset/{split}", Features(BASE))
    lens = [len("\n".join(a)) for a in teacher.values() if len(a) == REC_NUM]
    return {"sft_dataset": {"rows": out, "dropped": dropped,
                            "train_rule": "training questions whose teacher list has exactly 10 answers, none repeated under the matcher "
                                          "(teacher_dup_qids.json); validation keeps all 400 questions (evaluation split)",
                            "completion_chars_median": statistics.median(lens) if lens else None,
                            "completion_chars_max": max(lens) if lens else None}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["grpo", "sft", "all"], default="all")
    a = ap.parse_args()
    CRS_DATA.mkdir(parents=True, exist_ok=True)
    man_path = CRS_DATA / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    if a.part in ("grpo", "all"):
        man.update(build_grpo())
    if a.part in ("sft", "all"):
        man.update(build_sft())
    man["script"] = "remark/scripts/n928_r9_build.py"
    man["git"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    man["inputs"] = {"splits": "remark/data/protoqa/splits (manifest.json there)",
                     "teacher": str(TEACHER.relative_to(REPO))}
    man["encoding"] = "lib/r9_common.py docstring"
    man_path.write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
