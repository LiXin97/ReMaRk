"""ProtoQA: shared prompt, answer parser, split loader and NLTK setup.

One parser (`parse_answer`) is used everywhere: teacher lists, SFT targets, training rewards, the trainer's item test
(`lib/r9_trainer.py`), and evaluation, so an SFT target is a fixed point of the parser. Line splitting replicates
`lib.rank_grpo_trl.RankGRPOTrainer._blank` (a blank line or a bare think tag takes no rank position; checked in
n928_r9_unit.py).

Dataset encoding (remark/data/crs/protoqa, built by n928_r9_build.py; columns shared by every split):
  prompt                         [{"role": "user", "content": HEADER + "Question: <question>"}]
  seen_titles                    [] (the trainer passes it through; ProtoQA has no seen items)
  qid                            ProtoQA question id
  question                       normalised question text
  all_clusters                   list[list[str]]: every answer cluster of the question, each a list of alternative strings
                                 (board order; C5 adds the "/"-separated parts of a scraped string)
  cluster_counts                 list[int]: respondents per cluster, aligned with all_clusters
  n_respondents                  int: sum of cluster_counts
  resp_cluster                   int: per-row GRPO split: index (into all_clusters) of the row's respondent's cluster;
                                 -1 in per-question splits
  groundtruth_with_release_year  list[list[str]]: the row's target clusters in the same encoding as all_clusters (no
                                 years): per-row GRPO split [all_clusters[resp_cluster]]; elsewhere all_clusters
  completion                     (sft_dataset train / validation only) [{"role": "assistant", "content": teacher list}]
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROTOQA = REPO / "remark/data/protoqa"
SPLITS = PROTOQA / "splits"
CRS_DATA = REPO / "remark/data/crs/protoqa"
OUT = REPO / "remark/outputs/protoqa"
STOPWORDS_FILE = PROTOQA / "nltk_stopwords_english.txt"  # the pinned 198-word NLTK English list
NLTK_ZIPS = PROTOQA / "nltk_data"  # corpora/{stopwords,wordnet,omw-1.4}.zip, tokenizers/{punkt,punkt_tab}.zip

REC_NUM = 10
HEADER = ("We asked 100 people the question below. List the 10 answers they gave most often, the most common answer "
          "first. Reply with exactly 10 answers, one short answer per line, with NO numbering and NO extra words or "
          "sentences.\n")
THINK_TAGS = ("<think>", "</think>")


def prompt_text(question: str) -> str:
    return HEADER + "Question: " + question


def prompt_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": prompt_text(question)}]


def blank(line: str) -> bool:
    """RankGRPOTrainer._blank: a blank line or one holding only a think tag takes no rank position."""
    t = line.strip()
    return not t or t in THINK_TAGS


def nonblank_lines(text: str, rec_num: int | None = REC_NUM) -> list[str]:
    lines = [x for x in (text or "").split("\n") if not blank(x)]
    return lines[:rec_num] if rec_num else lines


_MARKER = re.compile(r"^(?:[-*•·–—>]+\s+|\(?\d{1,2}[.)\]:]\s+|\(?\d{1,2}[.)\]:]$)")
_EMPH = re.compile(r"(\*\*|__|`|\*)")
_QUOTES = "\"'“”‘’"


def parse_answer(line: str) -> str:
    """One answer from one line: leading list markers ("1.", "2)", "-", "*", bullets) and markdown emphasis removed,
    surrounding quotes and trailing .,;:! stripped, whitespace collapsed. "" when nothing is left (not an answer).
    Numbers that are part of an answer stay ("24 hours", "7-eleven", "9 to 5")."""
    s = (line or "").strip()
    for _ in range(10):  # "1. - **dog**", "**1. Dog**": repeated until a fixed point, so parse(parse(x)) == parse(x)
        t = _MARKER.sub("", s, count=1).strip()
        t = _EMPH.sub("", t).strip()
        t = t.strip(_QUOTES).strip()
        t = re.sub(r"[.,;:!]+$", "", t).strip()
        t = re.sub(r"\s+", " ", t)
        if t == s:
            break
        s = t
    return s if re.search(r"[^\W_]", s) else ""


def answers_from_text(text: str, rec_num: int | None = REC_NUM) -> list[str]:
    """Valid answers, in order, among the first `rec_num` non-blank lines (the trainer's rank positions)."""
    return [a for a in (parse_answer(x) for x in nonblank_lines(text, rec_num)) if a]


def line_answers(text: str, rec_num: int = REC_NUM) -> list[str]:
    """Per rank position (the first rec_num non-blank lines): the parsed answer or "" (a malformed line)."""
    return [parse_answer(x) for x in nonblank_lines(text, rec_num)]


def load_split(name: str) -> list[dict]:
    """splits/<name>.jsonl as rows {qid, question, clusters: list[list[str]], counts: list[int], n}."""
    rows = []
    for line in (SPLITS / f"{name}.jsonl").open():
        r = json.loads(line)
        cl = list(r["clusters"].values())
        clusters = [list(c["answers"]) for c in cl]
        counts = [int(c["count"]) for c in cl]
        assert sum(counts) == r["n_respondents"] and all(c > 0 for c in counts) and len(cl) >= 2, r["id"]
        rows.append({"qid": r["id"], "question": r["question"], "clusters": clusters, "counts": counts,
                     "n": int(r["n_respondents"])})
    return rows


def setup_nltk() -> str:
    """Point NLTK at the pinned data before nltk is imported: $NLTK_DATA if set, else /tmp/nltk_data (staged by
    n928_r9_node_setup.sh), else the zip copies on shared storage (nltk reads corpora from zips)."""
    if not os.environ.get("NLTK_DATA"):
        os.environ["NLTK_DATA"] = "/tmp/nltk_data" if Path("/tmp/nltk_data/corpora/wordnet").exists() else str(NLTK_ZIPS)
    return os.environ["NLTK_DATA"]


def pinned_stopwords() -> frozenset[str]:
    words = [w.strip() for w in STOPWORDS_FILE.read_text().splitlines() if w.strip()]
    assert len(words) == 198, len(words)
    return frozenset(words)
