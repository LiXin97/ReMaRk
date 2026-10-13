"""ProtoQA: per-position 0/1 (or weighted) rewards for lists of rec_num = 10 answers, on the official
WordNet matcher (remark/scripts/n928_r9_matcher.py, 0 disagreements with the evaluator on 184,420 pairs).

Primitive. For a list of answers a_1..a_K and the question's clusters c = 1..C, the per-cluster first-hit matrix
  H[c, k] = 1  iff  a_k matches cluster c and no earlier a_j (j < k) matches c
(`cluster_hit_matrix`). H[c] is exactly the per-respondent reward of a respondent in cluster c: a position hits the
respondent's cluster if that cluster was not matched by an earlier position (= n928_r9_matcher.list_hits(answers, [c])).

Arms (amendment, 2026-10-02 09:57 UTC):
  single (a): H[c_r], the row's respondent's cluster c_r (per-row data; Rank-GRPO analogue)
  rb     (b): the policy reward logged by TRL is (a)'s; the advantage uses every respondent's H (lib/r9_trainer._rb_hits)
  mean   (c): sum_c (n_c / n) H[c, k]  -- exactly E_r[(a)'s reward] over the question's respondents (per-row data)
  union  (d): 1 if a_k consumes a not-yet-consumed cluster in the registered greedy order (n928_r9_matcher.list_hits over
              all clusters: each position consumes at most one cluster, the first matching unconsumed one)
(c) uses the per-cluster sum, not the greedy assignment: only that sum equals the expectation of (a) when one answer
matches two clusters (checked in n928_r9_unit.py). (d) keeps the registered greedy rule; how often it differs from the
per-cluster "any cluster first hit here" rule is logged per batch (r9/union_greedy_vs_anymatch).

Reward functions follow lib.rank_grpo_trl's contract (lines mode): called with prompts, completions,
completion_ids and the dataset columns; they return, per completion, rec_num floats indexed by *valid item* (parser
order among the first rec_num non-blank lines; RankGRPOTrainer._to_line_positions re-aligns them to line positions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.r9_common import REC_NUM, answers_from_text, line_answers, pinned_stopwords, setup_nltk  # noqa: E402

setup_nltk()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import n928_r9_matcher as M  # noqa: E402

# the WordNet score depends on the stopword list: refuse to run with anything but the pinned 198-word list
assert M.EN_STOPWORDS == pinned_stopwords(), (
    f"NLTK English stopwords ({len(M.EN_STOPWORDS)} words) differ from remark/data/protoqa/nltk_stopwords_english.txt; "
    "set NLTK_DATA to the staged data (n928_r9_node_setup.sh)")

KINDS = ("single", "rb", "mean", "union")


def matches(answer: str, cluster) -> bool:
    """Official WordNet match (rounded score) of one predicted answer against a cluster (any of its strings)."""
    return M.match_cluster(answer, tuple(cluster))


def match_matrix(answers, clusters) -> np.ndarray:
    """X[k, c] = answer k matches cluster c."""
    X = np.zeros((len(answers), len(clusters)), dtype=bool)
    for k, a in enumerate(answers):
        p = M.norm_pred(a)
        for c, cl in enumerate(clusters):
            X[k, c] = any(M.match_norm(p, t) for t in cl)
    return X


def first_hits(X: np.ndarray) -> np.ndarray:
    """(K, C) match matrix -> (C, K) per-cluster first-hit matrix."""
    if X.size == 0:
        return np.zeros((X.shape[1], X.shape[0]), dtype=np.int8)
    before = np.cumsum(X, axis=0) - X  # earlier matches of each cluster
    return (X & (before == 0)).T.astype(np.int8)


def cluster_hit_matrix(answers, clusters) -> np.ndarray:
    return first_hits(match_matrix(answers, clusters))


def greedy_hits(answers, clusters) -> list[int]:
    """Registered greedy rule (n928_r9_matcher.list_hits): per position the consumed cluster index or -1."""
    return [c for _, c in M.list_hits(list(answers), [list(x) for x in clusters])]


def _text(c) -> str:
    return c[0]["content"] if isinstance(c, list) else c


def _pad(v, rec_num: int) -> list[float]:
    v = [float(x) for x in v][:rec_num]
    return v + [0.0] * (rec_num - len(v))


def item_rewards(kind: str, answers, clusters, counts, resp_cluster: int, rec_num: int = REC_NUM) -> list[float]:
    """Per valid item (answers already parsed, in order), padded to rec_num."""
    answers = list(answers)[:rec_num]
    if kind in ("single", "rb"):
        if resp_cluster < 0:
            raise ValueError("single / rb rewards need the per-row split (resp_cluster >= 0)")
        return _pad(cluster_hit_matrix(answers, [clusters[resp_cluster]])[0], rec_num)
    if kind == "mean":
        p = np.asarray(counts, dtype=np.float64) / float(sum(counts))
        return _pad(p @ cluster_hit_matrix(answers, clusters) if answers else [], rec_num)
    if kind == "union":
        return _pad([int(c >= 0) for c in greedy_hits(answers, clusters)], rec_num)
    raise ValueError(kind)


def line_cluster_hits(text: str, clusters, rec_num: int = REC_NUM) -> np.ndarray:
    """(C, rec_num) per-cluster first-hit matrix at line positions (the first rec_num non-blank lines; a malformed line
    is a miss that consumes nothing). Used by the RB estimator."""
    la = line_answers(text, rec_num)
    pos = [k for k, a in enumerate(la) if a]
    out = np.zeros((len(clusters), rec_num), dtype=np.int8)
    if pos:
        out[:, pos] = cluster_hit_matrix([la[k] for k in pos], clusters)
    return out


def make_r9_reward(kind: str, rec_num: int = REC_NUM):
    if kind not in KINDS:
        raise ValueError(kind)

    def reward(prompts=None, completions=None, all_clusters=None, cluster_counts=None, resp_cluster=None, **kwargs):
        out, diff, pos = [], 0, 0
        for i, comp in enumerate(completions):
            answers = answers_from_text(_text(comp), rec_num)
            r = resp_cluster[i] if resp_cluster is not None else -1
            out.append(item_rewards(kind, answers, all_clusters[i], cluster_counts[i], r, rec_num))
            if kind == "union":  # registered greedy vs per-cluster "some cluster first hit here"
                anym = cluster_hit_matrix(answers, all_clusters[i]).max(axis=0) if answers else []
                diff += sum(int(a) != int(b) for a, b in zip(anym, out[-1]))
                pos += len(answers)
        reward.last_stats = {"r9/union_greedy_vs_anymatch": diff / max(pos, 1)} if kind == "union" else {}
        return out

    reward.__name__ = f"r9_{kind}"
    reward.last_stats = {}
    return reward
