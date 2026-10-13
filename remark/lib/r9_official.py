"""ProtoQA's official list metrics (Max Answers@k, Max Incorrect@k, WordNet similarity) re-implemented on the
ProtoQA matcher, so they run where the evaluator package is not installed (training nodes).

Mirrors protoqa_evaluator.evaluation.general_eval (iesl/protoqa-evaluator @860dfd2) with
common_evaluations.wordnet_all_eval_funcs (score_func=wordnet_score, score_matrix_transformation=np.round):
  1. predictions truncated to max_pred_answers, then lower()[:50].strip();
  2. 0/1 matrix S[answer, cluster] = rounded WordNet score, max over the cluster's strings (n928_r9_matcher.match_norm,
     identical to the rounded official score on 184,420 pairs; round(max) = max(round) since rounding is monotone);
  3. max_incorrect: rows kept up to and including the k-th answer that matches no cluster (limit_total_wrong);
  4. S multiplied by the cluster counts; Munkres (scipy linear_sum_assignment) maximises the matched count;
  5. divided by the same score of the oracle list (one string per cluster, clusters by count descending, the
     string = first element of the cluster's frozenset, as the official code takes it).
Empty prediction lists score 0 (the official code returns 0 for Max Answers and raises for Max Incorrect).
Verified per question against the official evaluator on its shipped dev.crowdsourced human and GPT-2 predictions by
remark/scripts/n928_r9_official_check.py.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from lib.r9_rewards import M  # the matcher module, NLTK already pinned

METRICS = {f"Max Answers - {k}": (k, None) for k in (None, 1, 3, 5, 10)}
METRICS.update({f"Max Incorrect - {k}": (None, k) for k in (None, 1, 3, 5)})


def _pre(s: str) -> str:
    return s.lower()[:50].strip()


def _limit_total_wrong(S: np.ndarray, k: int) -> np.ndarray:
    incorrect = 0
    for i, a in enumerate(S.max(axis=1)):
        if a == 0:
            incorrect += 1
            if incorrect >= k:
                return S[: i + 1]
    return S


def _score(preds, clusters, counts, k, max_inc) -> float:
    if k is not None:
        preds = preds[:k]
    preds = [_pre(p) for p in preds]
    if not preds:
        return 0.0
    S = np.array([[float(any(M.match_norm(p, t) for t in cl)) for cl in clusters] for p in preds])
    if max_inc is not None:
        S = _limit_total_wrong(S, max_inc)
    S = S * np.asarray(counts, dtype=float)[None]
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].sum())


def general_eval(preds, clusters, counts, max_pred_answers=None, max_incorrect=None) -> float:
    order = sorted(range(len(clusters)), key=lambda j: counts[j], reverse=True)  # stable, as the official sort
    oracle = [next(iter(frozenset(clusters[j]))) for j in order]
    return _score(list(preds), clusters, counts, max_pred_answers, max_incorrect) / _score(
        oracle, clusters, counts, max_pred_answers, max_incorrect)


def all_metrics(preds, clusters, counts) -> dict[str, float]:
    return {name: general_eval(preds, clusters, counts, k, mi) for name, (k, mi) in METRICS.items()}
