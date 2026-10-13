"""Fast deterministic 0/1 answer-to-cluster matcher for ProtoQA training rewards.

Reproduces the official ProtoQA evaluator's WordNet similarity (iesl/protoqa-evaluator, commit 860dfd2:
scoring.wordnet_score, used by common_evaluations.wordnet_all_eval_funcs with score_matrix_transformation=np.round):
  - the prediction is normalised by data_processing.default_string_preprocessing: lower()[:50].strip();
    target strings are used as stored (all ProtoQA cluster strings are already lower case);
  - tokens = nltk.word_tokenize(s) without NLTK English stopwords;
  - every order-preserving partition of the tokens into contiguous groups is tried on both sides;
  - two groups match iff they are equal strings or wn.synsets(g.replace(" ", "_")) of both share a synset
    (wn.synsets applies WordNet's morphy, so "peach" ~ "peaches", "fly" ~ "flying");
  - a partition pair scores (maximum matching) / max(|P|, |Q|); the string pair scores the max over partition pairs;
  - np.round turns the score into 0/1 (round half to even, so exactly 0.5 -> 0);
  - a cluster scores the max over its strings.
Hence match(pred, true) <=> some partition pair has 2 * maxmatch > max(|P|, |Q|). A string whose tokens are all
stopwords (or an empty string) becomes the single empty group "" (more_itertools.partitions([]) == [[[]]]), so it
matches exactly the other all-stopword strings ("the" ~ "it") and nothing else.

Reward semantics (mirrors Rank-GRPO's evaluate_direct_match_aligned, which consumes each ground-truth item once):
`list_hits` walks the list in rank order and gives position k a 1 if its answer matches a cluster not yet matched by an
earlier position (first matching unmatched cluster in cluster order). The official evaluator instead assigns answers
to clusters with Munkres on count-weighted scores; the two differ only when one answer matches >= 2 clusters
(measured in n928_r9_matcher_test.py).

Requires nltk and the WordNet / stopwords / punkt_tab data (NLTK_DATA). Pure CPU, memoised per string and per pair.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn

EN_STOPWORDS = frozenset(stopwords.words("english"))


def norm_pred(s: str) -> str:
    """Official default_string_preprocessing."""
    return s.lower()[:50].strip()


@lru_cache(maxsize=None)
def _tokens(s: str) -> tuple[str, ...]:
    return tuple(t for t in word_tokenize(s) if t not in EN_STOPWORDS)


@lru_cache(maxsize=None)
def _synsets(group: str) -> frozenset[str]:
    return frozenset(x.name() for x in wn.synsets(group.replace(" ", "_")))


@lru_cache(maxsize=None)
def _partitions(s: str) -> tuple[tuple[str, ...], ...]:
    toks = _tokens(s)
    n = len(toks)
    if n == 0:
        # official: more_itertools.partitions([]) == [[[]]] -> one partition holding the empty group "", so an
        # all-stopword string equals any other all-stopword string (exact match "" == "") and nothing else
        return (("",),)
    out = []
    for mask in range(1 << (n - 1)):
        groups, start = [], 0
        for i in range(n - 1):
            if mask >> i & 1:
                groups.append(" ".join(toks[start:i + 1]))
                start = i + 1
        groups.append(" ".join(toks[start:]))
        out.append(tuple(groups))
    return tuple(out)


@lru_cache(maxsize=None)
def _spans(s: str) -> frozenset[str]:
    """All contiguous token spans (the groups any partition can contain)."""
    toks = _tokens(s)
    if not toks:
        return frozenset({""})
    return frozenset(" ".join(toks[i:j]) for i in range(len(toks)) for j in range(i + 1, len(toks) + 1))


@lru_cache(maxsize=None)
def _edge(a: str, b: str) -> bool:
    return a == b or bool(_synsets(a) & _synsets(b))


def _max_matching(p: Sequence[str], q: Sequence[str]) -> int:
    """Maximum bipartite matching between groups (Kuhn's augmenting paths; groups are few)."""
    adj = [[j for j, b in enumerate(q) if _edge(a, b)] for a in p]
    match_q = [-1] * len(q)

    def augment(i: int, seen: list[bool]) -> bool:
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                if match_q[j] < 0 or augment(match_q[j], seen):
                    match_q[j] = i
                    return True
        return False

    return sum(augment(i, [False] * len(q)) for i in range(len(p)))


@lru_cache(maxsize=1 << 20)
def match_norm(pred: str, true: str) -> bool:
    """0/1 match for an already normalised prediction and a target string."""
    P, Q = _partitions(pred), _partitions(true)
    # every group of every partition is a contiguous token span; if no span pair has an edge, nothing can match
    if not any(_edge(a, b) for a in _spans(pred) for b in _spans(true)):
        return False
    for p in P:
        for q in Q:
            if 2 * _max_matching(p, q) > max(len(p), len(q)):
                return True
    return False


def match(pred: str, true: str) -> bool:
    return match_norm(norm_pred(pred), true)


def match_cluster(pred: str, cluster: Iterable[str]) -> bool:
    p = norm_pred(pred)
    return any(match_norm(p, t) for t in cluster)


def list_hits(answers: Sequence[str], clusters: Sequence[Sequence[str]], k: int | None = None) -> list[tuple[int, int]]:
    """Per rank position: (hit 0/1, index of the cluster it consumed or -1). Each cluster is credited at most once."""
    answers = list(answers)[:k] if k else list(answers)
    used, out = set(), []
    for a in answers:
        p = norm_pred(a)
        hit = -1
        for ci, cl in enumerate(clusters):
            if ci not in used and any(match_norm(p, t) for t in cl):
                hit = ci
                break
        if hit >= 0:
            used.add(hit)
        out.append((int(hit >= 0), hit))
    return out


def cache_info() -> dict:
    return {"match_norm": match_norm.cache_info()._asdict(), "tokens": _tokens.cache_info()._asdict(),
            "synsets": _synsets.cache_info()._asdict()}


def cache_clear() -> None:
    for f in (match_norm, _tokens, _synsets, _partitions, _spans, _edge):
        f.cache_clear()
