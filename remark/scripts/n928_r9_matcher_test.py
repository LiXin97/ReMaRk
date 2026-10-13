"""The official ProtoQA evaluator vs our fast matcher (agreement, speed, single-string damage).

Run on the CPU host with the scratch venv that has the official evaluator (iesl/protoqa-evaluator @860dfd2, pip -e)
and NLTK data (wordnet, stopwords, punkt, punkt_tab):
  NLTK_DATA=<scratch>/nltk_data <scratch>/r9venv/bin/python remark/scripts/n928_r9_matcher_test.py
Output: remark/data/protoqa/r9_matcher_test.json. No GPU, no API.

Rebuild of the scratch environment (as run on 2026-10-02; <S> = the session scratchpad or any temp dir):
  python3 -m venv <S>/r9venv                                   # system Python 3.11.2
  git clone --depth 1 https://github.com/iesl/protoqa-evaluator.git <S>/r9_evaluator   # commit 860dfd2
  <S>/r9venv/bin/pip install -e <S>/r9_evaluator pytest        # nltk 3.10.3, numpy 2.4.6, scipy 1.17.1
  # nltk.download() refuses the proxied egress here (SSRF guard), so fetch the data zips directly:
  B=https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages
  mkdir -p <S>/nltk_data/corpora <S>/nltk_data/tokenizers
  cd <S>/nltk_data/corpora && for p in stopwords wordnet omw-1.4; do curl -sSfL -o $p.zip $B/corpora/$p.zip && unzip -q -o $p.zip; done
  cd <S>/nltk_data/tokenizers && for p in punkt punkt_tab; do curl -sSfL -o $p.zip $B/tokenizers/$p.zip && unzip -q -o $p.zip; done
The stopword list fetched this way has 198 English words (copy: remark/data/protoqa/nltk_stopwords_english.txt);
older NLTK data releases have 179, and the official WordNet score depends on the list, so training nodes must use the
same file.
"""

from __future__ import annotations

import json
import random
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
from protoqa_evaluator.common_evaluations import all_eval_funcs
from protoqa_evaluator.data_processing import QuestionAndAnswerClusters, load_question_answer_clusters_from_jsonl
from protoqa_evaluator.evaluation import evaluate, general_eval
from protoqa_evaluator.scoring import wordnet_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import n928_r9_matcher as M  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "remark/data/protoqa/raw/data"
OUT = REPO / "remark/data/protoqa/r9_matcher_test.json"
EVAL_REPO = Path(sys.modules["protoqa_evaluator"].__path__[0]).resolve().parents[1]


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open()]


def official_means(qdata: dict, answers: dict) -> dict:
    out = {}
    for sim in ("exact_match", "wordnet"):
        for name, fn in all_eval_funcs[sim].items():
            r = evaluate(fn, qdata, answers_dict={k: answers[k] for k in qdata})
            out[f"{sim} | {name}"] = round(statistics.mean(x.score for x in r.values()), 4)
    return out


def single_string(qdata: dict, raw_rows: dict) -> dict:
    """Simulate the scraped format: every cluster keeps one string, its most frequent raw answer (ties: shorter, then
    alphabetical). The scraped data keep the show's board label instead, which we cannot reconstruct."""
    out = {}
    for qid, q in qdata.items():
        raw = raw_rows[qid]["answers"]["raw"]
        cl = {}
        for strings, cnt in q.answer_clusters.items():
            rep = sorted(strings, key=lambda s: (-raw.get(s, 0), len(s), s))[0]
            cl[frozenset([rep])] = cnt
        out[qid] = QuestionAndAnswerClusters(q.question_id, q.question, cl)
    return out


def official_pair(pred_norm: str, true: str) -> int:
    return int(np.round(wordnet_score(pred_norm, true)))


def main() -> None:
    res: dict = {"evaluator": {"repo": "https://github.com/iesl/protoqa-evaluator",
                               "commit": subprocess.run(["git", "-C", str(EVAL_REPO), "rev-parse", "HEAD"],
                                                        capture_output=True, text=True).stdout.strip(),
                               "licence": "Apache-2.0 (evaluator code; separate from the CC BY 4.0 data)",
                               "pypi": "no 'protoqa-evaluator' distribution on PyPI (pip download fails); "
                                       "installed from the git clone with pip -e",
                               "unit_tests": "pytest: 102 passed; 1 failed + 17 errors all need pandas/openpyxl "
                                             "(Excel loaders, optional extra 'crowdsource-conversion'), unrelated to "
                                             "scoring"}}
    t_load = time.time()
    wordnet_score("dog", "dogs")
    res["wordnet_first_call_load_s"] = round(time.time() - t_load, 2)

    cs_path = RAW / "dev/dev.crowdsourced.jsonl"
    qdata = load_question_answer_clusters_from_jsonl(cs_path)
    raw_rows = {r["metadata"]["id"]: r for r in jl(cs_path)}
    human = {k: v for d in jl(RAW / "dev/dev.predictions.human.jsonl") for k, v in d.items()}
    gpt2 = json.load((RAW / "dev/dev.predictions.gpt2finetuned.json").open())
    res["shipped_predictions"] = {"n_questions": len(qdata), "human_lists": len(human), "gpt2_lists": len(gpt2),
                                  "human_list_len_mean": round(statistics.mean(len(v) for v in human.values()), 2),
                                  "gpt2_list_len_mean": round(statistics.mean(len(v) for v in gpt2.values()), 2)}

    # A. official evaluator, unmodified, on the shipped prediction files (dev.crowdsourced)
    single = single_string(qdata, raw_rows)
    res["A_official_dev_crowdsourced"] = {
        "note": "means over the 52 dev.crowdsourced questions; not compared with the paper's tables (unverified)",
        "human_full_clusters": official_means(qdata, human),
        "gpt2_full_clusters": official_means(qdata, gpt2),
        "human_single_string_clusters": official_means(single, human),
        "gpt2_single_string_clusters": official_means(single, gpt2),
    }

    # B. pair-level: does each predicted answer match some cluster (official WordNet, rounded)?
    def answer_level(qd: dict, preds: dict) -> dict:
        n = hit = 0
        for qid, q in qd.items():
            for a in dict.fromkeys(M.norm_pred(x) for x in preds[qid]):
                n += 1
                hit += any(max(official_pair(a, s) for s in strings) for strings in q.answer_clusters)
        return {"answers": n, "matched": hit, "share": round(hit / n, 4)}

    def respondent_level(qd: dict, full: dict, preds: dict, k: int | None = None) -> float:
        """Share of a question's respondents (cluster counts) whose cluster is hit by the list, matching on `qd`'s
        strings, counted on the full clusters' sizes; mean over questions."""
        vals = []
        for qid, q in qd.items():
            cl = [list(s) for s in q.answer_clusters]
            cnt = list(full[qid].answer_clusters.values())
            hits = M.list_hits(preds[qid], cl, k)
            vals.append(sum(cnt[c] for h, c in hits if h) / sum(cnt))
        return round(statistics.mean(vals), 4)

    res["B_single_string_damage"] = {
        "definition": "unique normalised answers of the shipped lists that match >= 1 cluster (official wordnet_score, "
                      "np.round); 'single' keeps one string per cluster as in the scraped train/dev data",
        "human_full": answer_level(qdata, human), "human_single": answer_level(single, human),
        "gpt2_full": answer_level(qdata, gpt2), "gpt2_single": answer_level(single, gpt2),
        "per_respondent_hit_share_top10": {
            "human_full": respondent_level(qdata, qdata, human, 10),
            "human_single": respondent_level(single, qdata, human, 10),
            "gpt2_full": respondent_level(qdata, qdata, gpt2, 10),
            "gpt2_single": respondent_level(single, qdata, gpt2, 10)},
    }

    # C. agreement of the fast matcher with the official pair score on every (answer, cluster string) pair
    pairs = set()
    for qid, q in qdata.items():
        strings = [s for cl in q.answer_clusters for s in cl]
        answers = set(M.norm_pred(a) for a in human[qid] + gpt2[qid] + list(raw_rows[qid]["answers"]["raw"]))
        pairs.update((a, s) for a in answers for s in strings)
    hand = json.load((REPO / "remark/data/protoqa/r9_hand_lists.json").open())
    scraped = {r["metadata"]["id"]: r for r in jl(RAW / "dev/dev.scraped.jsonl")}
    for qid, lst in hand["lists"].items():
        for a in lst:
            for c in scraped[qid]["answers"]["clusters"].values():
                pairs.update((M.norm_pred(a), s) for s in c["answers"])
    pairs = sorted(pairs)
    t = time.time()
    off = [official_pair(a, s) for a, s in pairs]
    t_off = time.time() - t
    M.cache_clear()
    t = time.time()
    fast = [int(M.match_norm(a, s)) for a, s in pairs]
    t_fast_cold = time.time() - t
    t = time.time()
    fast2 = [int(M.match_norm(a, s)) for a, s in pairs]
    t_fast_warm = time.time() - t
    assert fast == fast2
    dis = [(a, s, o, f) for (a, s), o, f in zip(pairs, off, fast) if o != f]
    res["C_agreement"] = {
        "pairs": len(pairs), "official_positive": int(sum(off)), "fast_positive": int(sum(fast)),
        "disagreements": len(dis), "disagreement_examples": dis[:20],
        "pair_sources": "dev.crowdsourced: (shipped human + gpt2 predictions + raw answers) x all cluster strings of "
                        "the same question; dev.scraped hand lists x cluster strings",
        "speed_pairs_per_s": {"official_wordnet_score": round(len(pairs) / t_off),
                              "fast_cold_cache": round(len(pairs) / t_fast_cold),
                              "fast_warm_cache": round(len(pairs) / t_fast_warm)},
        "seconds": {"official": round(t_off, 2), "fast_cold": round(t_fast_cold, 3), "fast_warm": round(t_fast_warm, 4)},
    }

    # D. list-level reproduction: our matrix + official Munkres/oracle vs official general_eval (WordNet)
    def munkres_with(fast_match, preds, q, k=None, max_inc=None):
        fn = partial(general_eval, score_func=lambda a, b: float(fast_match(a, b)),
                     score_matrix_transformation=np.round, max_pred_answers=k, max_incorrect=max_inc)
        return fn(preds, q.answer_clusters.copy(), question_string=q.question).score

    from functools import lru_cache
    wn_cached = lru_cache(maxsize=None)(wordnet_score)  # the official function, memoised (same values, faster)
    worst, greedy_gap = 0.0, []
    for qid, q in qdata.items():
        for k, mi in ((1, None), (3, None), (5, None), (10, None), (None, 1), (None, 3), (None, 5)):
            for preds in (human[qid], gpt2[qid]):
                o = general_eval(preds, q.answer_clusters.copy(), score_func=wn_cached,
                                 score_matrix_transformation=np.round, max_pred_answers=k, max_incorrect=mi,
                                 question_string=q.question).score
                f = munkres_with(M.match_norm, preds, q, k, mi)
                worst = max(worst, abs(o - f))
                if k is not None:  # greedy first-come (reward semantics) vs Munkres, Max Answers@k
                    cl = [list(s) for s in q.answer_clusters]
                    cnt = list(q.answer_clusters.values())
                    oracle = sum(sorted(cnt, reverse=True)[:k])
                    g = sum(cnt[c] for h, c in M.list_hits(preds, cl, k) if h) / oracle
                    greedy_gap.append(o - g)
    res["D_list_level"] = {"max_abs_diff_official_vs_fast_in_official_munkres": worst,
                           "greedy_reward_vs_munkres_max_answers": {
                               "n": len(greedy_gap), "mean_gap": round(statistics.mean(greedy_gap), 5),
                               "share_nonzero": round(sum(1 for x in greedy_gap if abs(x) > 1e-12) / len(greedy_gap), 4),
                               "max_gap": round(max(greedy_gap), 4)}}

    # multi-cluster answers (the only case where greedy first-come and Munkres can differ)
    multi = tot = 0
    for qid, q in qdata.items():
        cl = [list(s) for s in q.answer_clusters]
        for a in set(M.norm_pred(x) for x in human[qid] + gpt2[qid]):
            m = sum(any(M.match_norm(a, s) for s in c) for c in cl)
            tot += m > 0
            multi += m > 1
    res["D_list_level"]["answers_matching_ge2_clusters"] = {"matched_answers": tot, "ge2": multi}

    # E. rounding check and examples
    ex = [("red car", "car"), ("sun hat", "hat"), ("bathing suit", "swimsuit"), ("x-ray vision", "x ray vision"),
          ("peach", "peaches"), ("noise", "noice"), ("fly", "flying"), ("the", "it"), ("the", "car"),
          ("tv", "television"), ("sun glasses", "sunglasses"), ("bread", "bread/yeast"), ("sunblock", "sunscreen"),
          ("knock on the door", "knock on door"), ("throw tomatoes", "throw things")]
    res["E_examples"] = [{"pred": a, "true": b, "official_score": round(float(wordnet_score(a, b)), 4),
                          "official_rounded": official_pair(a, b), "fast": int(M.match(a, b)),
                          "exact_after_norm": int(M.norm_pred(a) == b)} for a, b in ex]
    res["E_rounding"] = {"np_round_0.5": float(np.round(0.5)), "note": "half-to-even: a 2-group answer with one "
                         "matching group (score 0.5) does not match ('red car' vs 'car')"}

    # F. hand-made lists on dev.scraped (official per-answer assignment, WordNet and exact)
    hq = {qid: QuestionAndAnswerClusters(qid, scraped[qid]["question"]["normalized"],
                                         {frozenset(c["answers"]): c["count"]
                                          for c in scraped[qid]["answers"]["clusters"].values()})
          for qid in hand["lists"]}
    fl = []
    for qid, lst in hand["lists"].items():
        q = hq[qid]
        wn10 = general_eval(lst, q.answer_clusters.copy(), score_func=wordnet_score,
                            score_matrix_transformation=np.round, max_pred_answers=10, question_string=q.question)
        ex10 = general_eval(lst, q.answer_clusters.copy(), max_pred_answers=10, question_string=q.question)
        cl = [list(s) for s in q.answer_clusters]
        fl.append({"qid": qid, "question": q.question,
                   "clusters": {next(iter(s)): c for s, c in q.answer_clusters.items()}, "list": lst,
                   "official_wordnet_assignment": {a: (next(iter(c)) if c else None)
                                                   for a, c in wn10.answer_assignment.items()},
                   "official_wordnet_max_answers_10": round(wn10.score, 4),
                   "official_exact_max_answers_10": round(ex10.score, 4),
                   "fast_list_hits": [h for h, _ in M.list_hits(lst, cl, 10)]})
    res["F_hand_lists_dev_scraped"] = fl

    # G. speed at training scale: 384 lists x 10 answers against train questions' clusters (one string each)
    train = jl(RAW / "train/train.jsonl")
    pool = [c["answers"][0] for r in train for c in r["answers"]["clusters"].values()]
    rng = random.Random(42)
    qs = rng.sample(train, 48)  # 48 requests x G=8 lists = 384 lists
    lists = []
    for r in qs:
        own = [c["answers"][0] for c in r["answers"]["clusters"].values()]
        for _ in range(8):
            cand = own[:] + rng.sample(pool, 12)
            rng.shuffle(cand)
            lists.append((cand[:10], [c["answers"] for c in r["answers"]["clusters"].values()]))
    n_pairs = sum(len(a) * sum(len(c) for c in cl) for a, cl in lists)
    t = time.time()
    for a, cl in lists[:48]:
        for x in a:
            for c in cl:
                max(official_pair(M.norm_pred(x), s) for s in c)
    t_off_step = (time.time() - t) * len(lists) / 48
    M.cache_clear()
    t = time.time()
    for a, cl in lists:
        M.list_hits(a, cl, 10)
    t_cold = time.time() - t
    t = time.time()
    for a, cl in lists:
        M.list_hits(a, cl, 10)
    t_warm = time.time() - t
    res["G_training_step_speed"] = {
        "setup": "48 train questions x 8 lists x 10 answers (own cluster strings mixed with random train answer "
                 "strings), seed 42; one CPU core",
        "pairs_per_step_upper_bound": n_pairs,
        "official_s_per_step_extrapolated_from_48_lists": round(t_off_step, 2),
        "fast_list_hits_s_per_step_cold_cache": round(t_cold, 3),
        "fast_list_hits_s_per_step_warm_cache": round(t_warm, 4),
    }
    res["fast_cache_info"] = M.cache_info()
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("F_hand_lists_dev_scraped", "fast_cache_info")},
                     indent=1))


if __name__ == "__main__":
    main()
