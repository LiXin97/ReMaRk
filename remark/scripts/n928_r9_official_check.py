"""Verify lib/r9_official.py (our re-implementation of ProtoQA's Max Answers / Max Incorrect, WordNet) against the
official evaluator, per question, on the predictions shipped with the data (dev.crowdsourced: human and fine-tuned GPT-2).

Run with the scratch venv that has the official evaluator (see n928_r9_matcher_test.py for the rebuild):
  NLTK_DATA=<scratch>/nltk_data <scratch>/r9venv/bin/python remark/scripts/n928_r9_official_check.py
Output: remark/data/protoqa/r9_official_check.json. No GPU, no API.
Checks: (i) ours on the official loader's clusters vs the official general_eval, all 9 WordNet metrics, every question,
both prediction files; (ii) ours on splits/test_crowdsourced.jsonl (the clusters our evaluator reads) vs the official;
(iii) the same with predictions cut to their first 10 answers (what n928_r9_eval.py passes).
"""

from __future__ import annotations

import json
import statistics
import sys
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
from protoqa_evaluator.common_evaluations import wordnet_all_eval_funcs
from protoqa_evaluator.data_processing import load_question_answer_clusters_from_jsonl
from protoqa_evaluator.evaluation import general_eval as official_general_eval
from protoqa_evaluator.scoring import wordnet_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.r9_common import PROTOQA, load_split  # noqa: E402
from lib.r9_official import METRICS, general_eval  # noqa: E402

RAW = PROTOQA / "raw/data/dev"
WN_CACHED = lru_cache(maxsize=None)(wordnet_score)  # the official score function, memoised (same values, faster)


def official_fn(metric: str):
    """wordnet_all_eval_funcs[metric] with the memoised official wordnet_score (the partial's other arguments kept)."""
    f = wordnet_all_eval_funcs[metric]
    assert f.keywords["score_func"] is wordnet_score and f.keywords["score_matrix_transformation"] is np.round
    return partial(official_general_eval, **{**f.keywords, "score_func": WN_CACHED})


def main() -> None:
    qdata = load_question_answer_clusters_from_jsonl(RAW / "dev.crowdsourced.jsonl")
    preds = {"human": {k: v for line in (RAW / "dev.predictions.human.jsonl").open() for k, v in json.loads(line).items()},
             "gpt2": json.load((RAW / "dev.predictions.gpt2finetuned.json").open())}
    split = {q["qid"]: q for q in load_split("test_crowdsourced")}
    assert set(split) == set(qdata), (len(split), len(qdata))
    same_clusters = all(
        sorted((tuple(sorted(c)), n) for c, n in zip(split[k]["clusters"], split[k]["counts"]))
        == sorted((tuple(sorted(fs)), n) for fs, n in q.answer_clusters.items()) for k, q in qdata.items())
    res = {"questions": len(qdata), "split_clusters_equal_official_loader": same_clusters, "checks": {}}
    for cut in (None, 10):
        for name, P in preds.items():
            for src in ("official_clusters", "split_clusters"):
                worst, n_diff, off_means, our_means = 0.0, 0, {}, {}
                for metric in METRICS:
                    fn = official_fn(metric)
                    k, mi = METRICS[metric]
                    o_all, m_all = [], []
                    for qid, q in qdata.items():
                        p = P[qid][:cut] if cut else P[qid]
                        o = fn(p, q.answer_clusters.copy(), question_string=q.question).score
                        if src == "official_clusters":
                            cl = [list(fs) for fs in q.answer_clusters]
                            cnt = list(q.answer_clusters.values())
                        else:
                            cl, cnt = split[qid]["clusters"], split[qid]["counts"]
                        m = general_eval(p, cl, cnt, k, mi)
                        worst = max(worst, float(abs(o - m)))
                        n_diff += int(abs(o - m) > 1e-9)
                        o_all.append(float(o))
                        m_all.append(m)
                    off_means[metric] = round(statistics.mean(o_all), 6)
                    our_means[metric] = round(statistics.mean(m_all), 6)
                res["checks"][f"{name}|{src}|first_{cut or 'all'}"] = {
                    "max_abs_diff": worst, "question_metric_pairs_differing": n_diff,
                    "official_means": off_means, "ours_means": our_means}
                print(f"{name:6s} {src:18s} cut={cut}: max |diff| {worst:.2e}, differing {n_diff}", flush=True)
    res["all_equal"] = all(v["question_metric_pairs_differing"] == 0 for v in res["checks"].values())
    out = PROTOQA / "r9_official_check.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"all_equal": res["all_equal"], "split_clusters_equal_official_loader": same_clusters}))


if __name__ == "__main__":
    main()
