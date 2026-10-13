"""Evaluate a model on ProtoQA (lists of 10 answers) with vLLM, as crs_eval.py does for Reddit-v2.

Generation: one greedy completion per question (temperature 0, the dataset's single user turn through the checkpoint's
chat template via LLM.chat, exactly as crs_eval.py). Answers: r9_common.parse_answer per non-blank line; the first 10
answers are scored.
Metrics per question (means in summary.json):
  resp_recall@k (k = 1, 3, 5, 10; selection and the selection criterion use @10): share of the question's respondents whose
      cluster is matched within the first k answers, with the registered greedy assignment (n928_r9_matcher.list_hits:
      each answer consumes at most one cluster; the gate's definition) = sum of consumed clusters' counts / respondents
  resp_recall_anymatch@k: the same with per-cluster matching (a cluster counts if any of the first k answers matches it;
      = the expected per-respondent reward of arms a-c); differs only when one answer matches two clusters
  official ProtoQA metrics (WordNet; lib/r9_official.py, verified equal to the evaluator per question): Max Answers
      @1/3/5/10/None and Max Incorrect @1/3/5/None on the first 10 answers (primary on test_crowdsourced, reported on
      every split)
  hygiene: lines, answers, exactly-10-line share, finish_reason == "length", answers matching an already-matched cluster
Outputs (<out>/<name>/): generations.jsonl, per_instance.jsonl, summary.json.

Usage (node, qenv after n928_r9_node_setup.sh):
  /tmp/qenv/bin/python remark/scripts/n928_r9_eval.py --model <ckpt> --name <tag> --split validation|test|test_crowdsourced
Re-score saved generations without a GPU (any python with nltk + scipy, e.g. the r9 scratch venv):
  python remark/scripts/n928_r9_eval.py --score_only <out>/<name>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "remark"))
from lib.r9_common import CRS_DATA, OUT, REC_NUM, answers_from_text, load_split, nonblank_lines, prompt_text  # noqa: E402

SPLIT_SRC = {"validation": "validation", "test": "test_scraped", "test_crowdsourced": "test_crowdsourced"}
KS = (1, 3, 5, 10)


def generate(a) -> list[dict]:
    from datasets import load_from_disk
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(REPO / "remark/scripts"))
    from crs_eval import prepare_qwen35

    ds = load_from_disk(str(Path(a.data) / a.split))
    qs = {q["qid"]: q for q in load_split(SPLIT_SRC[a.split])}
    rows = []
    for ex in ds:
        q = qs[ex["qid"]]
        assert ex["prompt"][0]["content"] == prompt_text(q["question"]) and ex["all_clusters"] == q["clusters"]
        rows.append({"qid": ex["qid"], "prompt": ex["prompt"][0]["content"]})
    assert len(rows) == len(qs)
    if a.limit:
        rows = rows[:a.limit]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util, max_model_len=4096,
              **prepare_qwen35(Path(a.model)))
    outs = llm.chat([[{"role": "user", "content": r["prompt"]}] for r in rows],
                    SamplingParams(temperature=0, max_tokens=a.max_tokens))
    for r, o in zip(rows, outs):
        r.update({"raw": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                  "n_tokens": len(o.outputs[0].token_ids)})
    return rows


def score(gens: list[dict], split: str) -> tuple[list[dict], dict]:
    from lib.r9_official import all_metrics
    from lib.r9_rewards import greedy_hits, match_matrix, first_hits

    qs = {q["qid"]: q for q in load_split(SPLIT_SRC[split])}
    per = []
    for g in gens:
        q = qs[g["qid"]]
        cl, cnt, n = q["clusters"], q["counts"], q["n"]
        ans = answers_from_text(g["raw"], REC_NUM)
        gh = greedy_hits(ans, cl)
        X = match_matrix(ans, cl)
        H = first_hits(X)  # (C, K)
        rec = {"qid": g["qid"], "answers": ans, "n_lines": len(nonblank_lines(g["raw"], None)),
               "n_answers_all": len(answers_from_text(g["raw"], None)), "greedy_cluster": gh,
               "finish_reason": g.get("finish_reason"),
               "redundant_answers": int(sum(1 for k in range(len(ans)) if X[k].any() and not H[:, k].any()))}
        for k in KS:
            rec[f"resp_recall@{k}"] = sum(cnt[c] for c in gh[:k] if c >= 0) / n
            rec[f"resp_recall_anymatch@{k}"] = (sum(cnt[c] for c in range(len(cl)) if H[c, :k].any()) / n
                                                if ans else 0.0)
        rec.update({f"official/{m}": v for m, v in all_metrics(ans, cl, cnt).items()})
        per.append(rec)
    keys = [k for k in per[0] if k.startswith(("resp_recall", "official/"))] if per else []
    summ = {k: statistics.mean(r[k] for r in per) for k in keys}
    summ.update({
        "n": len(per),
        "exact_10_lines": sum(r["n_lines"] == REC_NUM for r in per) / max(len(per), 1),
        "exact_10_answers_first_10_lines": sum(len(r["answers"]) == REC_NUM and r["n_lines"] == REC_NUM
                                               for r in per) / max(len(per), 1),
        "mean_lines": statistics.mean(r["n_lines"] for r in per) if per else 0.0,
        "mean_answers_scored": statistics.mean(len(r["answers"]) for r in per) if per else 0.0,
        "empty": sum(not r["answers"] for r in per),
        "finish_length": sum(r["finish_reason"] == "length" for r in per),
        "redundant_answers_per_list": statistics.mean(r["redundant_answers"] for r in per) if per else 0.0,
    })
    return per, summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--name")
    ap.add_argument("--split", default="validation", choices=list(SPLIT_SRC))
    ap.add_argument("--data", default=str(CRS_DATA / "sft_dataset"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--out", default=str(OUT / "eval"))
    ap.add_argument("--score_only", default=None, help="an existing <out>/<name> dir: re-score generations.jsonl")
    a = ap.parse_args()
    if a.score_only:
        out = Path(a.score_only)
        meta = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        split = meta.get("split", a.split)
        gens = [json.loads(x) for x in (out / "generations.jsonl").open()]
        meta = {k: meta[k] for k in ("model", "split", "data", "max_tokens") if k in meta}
    else:
        assert a.model and a.name, "--model and --name are required unless --score_only"
        out = Path(a.out) / a.name
        out.mkdir(parents=True, exist_ok=True)
        split = a.split
        gens = generate(a)
        with (out / "generations.jsonl").open("w") as f:
            for g in gens:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        meta = {"model": a.model, "split": split, "data": a.data, "max_tokens": a.max_tokens}
    per, summ = score(gens, split)
    with (out / "per_instance.jsonl").open("w") as f:
        for r in per:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {**meta, "split": split, "split_source": f"remark/data/protoqa/splits/{SPLIT_SRC[split]}.jsonl",
               "rec_num": REC_NUM, **summ}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
