"""GPT-5.6 teacher lists (10 answers per question) for the ProtoQA SFT targets, plus teacher memorization stats.

One call per question of the train (8,581) and validation (400) splits. The user message is exactly the dataset prompt
(lib/r9_common.prompt_text: HEADER + "Question: <question>"); the system message asks for exactly 10 lines, the 10 most
common answers of 100 surveyed people, most common first, one short answer per line, no numbering or extra words. The
question is the only input: no answer of any board is in the prompt, so SFT never sees the clusters or their union.
Lines are parsed with r9_common.parse_answer (numbering / markup stripped); the first 10 non-empty answers are kept.

Usage (CPU):
  # API run (repo env; resumable cache, workers 48)
  env/bin/python remark/scripts/n928_r9_teacher.py [--limit 20]
  # memorization / duplicate stats (needs nltk: the r9 scratch venv or a node after n928_r9_node_setup.sh)
  NLTK_DATA=... <python with nltk> remark/scripts/n928_r9_teacher.py --stats_only
Outputs (remark/outputs/protoqa/teacher/): api_cache.jsonl (batch_chat cache), teacher_lists.jsonl
({qid, split, answers, n_lines_raw, n_answers, ok}), stats.json (counts; memorization block with --stats_only), and
remark/outputs/protoqa/api_usage.json (calls and tokens).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.r9_common import OUT, REC_NUM, load_split, nonblank_lines, parse_answer, prompt_text  # noqa: E402

VERSION = "v1"
TDIR = OUT / "teacher"
CACHE = TDIR / "api_cache.jsonl"
LISTS = TDIR / "teacher_lists.jsonl"
SYSTEM = ("You answer survey-guessing questions. 100 people were surveyed with the question in the user's message. "
          "Reply with exactly 10 lines: the 10 answers those 100 people would give most often, the most common answer "
          "first. Each line holds one short answer (a few words) and nothing else: no numbering, no bullets, no "
          "explanations, no extra words. The 10 answers must be different from each other.")


def req_id(qid: str) -> str:
    return f"r9t{VERSION}:{qid}"


def parse_teacher(text: str) -> tuple[list[str], int]:
    lines = nonblank_lines(text or "", None)
    answers = [a for a in (parse_answer(x) for x in lines) if a]
    return answers[:REC_NUM], len(lines)


def log_usage(cache: Path) -> dict:
    n = err = pt = ct = rt = rl = 0
    lat = []
    if cache.exists():
        for line in cache.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                err += 1
                continue
            n += 1
            u = r.get("usage") or {}
            pt += u.get("prompt_tokens", 0) or 0
            ct += u.get("completion_tokens", 0) or 0
            rt += ((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
            rl += r.get("rate_limited") or 0
            if r.get("latency_s"):
                lat.append(r["latency_s"])
    rec = {"stage": "n928_r9_teacher", "cache": str(cache.relative_to(OUT.parents[1])), "model": "gpt-5.6",
           "calls_ok": n, "error_records": err, "prompt_tokens": pt, "completion_tokens": ct, "reasoning_tokens": rt,
           "rate_limited_retries": rl, "latency_s_median": statistics.median(lat) if lat else None}
    path = OUT / "api_usage.json"
    allu = json.loads(path.read_text()) if path.exists() else {}
    allu["teacher"] = rec
    path.write_text(json.dumps(allu, indent=1))
    return rec


def run_api(limit: int, workers: int) -> None:
    from lib.llm import batch_chat  # API client only for the API run (its env has openai, not nltk)
    rows = [(r, "train") for r in load_split("train")] + [(r, "validation") for r in load_split("validation")]
    rng = random.Random(42)
    rng.shuffle(rows)  # calls run in this order, so a partial run covers a random subset of questions
    if limit:
        rows = rows[:limit]
    reqs = [{"id": req_id(r["qid"]), "max_tokens": 4000, "meta": {"qid": r["qid"], "split": s},
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt_text(r["question"])}]}
            for r, s in rows]
    print(f"{len(reqs)} teacher calls", flush=True)
    t0 = time.time()
    res = batch_chat(reqs, CACHE, workers=workers)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    n10 = 0
    out = []
    for r, s in sorted(rows, key=lambda x: (x[1], x[0]["qid"])):
        rec = res.get(req_id(r["qid"]))
        answers, n_lines = parse_teacher(rec["text"]) if rec else ([], 0)
        n10 += len(answers) == REC_NUM
        out.append({"qid": r["qid"], "split": s, "answers": answers, "n_lines_raw": n_lines,
                    "n_answers": len(answers), "ok": rec is not None})
    if not limit:
        with LISTS.open("w") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
    usage = log_usage(CACHE)
    by = {}
    for s in ("train", "validation"):
        o = [x for x in out if x["split"] == s]
        by[s] = {"questions": len(o), "calls_ok": sum(x["ok"] for x in o),
                 "with_10_answers": sum(x["n_answers"] == REC_NUM for x in o),
                 "raw_lines_eq_10": sum(x["n_lines_raw"] == REC_NUM for x in o),
                 "answers_hist": {str(k): sum(x["n_answers"] == k for x in o) for k in range(REC_NUM + 1)
                                  if any(x["n_answers"] == k for x in o)}}
    stats = {"version": VERSION, "system": SYSTEM, "limit": limit, "by_split": by, "with_10_answers_total": n10,
             "api_usage_cumulative": usage}
    TDIR.mkdir(parents=True, exist_ok=True)
    (TDIR / ("stats_limit.json" if limit else "stats.json")).write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def memo_stats() -> None:
    """Teacher memorization risk on validation (and train, for reference): top-1 vs the board's top cluster, and
    per-respondent recall@10 of the teacher list; within-list duplicates under the matcher."""
    from lib.r9_rewards import cluster_hit_matrix, greedy_hits, matches  # needs nltk
    lists = {}
    for line in LISTS.open():
        r = json.loads(line)
        lists[r["qid"]] = r
    res = {}
    for split in ("validation", "train"):
        top1, rec_g, rec_a, dup_lists, dup_pairs, n = 0, [], [], 0, 0, 0
        for q in load_split(split):
            t = lists.get(q["qid"])
            if t is None:
                continue
            a = t["answers"][:REC_NUM]
            n += 1
            top = max(range(len(q["counts"])), key=lambda c: q["counts"][c])  # first among ties (board order)
            top1 += bool(a) and matches(a[0], q["clusters"][top])
            g = greedy_hits(a, q["clusters"])
            rec_g.append(sum(q["counts"][c] for c in g if c >= 0) / q["n"])
            H = cluster_hit_matrix(a, q["clusters"])
            rec_a.append(sum(q["counts"][c] for c in range(len(q["counts"])) if H[c].any()) / q["n"])
            d = sum(1 for i in range(len(a)) for j in range(i) if matches(a[i], [a[j]]))
            dup_pairs += d
            dup_lists += d > 0
        res[split] = {"questions": n, "top1_matches_top_cluster": top1, "top1_share": round(top1 / max(n, 1), 4),
                      "resp_recall@10_greedy": round(statistics.mean(rec_g), 4) if rec_g else None,
                      "resp_recall@10_anymatch": round(statistics.mean(rec_a), 4) if rec_a else None,
                      "lists_with_matcher_duplicates": dup_lists, "duplicate_pairs": dup_pairs}
    stats_path = TDIR / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    stats["memorization"] = {"note": "GPT-5.6 teacher lists scored against the scraped boards with the ProtoQA matcher; "
                                     "this is the teacher (question-only prompt), not our model",
                             **res}
    stats_path.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats["memorization"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N calls in the shuffled order (no lists file)")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--stats_only", action="store_true", help="memorization / duplicate stats from teacher_lists.jsonl")
    a = ap.parse_args()
    TDIR.mkdir(parents=True, exist_ok=True)
    if a.stats_only:
        memo_stats()
    else:
        run_api(a.limit, a.workers)


if __name__ == "__main__":
    main()
