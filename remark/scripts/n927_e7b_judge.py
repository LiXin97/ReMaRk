"""Blind pairwise LLM judge on the FINAL configuration, ReMaRk (union + group) vs Rank-GRPO (Qwen3.5-9B)
on Reddit-v2 test. Same 300 requests, same prompt file and the same both-orders protocol as the first judge run, which judged the
batch-scaled ReMaRk arm; only the A run changes.

300 test requests drawn uniformly over requests (seed 42). The judge (GPT-5.6, prompt fixed in configs/judge_prompt.txt
before the first call) sees the request text and the two generated lists labelled A and B, never the responders' items.
Each request is judged in both orders. Score for ReMaRk per order: win 1, tie 0.5, loss 0; averaged over the two orders,
then over requests; 95% CI by bootstrap over requests (2,000, seed 42). Also: order consistency and the A-rate.
Usage: env/bin/python remark/scripts/n927_e7b_judge.py [--workers 16]
       [--a_run <ReMaRk test dir> --b_run <baseline test dir> --tag <name>]   (other pairs; the tag gives a
       separate cache and output file, and the run names enter the request ids, so no verdict is reused across pairs)
Output: outputs/crs/analysis/n927_e7b_judge.json; API cache outputs/crs/api_cache/n927_e7b_judge.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n917_common import DATA, REPO  # noqa: E402
from lib.llm import batch_chat, default_model  # noqa: E402

T = REPO / "remark/outputs/crs/test"
A_RUN, B_RUN = "test_9B_union_group_s3407_p0c_step2000", "test_9B_single_group_s3407_step2000"  # ReMaRk (final), Rank-GRPO


def conversation(p: str) -> str:
    return p.split("Here is the conversation:", 1)[-1].strip()


def fmt(recs) -> str:
    return "\n".join(f"{i + 1}. {t} ({y})" for i, (t, y) in enumerate(recs)) or "(empty list)"


def verdict(text: str) -> str | None:
    first = (text or "").strip().splitlines()[0].strip().lower() if (text or "").strip() else ""
    first = re.sub(r"[^a-z]", "", first)
    return {"a": "A", "b": "B", "tie": "tie"}.get(first)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--a_run", default=A_RUN)
    ap.add_argument("--b_run", default=B_RUN)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a_run, b_run = a.a_run, a.b_run
    sfx = f"_{a.tag}" if a.tag else ""
    template = (REPO / "remark/configs/judge_prompt.txt").read_text()
    ds = load_from_disk(str(DATA / "test"))
    first_row = OrderedDict()
    for i, p in enumerate(ds["prompt"]):
        first_row.setdefault(p[0]["content"], i)
    ctxs = list(first_row)
    pick = sorted(np.random.default_rng(42).choice(len(ctxs), size=a.n, replace=False).tolist())
    lists = {}
    for run in (a_run, b_run):
        rows = [json.loads(l) for l in open(T / run / "per_instance.jsonl")]
        lists[run] = {c: rows[first_row[c]]["recs"] for c in ctxs}
    reqs = []
    for j in pick:
        c = ctxs[j]
        for order in ("RB", "BR"):  # R = ReMaRk shown as A; B = Rank-GRPO shown as A
            la, lb = (lists[a_run][c], lists[b_run][c]) if order == "RB" else (lists[b_run][c], lists[a_run][c])
            msg = template.format(request=conversation(c), list_a=fmt(la), list_b=fmt(lb))
            key = (c + order) if not a.tag else (a_run + "|" + b_run + "|" + c + order)
            reqs.append({"id": hashlib.sha256(key.encode()).hexdigest()[:20], "messages": [{"role": "user", "content": msg}],
                         "max_tokens": 1024, "meta": {"ctx_j": j, "order": order}})
    cache = REPO / f"remark/outputs/crs/api_cache/n927_e7b_judge{sfx}.jsonl"
    done = batch_chat(reqs, cache_path=cache, workers=a.workers)
    per = {}
    n_unparsed, a_rate = 0, []
    for r in reqs:
        if r["id"] not in done:
            continue
        v = verdict(done[r["id"]]["text"])
        if v is None:
            n_unparsed += 1
            continue
        j, order = r["meta"]["ctx_j"], r["meta"]["order"]
        a_rate.append(v == "A")
        remark_is_a = order == "RB"
        s = 0.5 if v == "tie" else float((v == "A") == remark_is_a)
        per.setdefault(j, {})[order] = (s, v)
    both = {j: d for j, d in per.items() if len(d) == 2}
    scores = np.array([(d["RB"][0] + d["BR"][0]) / 2 for d in both.values()])
    consistent = np.mean([d["RB"][0] == d["BR"][0] for d in both.values()])
    boot = scores[np.random.default_rng(42).integers(0, len(scores), size=(2000, len(scores)))].mean(1)
    outcomes = {"remark_wins_both": int(sum(d["RB"][0] == 1 and d["BR"][0] == 1 for d in both.values())),
                "rankgrpo_wins_both": int(sum(d["RB"][0] == 0 and d["BR"][0] == 0 for d in both.values())),
                "tie_both": int(sum(d["RB"][1] == "tie" and d["BR"][1] == "tie" for d in both.values()))}
    usage = {k: sum((done[r["id"]].get("usage") or {}).get(k, 0) or 0 for r in reqs if r["id"] in done)
             for k in ("prompt_tokens", "completion_tokens")}
    res = {"judge": default_model(), "prompt_file": "remark/configs/judge_prompt.txt",
           "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(), "a_run": a_run, "b_run": b_run,
           "requests_sampled": a.n, "requests_scored_both_orders": len(both), "unparsed_answers": n_unparsed,
           "remark_score": float(scores.mean()), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
           "order_consistency": float(consistent), "A_rate": float(np.mean(a_rate)), **outcomes,
           "api_calls": len(done), "usage": usage,
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()}
    (REPO / f"remark/outputs/crs/analysis/n927_e7b_judge{sfx}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
