"""Judge robustness on Reddit-v2 test (API only, existing test lists; protocol registered before the run).

(a) `pair`: second-family pairwise judge. Exactly the protocol of n927_e7b_judge.py (same 300 requests, same prompt
    configs/judge_prompt.txt, both orders, win 1 / tie 0.5 / loss 0 averaged over the orders, request bootstrap 2,000,
    seed 42), with another judge model. Adds Cohen's kappa against the GPT-5.6 verdicts read from the judge caches: per call
    (labels ReMaRk / Rank-GRPO / tie) and per request (winner of the order-averaged score).
(b) `prec`: judged precision@20. 500 requests (unique test contexts, default_rng(42)), one call per (request, arm) that
    judges each of the list's 20 items yes/no against the request (configs/judge_precision_prompt.txt). precision@20 =
    yes / 20 (missing or unparsable items count as no). Paired difference over requests, bootstrap 2,000, seed 42. Strata by
    the training popularity of each recommended item with the Table 9 cutoffs (n928_r7a: q50 / q80 over test GT
    occurrences; tail <= q50 < torso <= q80 < head; items absent from the training GT = 0). The analysis reads every judge
    cache present for the pair and adds item-level kappa between judges.
Usage: env/bin/python remark/scripts/n932_e2_judges.py pair --pair {9B2k,4Bplat} --judge gemini-3.1-pro-preview
       env/bin/python remark/scripts/n932_e2_judges.py prec --pair {9B2k,4Bplat} --judge {gemini-3.1-pro-preview,gpt}
       (add --no_calls to only re-analyse the caches)
Outputs: outputs/crs/analysis/n932_e2a_<pair>_<judge>.json, n932_e2b_<pair>.json; caches outputs/crs/api_cache/n932_e2*.jsonl
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
from n927_e7b_judge import conversation, fmt, verdict  # noqa: E402
from lib.llm import batch_chat, default_model  # noqa: E402

C = REPO / "remark/outputs/crs"
T = C / "test"
# pair -> (ReMaRk test dir, Rank-GRPO test dir, pair GPT-5.6 cache, pair ids tagged with the run names)
PAIRS = {
    "9B2k": ("test_9B_union_group_s3407_p0c_step2000", "test_9B_single_group_s3407_step2000",
             "n927_e7b_judge.jsonl", False),
    "4Bplat": ("test_4B_union_group_s3407_p0c_ms8000_step3000", "test_4B_single_group_s3407_p0c_ms8000_step5250",
               "n927_e7b_judge_g6_4B_plateau.jsonl", True),
}
N_BOOT, SEED = 2000, 42
BUCKETS = ("tail", "torso", "head")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def load_cache(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in r:
                out[r["id"]] = r
    return out


def kappa(x: list, y: list) -> float:
    labs = sorted(set(x) | set(y))
    x, y = np.array(x), np.array(y)
    po = float(np.mean(x == y))
    pe = float(sum(np.mean(x == l) * np.mean(y == l) for l in labs))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def boot_ci(v: np.ndarray) -> list[float]:
    idx = np.random.default_rng(SEED).integers(0, len(v), size=(N_BOOT, len(v)))
    return [float(x) for x in np.percentile(v[idx].mean(1), [2.5, 97.5])]


def setup(pair: str):
    ds = load_from_disk(str(DATA / "test"))
    first_row = OrderedDict()
    for i, p in enumerate(ds["prompt"]):
        first_row.setdefault(p[0]["content"], i)
    ctxs = list(first_row)
    a_run, b_run = PAIRS[pair][:2]
    lists = {}
    for run in (a_run, b_run):
        rows = [json.loads(l) for l in open(T / run / "per_instance.jsonl")]
        lists[run] = {c: rows[first_row[c]]["recs"] for c in ctxs}
    return ds, ctxs, lists, a_run, b_run


def judge_model(j: str) -> str:
    return default_model() if j == "gpt" else j


def git() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()


# ---------------------------------------------------------------- (a) pairwise
def run_pair(a) -> None:
    ds, ctxs, lists, a_run, b_run = setup(a.pair)
    gpt_cache, tagged = PAIRS[a.pair][2:]
    template = (REPO / "remark/configs/judge_prompt.txt").read_text()
    pick = sorted(np.random.default_rng(42).choice(len(ctxs), size=300, replace=False).tolist())
    model = judge_model(a.judge)
    reqs, gpt_id = [], {}
    for j in pick:
        c = ctxs[j]
        for order in ("RB", "BR"):
            la, lb = (lists[a_run][c], lists[b_run][c]) if order == "RB" else (lists[b_run][c], lists[a_run][c])
            msg = template.format(request=conversation(c), list_a=fmt(la), list_b=fmt(lb))
            rid = sha(f"{model}|{a_run}|{b_run}|{c}{order}")
            gpt_id[rid] = sha((a_run + "|" + b_run + "|" + c + order) if tagged else (c + order))
            reqs.append({"id": rid, "model": model, "messages": [{"role": "user", "content": msg}],
                         "max_tokens": a.max_tokens, "meta": {"ctx_j": j, "order": order}})
    cache = C / f"api_cache/n932_e2a_{a.pair}_{model}.jsonl"
    done = load_cache(cache) if a.no_calls else batch_chat(reqs, cache_path=cache, workers=a.workers)
    gpt = load_cache(C / "api_cache" / gpt_cache)

    def score_of(v: str, order: str) -> float:
        return 0.5 if v == "tie" else float((v == "A") == (order == "RB"))

    def label(s: float) -> str:
        return {1.0: "remark", 0.0: "rankgrpo", 0.5: "tie"}[s]

    per, per_gpt, unparsed, a_rate, call_pairs = {}, {}, 0, [], []
    for r in reqs:
        j, order = r["meta"]["ctx_j"], r["meta"]["order"]
        g = gpt.get(gpt_id[r["id"]])
        vg = verdict(g["text"]) if g else None
        if vg is not None:
            per_gpt.setdefault(j, {})[order] = score_of(vg, order)
        if r["id"] not in done:
            continue
        v = verdict(done[r["id"]]["text"])
        if v is None:
            unparsed += 1
            continue
        a_rate.append(v == "A")
        per.setdefault(j, {})[order] = score_of(v, order)
        if vg is not None:
            call_pairs.append((label(score_of(v, order)), label(score_of(vg, order))))
    both = {j: d for j, d in per.items() if len(d) == 2}
    scores = np.array([(d["RB"] + d["BR"]) / 2 for d in both.values()])
    win = lambda s: "remark" if s > 0.5 else ("rankgrpo" if s < 0.5 else "tie")  # noqa: E731
    common = [j for j in both if len(per_gpt.get(j, {})) == 2]
    req_pairs = [(win((both[j]["RB"] + both[j]["BR"]) / 2), win((per_gpt[j]["RB"] + per_gpt[j]["BR"]) / 2)) for j in common]
    gpt_scores = np.array([(per_gpt[j]["RB"] + per_gpt[j]["BR"]) / 2 for j in common])
    usage = {k: sum((done[r["id"]].get("usage") or {}).get(k, 0) or 0 for r in reqs if r["id"] in done)
             for k in ("prompt_tokens", "completion_tokens", "thoughts_tokens")}
    versions = sorted({(done[r["id"]].get("usage") or {}).get("model_version") for r in reqs if r["id"] in done} - {None})
    res = {
        "judge": model, "model_versions": versions, "prompt_file": "remark/configs/judge_prompt.txt",
        "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(), "pair": a.pair, "a_run": a_run, "b_run": b_run,
        "requests_sampled": 300, "requests_scored_both_orders": len(both), "unparsed_answers": unparsed,
        "remark_score": float(scores.mean()), "ci95": boot_ci(scores),
        "order_consistency": float(np.mean([d["RB"] == d["BR"] for d in both.values()])),
        "A_rate": float(np.mean(a_rate)),
        "remark_wins_both": int(sum(d["RB"] == 1 and d["BR"] == 1 for d in both.values())),
        "rankgrpo_wins_both": int(sum(d["RB"] == 0 and d["BR"] == 0 for d in both.values())),
        "tie_both": int(sum(d["RB"] == 0.5 and d["BR"] == 0.5 for d in both.values())),
        "vs_gpt": {"gpt_cache": PAIRS[a.pair][2], "calls_both_parsed": len(call_pairs),
                   "kappa_per_call": kappa(*zip(*call_pairs)) if call_pairs else None,
                   "agreement_per_call": float(np.mean([x == y for x, y in call_pairs])) if call_pairs else None,
                   "requests_both_judges": len(common),
                   "kappa_per_request_winner": kappa(*zip(*req_pairs)) if req_pairs else None,
                   "gpt_remark_score_same_requests": float(gpt_scores.mean()) if len(common) else None,
                   "gpt_ci95_same_requests": boot_ci(gpt_scores) if len(common) else None},
        "api_calls": len(done), "usage": usage, "git": git()}
    out = C / f"analysis/n932_e2a_{a.pair}_{model}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


# ---------------------------------------------------------------- (b) precision@20
LINE = re.compile(r"^\s*\**\s*(\d+)\s*\**\s*[:.)\-]\s*\**\s*(yes|no)\b", re.I)


def parse_items(text: str, n: int) -> tuple[list[int], int]:
    got = {}
    for line in (text or "").splitlines():
        m = LINE.match(line)
        if m and 1 <= int(m.group(1)) <= n and int(m.group(1)) not in got:
            got[int(m.group(1))] = int(m.group(2).lower() == "yes")
    return [got.get(k, 0) for k in range(1, n + 1)], n - len(got)


def pop_bucket(ds) -> callable:
    pop = {tuple(json.loads(k)): v for k, v in json.load(open(C / "train_gt_popularity.json")).items()}
    popv = np.array([pop.get((g[0], int(g[1])), 0) for gts in ds["groundtruth_with_release_year"] for g in gts])
    q50, q80 = (float(x) for x in np.quantile(popv, [0.5, 0.8]))

    def bucket(t, y) -> str:
        try:
            p = pop.get((t, int(y)), 0)
        except (TypeError, ValueError):
            p = 0
        return "tail" if p <= q50 else ("head" if p > q80 else "torso")
    bucket.cutoffs = (q50, q80)
    return bucket


def prec_requests(ctxs, lists, runs, pick, model, template, max_tokens):
    reqs = []
    for j in pick:
        c = ctxs[j]
        for run in runs:
            recs = lists[run][c]
            if not recs:
                continue
            msg = template.format(request=conversation(c), items=fmt(recs))
            reqs.append({"id": sha(f"prec|{model}|{run}|{c}"), "model": model, "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": msg}], "meta": {"ctx_j": j, "run": run}})
    return reqs


def run_prec(a) -> None:
    ds, ctxs, lists, a_run, b_run = setup(a.pair)
    template = (REPO / "remark/configs/judge_precision_prompt.txt").read_text()
    pick = sorted(np.random.default_rng(42).choice(len(ctxs), size=500, replace=False).tolist())
    if a.judge and not a.no_calls:
        model = judge_model(a.judge)
        reqs = prec_requests(ctxs, lists, (a_run, b_run), pick, model, template, a.max_tokens)
        batch_chat(reqs, cache_path=C / f"api_cache/n932_e2b_{a.pair}_{model}.jsonl", workers=a.workers)
    bucket = pop_bucket(ds)
    judges = {}
    for model in ("gemini-3.1-pro-preview", default_model()):
        cache = C / f"api_cache/n932_e2b_{a.pair}_{model}.jsonl"
        if not cache.exists():
            continue
        done = load_cache(cache)
        reqs = prec_requests(ctxs, lists, (a_run, b_run), pick, model, template, a.max_tokens)
        labels, missing_calls, missing_items = {}, 0, 0
        for r in reqs:
            if r["id"] not in done:
                missing_calls += 1
                continue
            recs = lists[r["meta"]["run"]][ctxs[r["meta"]["ctx_j"]]]
            lab, miss = parse_items(done[r["id"]]["text"], len(recs))
            missing_items += miss
            labels[(r["meta"]["ctx_j"], r["meta"]["run"])] = lab
        # requests where both arms are judged (an empty list is judged as all-no without a call)
        keep = [j for j in pick if all((j, run) in labels or not lists[run][ctxs[j]] for run in (a_run, b_run))]
        lab = lambda j, run: labels.get((j, run), [])  # noqa: E731
        P = {run: np.array([sum(lab(j, run)) / 20 for j in keep]) for run in (a_run, b_run)}
        d = P[a_run] - P[b_run]
        idx = np.random.default_rng(SEED).integers(0, len(keep), size=(N_BOOT, len(keep)))
        strata = {}
        for b in BUCKETS:
            yes = {run: np.array([sum(l for l, (t, y) in zip(lab(j, run), lists[run][ctxs[j]]) if bucket(t, y) == b)
                                  for j in keep], float) for run in (a_run, b_run)}
            n = {run: np.array([sum(1 for t, y in lists[run][ctxs[j]][:len(lab(j, run))] if bucket(t, y) == b)
                                for j in keep], float) for run in (a_run, b_run)}
            pa = lambda run, ix=slice(None): yes[run][ix].sum(-1) / np.maximum(n[run][ix].sum(-1), 1)  # noqa: E731
            bd = pa(a_run, idx) - pa(b_run, idx)
            strata[b] = {"remark_precision": float(pa(a_run)), "rankgrpo_precision": float(pa(b_run)),
                         "diff": float(pa(a_run) - pa(b_run)), "ci95": [float(x) for x in np.percentile(bd, [2.5, 97.5])],
                         "remark_items": int(n[a_run].sum()), "rankgrpo_items": int(n[b_run].sum()),
                         "remark_share": float(n[a_run].sum() / max(1, 20 * len(keep))),
                         "rankgrpo_share": float(n[b_run].sum() / max(1, 20 * len(keep)))}
        usage = {k: sum((done[r["id"]].get("usage") or {}).get(k, 0) or 0 for r in reqs if r["id"] in done)
                 for k in ("prompt_tokens", "completion_tokens", "thoughts_tokens")}
        judges[model] = {"requests": len(keep), "missing_calls": missing_calls, "unparsed_items": missing_items,
                         "remark_p20": float(P[a_run].mean()), "rankgrpo_p20": float(P[b_run].mean()),
                         "diff": float(d.mean()), "ci95": boot_ci(d), "strata": strata, "usage": usage,
                         "_labels": labels}
    agree = None
    if len(judges) == 2:
        (m1, j1), (m2, j2) = judges.items()
        x, y = [], []
        for k in set(j1["_labels"]) & set(j2["_labels"]):
            x += j1["_labels"][k]
            y += j2["_labels"][k]
        agree = {"judges": [m1, m2], "items": len(x), "kappa_item": kappa(x, y),
                 "agreement_item": float(np.mean(np.array(x) == np.array(y))),
                 "yes_rate": [float(np.mean(x)), float(np.mean(y))]}
    for v in judges.values():
        v.pop("_labels")
    res = {"pair": a.pair, "a_run": a_run, "b_run": b_run, "requests_sampled": 500,
           "prompt_file": "remark/configs/judge_precision_prompt.txt",
           "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
           "pop_cutoffs_q50_q80": list(bucket.cutoffs), "judges": judges, "judge_agreement": agree, "git": git()}
    out = C / f"analysis/n932_e2b_{a.pair}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pair", "prec"])
    ap.add_argument("--pair", choices=list(PAIRS), required=True)
    ap.add_argument("--judge", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--no_calls", action="store_true")
    a = ap.parse_args()
    (run_pair if a.cmd == "pair" else run_prec)(a)


if __name__ == "__main__":
    main()
