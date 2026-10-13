"""Distinct training requests and rows seen in 2,000 steps by each arm of the 2B ablation table, from a
replay of TRL 1.14's GRPO sampler. CPU only.

Inputs
  arms:      make_ablation_tables.ARMS rows 1-7 (seed 3407); training sets as mapped by the launcher (launch5.sh):
             single_group / single_batch: data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train (383,013 rows);
             singlethr: remark/data/crs/redditthr/grpo/train; consensusrw: remark/data/crs/grpo_consensus_rw/train;
             consensus: remark/data/crs/grpo_consensus/train; union / union_group: remark/data/crs/grpo_union/train.
  tokenizer: the 2B SFT checkpoint the runs were launched from (--sft_path, remark/outputs/crs/sft/qwen35_2B/
             checkpoint-1500; its chat template equals data/models/Qwen3.5-2B-Base's, checked here).
  logs:      remark/outputs/crs/train_logs/... (the logged 'epoch' of every record checks the filtered size).
Definitions
  Filter (crs_train_grpo_qwen.py): keep a row iff len(tok(tok.apply_chat_template(prompt, tokenize=False,
  add_generation_prompt=True))["input_ids"]) <= 2048; datasets.filter keeps the row order. Every prompt is one user
  message, so the decision is computed once per request (prompt text) and broadcast to its rows.
  Sampler (TRL 1.14 GRPOTrainer._get_train_sampler -> trl.trainer.utils.RepeatSampler): one torch.Generator seeded with
  args.seed (3407) at construction; each pass over the dataloader (one per epoch) calls torch.randperm(N, generator)
  with the generator state carried over, cuts the permutation into chunks of generation_batch_size / num_generations
  = (16 x 4 x 6) / 8 = 48 prompts and drops the incomplete last chunk; each chunk is repeated num_generations (8) times
  per index and repeat_count = steps_per_generation (6) times, i.e. one chunk = one optimizer step. Steps per epoch =
  N // 48; step s uses chunk s mod (N // 48) of epoch s // (N // 48). Accelerate shards the batches across the 4
  processes and broadcasts rank 0's generator state (identical on every rank), so the global sequence is the sampler's.
  Assumptions: CPU torch.randperm is the same algorithm in the CPU host's torch (this replay) and the node's torch
  2.10 (the CPU path for n < 2^32 / 20 is the long-standing Fisher-Yates loop); one __iter__ per epoch; no resume in
  these runs (none of their logs has a '[resume]' line; consensus / consensusrw were rerun from scratch with the same
  seed, which replays the same sequence).
  Request = distinct prompt text (rows sharing a prompt); n_x = rows of the request in grpo_dataset/train.
  For comparison, the expectation under uniform shuffling without replacement: a request with n_x rows among N
  filtered rows is seen with probability 1 - C(N - n_x, D) / C(N, D), D = 96,000 draws (if D <= N).
Output: remark/outputs/crs/analysis/n928_r7b_sampler_coverage.json
Usage:  env/bin/python remark/scripts/n928_r7b_sampler_coverage.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import REPO  # noqa: E402
from make_ablation_tables import ARMS  # noqa: E402

OUT = REPO / "remark/outputs/crs/analysis/n928_r7b_sampler_coverage.json"
LOGS = REPO / "remark/outputs/crs/train_logs"
SFT = REPO / "remark/outputs/crs/sft/qwen35_2B/checkpoint-1500"
BASE = REPO / "data/models/Qwen3.5-2B-Base"
ROWS = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"
D = REPO / "remark/data/crs"
DATA = {"single_group": ROWS, "single_batch": ROWS, "singlethr": D / "redditthr/grpo/train",
        "consensusrw": D / "grpo_consensus_rw/train", "consensus": D / "grpo_consensus/train",
        "union": D / "grpo_union/train", "union_group": D / "grpo_union/train"}
LOG = {"single_group": "run_logs/grpo_2B_single_group_s3407_p0c.log",
       "single_batch": "run_logs/grpo_2B_single_batch_s3407_p0c.log",
       "singlethr": "run_logs/grpo_2B_singlethr_batch_s3407_p0c.log",
       "consensusrw": "run_logs/grpo_2B_consensusrw_batch_s3407_p0c_cap320.log",
       "consensus": "run_logs/grpo_2B_consensus_batch_s3407_p0c_cap320.log",
       "union": "run_logs/grpo_2B_union_batch_s3407_p0c.log",
       "union_group": "run_logs/grpo_2B_union_group_s3407_p0c.log"}
SEED, STEPS, CHUNK, MAX_PROMPT = 3407, 2000, 48, 2048
NBUCKETS = {"1": (1, 1), "2-4": (2, 4), "5-9": (5, 9), ">=10": (10, 10**9)}


def replay(n: int, seed: int = SEED, steps: int = STEPS, chunk: int = CHUNK) -> np.ndarray:
    """Indices (into the filtered dataset) of the prompts used in steps 0..steps-1, shape (steps, chunk)."""
    g = torch.Generator()
    g.manual_seed(seed)
    per_epoch = n // chunk
    out = []
    while len(out) < steps:
        perm = torch.randperm(n, generator=g).tolist()
        for c in range(per_epoch):
            out.append(perm[c * chunk:(c + 1) * chunk])
            if len(out) == steps:
                break
    return np.array(out)


def logged_epochs(path: Path) -> list[float]:
    txt = open(path, errors="replace").read()
    return [float(m.group(1)) for rec in re.findall(r"\{'loss': .*?\}", txt)
            for m in [re.search(r"'epoch': '?([0-9.e-]+)", rec)] if m]


def sig4(x: float) -> float:
    return float(f"{x:.4g}")


def p_seen_hypergeom(n_x: np.ndarray, n: int, d: int) -> np.ndarray:
    """P(at least one of a request's n_x rows among d draws without replacement from n rows)."""
    if d > n:
        return np.ones(len(n_x))
    # C(n - k, d) / C(n, d) = prod_{i=0}^{k-1} (n - d - i) / (n - i)
    out = np.empty(len(n_x))
    cache: dict[int, float] = {}
    for j, k in enumerate(n_x):
        if k not in cache:
            i = np.arange(k)
            cache[k] = 1.0 - float(np.exp(np.sum(np.log(np.maximum(n - d - i, 1e-300)) - np.log(n - i))))
        out[j] = cache[k]
    return out


def main() -> None:
    tok = AutoTokenizer.from_pretrained(SFT)
    base_tmpl = AutoTokenizer.from_pretrained(BASE).chat_template
    union = load_from_disk(str(D / "grpo_union/train")).select_columns(["prompt"])
    prompts = union["prompt"]
    assert all(len(p) == 1 and p[0]["role"] == "user" for p in prompts)
    index = {p[0]["content"]: i for i, p in enumerate(prompts)}
    n_req = len(prompts)
    texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    lens = np.array([len(x) for x in tok(texts)["input_ids"]])
    keep_req = lens <= MAX_PROMPT
    del texts
    res = {"definition": __doc__.split("Definitions", 1)[1].split("Output:", 1)[0].strip(),
           "tokenizer": str(SFT.relative_to(REPO)), "chat_template_equals_base": tok.chat_template == base_tmpl,
           "seed": SEED, "steps": STEPS, "prompts_per_step": CHUNK, "lists_per_step": CHUNK * 8,
           "n_requests": n_req, "requests_over_2048_tokens": int((~keep_req).sum()),
           "prompt_tokens": {"max": int(lens.max()), "p99": float(np.percentile(lens, 99)), "mean": float(lens.mean())},
           "arms": {}}
    try:
        res["git"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()
    except OSError:
        pass
    req_cache: dict[str, np.ndarray] = {}
    for row, key, *_ in ARMS:
        if row == 0:
            continue
        path = DATA[key]
        if str(path) not in req_cache:
            ds = load_from_disk(str(path)).select_columns(["prompt"])
            req = np.empty(len(ds), np.int64)
            i = 0
            for b in ds.iter(batch_size=20_000):
                for p in b["prompt"]:
                    assert len(p) == 1 and p[0]["role"] == "user"
                    req[i] = index[p[0]["content"]]
                    i += 1
            req_cache[str(path)] = req
        req = req_cache[str(path)]
        keep = keep_req[req]
        kept_rows = np.flatnonzero(keep)  # filtered index -> original row id
        n = len(kept_rows)
        idx = replay(n)
        seen_rows = np.unique(kept_rows[idx.ravel()])
        seen_req = np.unique(req[seen_rows])
        res["arms"][key] = {"table_row": row, "dataset": str(path.relative_to(REPO)), "rows": int(len(req)),
                            "rows_after_filter": int(n), "steps_per_epoch": n // CHUNK,
                            "epochs_touched": int((STEPS - 1) // (n // CHUNK) + 1), "draws": int(idx.size),
                            "distinct_rows_seen": int(len(seen_rows)), "share_rows_seen": len(seen_rows) / len(req),
                            "distinct_requests_seen": int(len(seen_req)), "share_requests_seen": len(seen_req) / n_req,
                            "_seen_req": seen_req, "_n_filtered": n, "_req": req, "_keep": keep}
        print(key, {k: v for k, v in res["arms"][key].items() if not k.startswith("_")}, flush=True)
    n_x = np.bincount(req_cache[str(ROWS)], minlength=n_req)
    for key, e in res["arms"].items():
        seen = np.zeros(n_req, bool)
        seen[e.pop("_seen_req")] = True
        n, req, keep = e.pop("_n_filtered"), e.pop("_req"), e.pop("_keep")
        e["share_n1_requests_seen"] = float(seen[n_x == 1].mean())
        e["share_requests_seen_by_n"] = {b: {"requests": int(((n_x >= lo) & (n_x <= hi)).sum()),
                                             "share_seen": float(seen[(n_x >= lo) & (n_x <= hi)].mean())}
                                         for b, (lo, hi) in NBUCKETS.items()}
        e["share_filtered_requests_seen"] = float(seen[keep_req].mean())
        # expectation under uniform shuffling without replacement (labelled; not the replay)
        rows_per_req = np.bincount(req[keep], minlength=n_req)
        d = STEPS * CHUNK
        p = np.zeros(n_req)  # requests removed by the filter are never seen
        p[keep_req] = p_seen_hypergeom(rows_per_req[keep_req], n, d) if d <= n else 1.0
        e["expectation_uniform_without_replacement"] = {
            "requests_seen": float(p.sum()), "share_requests_seen": float(p.sum() / n_req),
            "share_n1_requests_seen": float(p[n_x == 1].mean()),
            "rows_seen": float(min(d, n)), "note": "expectation, not the replay" if d <= n else
            "D > N: the first epoch draws every filtered row except the dropped tail chunk (< 48 rows)"}
        # filtered size checked against the logged epoch (TRL prints 4 significant digits)
        lp = LOGS / LOG[key]
        if lp.exists():
            ep = logged_epochs(lp)
            cand = {}
            for spe in range(n // CHUNK - 3, n // CHUNK + 4):
                cand[spe] = all(abs(sig4(10 * (j + 1) / spe) - v) <= 1e-12 + 5e-5 * abs(v) for j, v in enumerate(ep))
            e["epoch_check"] = {"log": str(lp.relative_to(REPO)), "records": len(ep),
                                "steps_per_epoch_consistent_with_log": [k for k, ok in cand.items() if ok],
                                "replay_steps_per_epoch": n // CHUNK,
                                "replay_consistent": bool(cand[n // CHUNK])}
        else:
            e["epoch_check"] = {"log": str(lp.relative_to(REPO)), "missing": True}
        print(key, e["share_requests_seen"], e["share_n1_requests_seen"], e["epoch_check"], flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
