"""Evaluate a conversational recommender on Reddit-v2 (Rank-GRPO release) with vLLM.

Protocol (Rank-GRPO, ICLR 2026): the model generates 20 "Title (Year)" lines; a recommendation hits a
ground-truth item if the title matches exactly and |year diff| <= 2. Two scoring variants:
  truncate: positions 1..k of the parsed generation; seen or out-of-catalog items count as misses
  filter:   drop seen / out-of-catalog items first, then take top-k (Rank-GRPO libs/metrics.py)
Also reports item-level recall@20 by ground-truth popularity (train-split GT frequency) for the gate.

Usage (GPU node): /tmp/env/bin/python remark/scripts/crs_eval.py --model <ckpt> --name <tag> [--split test]
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

REPO = Path(__file__).resolve().parents[2]
RG = REPO / "remark/third_party/Rank-GRPO"
sys.path.insert(0, str(RG / "libs"))
from utils import process_rec_raw  # noqa: E402  (Rank-GRPO's title parser)

DATA = REPO / "data/raw/reddit_v2/processed_datasets/sft_dataset"
KS = (5, 10, 20)


def prepare_qwen35(model: Path) -> dict:
    """Qwen3.5 checkpoints keep the VLM layout; vLLM needs the base model's processor configs next to them."""
    cfg = json.loads((model / "config.json").read_text())
    if not any(x.startswith("Qwen3_5") for x in cfg.get("architectures") or []):
        return {}
    tc = cfg["text_config"]
    for f in ("preprocessor_config.json", "video_preprocessor_config.json"):
        if (model / f).exists():
            continue
        # local base models, then the 27B/35B-A3B SFT checkpoints (their bases are only staged on the nodes)
        bases = sorted((REPO / "data/models").glob("Qwen3.5-*-Base")) + sorted(
            c for c in (REPO / "remark/outputs/crs/sft").glob("qwen35_*/checkpoint-*") if (c / f).exists())
        for base in bases:
            bc = json.loads((base / "config.json").read_text())["text_config"]
            if (bc["hidden_size"], bc["num_hidden_layers"]) == (tc["hidden_size"], tc["num_hidden_layers"]):
                shutil.copy(base / f, model / f)
                break
        else:
            raise FileNotFoundError(f"no Qwen3.5 base model matches {model}")
    return {"limit_mm_per_prompt": {"image": 0, "video": 0}}


def hits_for(recs, gts, seen, catalog, mode):
    if mode == "filter":
        recs = [r for r in recs if r[0] not in seen and r in catalog]
    hits, matched_gt = np.zeros(len(recs), int), set()
    for gi, gt in enumerate(gts):
        for i, r in enumerate(recs):
            if mode == "truncate" and (r[0] in seen or r not in catalog):
                continue
            if hits[i] == 0 and r[0] == gt[0] and abs(int(gt[1]) - int(r[1])) <= 2:
                hits[i] = 1
                matched_gt.add((gi, i))
                break
    return hits, matched_gt


def recall_ndcg(hits, n_gt, k):
    h = hits[:k]
    if len(h) == 0 or n_gt == 0:
        return 0.0, 0.0
    dcg = sum(x / np.log2(i + 2) for i, x in enumerate(h))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(n_gt, k)))
    return float(h.sum() / n_gt), float(dcg / idcg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--out", default=str(REPO / "remark/outputs/crs"))
    ap.add_argument("--data", default=str(DATA), help="sft_dataset dir with train/validation/test (default Reddit-v2)")
    ap.add_argument("--catalog", default=str(RG / "reddit_catalog/gt_catalog.pkl"))
    a = ap.parse_args()
    # every node's lock-and-test watcher tests any locked run it finds; older watchers only pass --data/--catalog
    # for reddit_books_* runs. A run name carries its dataset (runner naming: <dataset>_<size>_...), so when the caller
    # left the Reddit-v2 defaults, take the dataset from the name rather than evaluate on the wrong test set.
    if a.data == str(DATA) and a.catalog == str(RG / "reddit_catalog/gt_catalog.pkl"):
        for ds in ("music", "reddit_books"):
            if f"_{ds}_" in f"_{a.name}" or a.name.startswith(f"{ds}_") or f"test_{ds}_" in a.name or f"qwen_{ds}_" in a.name:
                a.data = str(REPO / f"remark/data/crs/{ds}/sft_dataset")
                a.catalog = str(REPO / f"remark/data/crs/{ds}/catalog.pkl")
                print(f"[crs_eval] dataset taken from the run name: {ds}", flush=True)
                break
    data = Path(a.data)
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(str(data / a.split))
    if a.limit:
        ds = ds.select(range(min(a.limit, len(ds))))
    catalog = set(tuple(x) for x in pickle.load(open(a.catalog, "rb")))

    ctx_idx = defaultdict(list)
    for i, it in enumerate(ds):
        ctx_idx[it["prompt"][0]["content"]].append(i)
    ctxs = list(ctx_idx)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util, max_model_len=8192,
              **prepare_qwen35(Path(a.model)))
    outs = llm.chat([[{"role": "user", "content": c}] for c in ctxs],
                    SamplingParams(temperature=0, max_tokens=a.max_tokens))
    raw = {c: o.outputs[0].text for c, o in zip(ctxs, outs)}

    # train-split GT popularity (for head/tail analysis)
    tag = "" if data == DATA else "_" + data.parent.name  # e.g. _redial for remark/data/crs/redial/sft_dataset
    pop_path = REPO / f"remark/outputs/crs/train_gt_popularity{tag}.json"
    if pop_path.exists():
        pop = Counter({tuple(json.loads(k)): v for k, v in json.load(open(pop_path)).items()})
    else:
        tr = load_from_disk(str(data / "train"))
        pop = Counter()
        for gts in tr["groundtruth_with_release_year"]:
            for g in gts:
                pop[(g[0], int(g[1]))] += 1
        pop_path.write_text(json.dumps({json.dumps([k[0], k[1]]): v for k, v in pop.items()}))

    res = {m: {k: [[], []] for k in KS} for m in ("truncate", "filter")}
    item_level = []  # (popularity, hit@20 truncate)
    n_parse_err, n_in_cat, n_recs = 0, 0, 0
    per = []
    for i, it in enumerate(ds):
        item = {"raw": raw[it["prompt"][0]["content"]]}
        err_item = process_rec_raw(item, "raw", "rec")
        recs = [(t, int(y)) for t, y in err_item[1]["rec"]] if isinstance(err_item, tuple) else [(t, int(y)) for t, y in item["rec"]]
        recs = recs[:20]
        n_recs += len(recs)
        n_in_cat += sum(r in catalog for r in recs)
        if not recs:
            n_parse_err += 1
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        seen = set(it["seen_titles"])
        for m in ("truncate", "filter"):
            hits, matched = hits_for(recs, gts, seen, catalog, m)
            for k in KS:
                r, n = recall_ndcg(hits, len(gts), k)
                res[m][k][0].append(r)
                res[m][k][1].append(n)
            if m == "truncate":
                hit_gts = {gi for gi, pos in matched if pos < 20}
                for gi, g in enumerate(gts):
                    item_level.append((pop.get(g, 0), int(gi in hit_gts)))
        per.append({"i": i, "recs": recs, "gts": gts})

    summary = {"model": a.model, "data": str(data), "split": a.split, "n": len(ds), "unique_contexts": len(ctxs),
               "parse_empty": n_parse_err, "in_catalog_rate": n_in_cat / max(1, n_recs)}
    for m in res:
        for k in KS:
            summary[f"{m}_R@{k}"] = float(np.mean(res[m][k][0]))
            summary[f"{m}_N@{k}"] = float(np.mean(res[m][k][1]))
    popv = np.array([p for p, _ in item_level]); hitv = np.array([h for _, h in item_level])
    qs = np.quantile(popv, [0.5, 0.8, 0.9])
    summary["gt_pop_quantiles_50_80_90"] = qs.tolist()
    buckets = {"unseen(0)": popv == 0, "tail(<=median)": popv <= qs[0], "mid": (popv > qs[0]) & (popv <= qs[1]),
               "head(top20%)": popv > qs[1], "top10%": popv > qs[2]}
    summary["item_recall@20_by_pop"] = {b: {"n": int(msk.sum()), "recall": float(hitv[msk].mean()) if msk.any() else None}
                                        for b, msk in buckets.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "per_instance.jsonl", "w") as f:
        for r in per:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
