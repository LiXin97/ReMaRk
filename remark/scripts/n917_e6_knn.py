"""The bge-kNN similar-request prior as a non-RL baseline on test.

For a test request x: embed its conversation text with bge-base-en-v1.5 (CLS pooling, L2-normalized; same recipe as
crs_neighbor_prior.py --phase embed), take the k nearest TRAINING requests (training split only, so no test labels are
used; neighbours from the same Reddit submission are excluded), and score each item by
mu_i(x) = sum_j max(sim_j, 0) * f_ij / sum_j max(sim_j, 0), where f_ij is the fraction of request j's responders who named
item i. The recommendation list is the top-20 items by mu_i, skipping titles mentioned in the request. Evaluated with
crs_eval's truncate matching; outputs in crs_eval format under outputs/crs/test/test_knn_prior_k<k>[_books].

Runs on CPU (CPU): Reddit-v2 reuses the training embeddings in data/crs/neighbor_prior/emb.npy; Reddit-Books
embeds its 32.5k training requests once (cached to data/crs/reddit_books/knn_emb_train.npy).
Music (Table 2) is handled as Reddit-Books: its 19.2k training requests are embedded once
(data/crs/music/knn_emb_train.npy).
Usage: env/bin/python remark/scripts/n917_e6_knn.py --dataset reddit|books|music [--k 200]
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import DATA, REPO, RG, KS, hits_for, recall_ndcg  # noqa: E402

torch.set_num_threads(6)


def conversation(prompt_text: str) -> str:
    return prompt_text.split("Here is the conversation:", 1)[-1].strip()


def submission_key(prompt_text: str) -> str:
    first = re.split(r"\n\s*(?:SYSTEM|USER):", conversation(prompt_text), maxsplit=1)[0]
    return " ".join(first.split())


def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # GPU node when available (music run); fp32 either way
    model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").eval().to(dev)
    out = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            e = model(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
            if s % (batch * 50) == 0:
                print(f"embedded {s}/{len(texts)}", flush=True)
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["reddit", "books", "music"], required=True)
    ap.add_argument("--k", type=int, default=200)
    a = ap.parse_args()
    if a.dataset == "reddit":
        data, catalog_path, suffix = DATA, RG / "reddit_catalog/gt_catalog.pkl", ""
        train = load_from_disk(str(REPO / "remark/data/crs/grpo_consensus/train"))
        E_tr = np.load(REPO / "remark/data/crs/neighbor_prior/emb.npy").astype(np.float32)
    else:
        B = REPO / ("remark/data/crs/reddit_books" if a.dataset == "books" else "remark/data/crs/music")
        data, catalog_path, suffix = B / "sft_dataset", B / "catalog.pkl", "_" + a.dataset
        train = load_from_disk(str(B / "grpo_consensus/train"))
        cache = B / "knn_emb_train.npy"
        if cache.exists():
            E_tr = np.load(cache)
        else:
            E_tr = embed([conversation(p[0]["content"]) for p in train["prompt"]])
            np.save(cache, E_tr)
    assert len(E_tr) == len(train)
    freq = [dict(((g[0], int(g[1])), float(w)) for g, w in zip(gts, ws))
            for gts, ws in zip(train["groundtruth_with_release_year"], train["groundtruth_weights"])]
    sub_tr = np.array([hash(submission_key(p[0]["content"])) for p in train["prompt"]])
    ds = load_from_disk(str(Path(data) / "test"))
    catalog = set(tuple(x) for x in pickle.load(open(catalog_path, "rb")))
    ctx_idx = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx_idx[p[0]["content"]].append(i)
    ctxs = list(ctx_idx)
    E_te = embed([conversation(c) for c in ctxs])
    lists = {}
    n_excl = 0
    for s in range(0, len(ctxs), 256):
        sims = E_te[s:s + 256] @ E_tr.T
        for r, c in enumerate(ctxs[s:s + 256]):
            row = sims[r]
            same = sub_tr == hash(submission_key(c))
            n_excl += int(same.sum())
            row = np.where(same, -2.0, row)
            nbr = np.argpartition(-row, a.k)[:a.k]
            mu, wsum = Counter(), 0.0
            for j in nbr:
                w = max(float(row[j]), 0.0)
                if w == 0.0:
                    continue
                wsum += w
                for item, f in freq[j].items():
                    mu[item] += w * f
            seen = set(ds[ctx_idx[c][0]]["seen_titles"])
            ranked = [it for it, _ in sorted(mu.items(), key=lambda kv: (-kv[1], kv[0]))]
            lists[c] = [it for it in ranked if it[0] not in seen][:20]
    out = REPO / f"remark/outputs/crs/test/test_knn_prior_k{a.k}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    res = {m: {k: [[], []] for k in KS} for m in ("truncate",)}
    per, n_in_cat, n_recs = [], 0, 0
    for i, it in enumerate(ds):
        recs = lists[it["prompt"][0]["content"]]
        n_recs += len(recs); n_in_cat += sum(r in catalog for r in recs)
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        hits, _ = hits_for(recs, gts, set(it["seen_titles"]), catalog, "truncate")
        for k in KS:
            r_, n_ = recall_ndcg(hits, len(gts), k)
            res["truncate"][k][0].append(r_); res["truncate"][k][1].append(n_)
        per.append({"i": i, "recs": recs, "gts": gts})
    summary = {"model": f"bge-kNN prior k={a.k} (training split only; same-submission neighbours excluded)",
               "data": str(data), "split": "test", "n": len(ds), "unique_contexts": len(ctxs),
               "same_submission_neighbours_excluded": n_excl, "in_catalog_rate": n_in_cat / max(1, n_recs),
               "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()}
    for k in KS:
        summary[f"truncate_R@{k}"] = float(np.mean(res["truncate"][k][0]))
        summary[f"truncate_N@{k}"] = float(np.mean(res["truncate"][k][1]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "per_instance.jsonl", "w") as f:
        for r in per:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
