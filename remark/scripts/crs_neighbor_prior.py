"""Stage 1: neighbour-prior shrinkage of consensus weights, selected by leave-one-responder-out (LORO).

Per training thread x with responders r = 1..n (Reddit-v2 GRPO train split, one row per (thread, responder)):
  c_i   = number of responders recommending item i;   f_i = c_i / n  (plug-in consensus weight)
  mu_i  = similarity-weighted mean of f_i over the k nearest *other* training threads (bge-base-en-v1.5, cosine)
  p~_i  = (c_i + alpha * mu_i) / (n + alpha)            (Beta-Binomial posterior mean per item)
LORO: on threads with n >= 2, hold out one responder, estimate p~ from the other n-1 responders, rank the items by
p~ (ties broken at random, seed 42), and score the top 20 against the held-out responder's items (NDCG@20, Recall@20);
also the Brier score sum_i (p~_i - 1[i in S_r])^2 over all items (items outside the candidates contribute 0).

Phases (node): --phase embed (GPU; writes embeddings + top-200 neighbours), --phase loro (CPU; writes the grid).
Usage: /tmp/qenv/bin/python remark/scripts/crs_neighbor_prior.py --phase embed|loro
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"
OUT = REPO / "remark/data/crs/neighbor_prior"
RES = REPO / "remark/outputs/crs/neighbor_prior"
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
KS = (10, 50, 200)
TOP_PRIOR = 100


def conversation(prompt_text: str) -> str:
    return prompt_text.split("Here is the conversation:", 1)[-1].strip()


def submission_key(prompt_text: str) -> str:
    """Reddit-v2 splits one Reddit submission into several threads (conversation paths) that share the opening post
    and can share responders' comments; neighbours from the same submission leak the held-out responder."""
    first = re.split(r"\n\s*(?:SYSTEM|USER):", conversation(prompt_text), maxsplit=1)[0]
    return " ".join(first.split())


def load_threads():
    """Threads in first-appearance order (same order as remark/data/crs/grpo_consensus/train)."""
    from datasets import load_from_disk
    d = load_from_disk(str(SRC))
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for i, p in enumerate(d["prompt"]):
        groups.setdefault(p[0]["content"], []).append(i)
    gts = d["groundtruth_with_release_year"]
    threads = []
    for ctx, idx in groups.items():
        resp = [sorted({(g[0], str(g[1])) for g in gts[i]}) for i in idx]
        threads.append({"ctx": ctx, "responders": resp})
    return threads


def phase_embed(batch: int) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer
    threads = load_threads()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "threads.jsonl", "w") as f:
        for t in threads:
            f.write(json.dumps({"responders": t["responders"]}) + "\n")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5", dtype=torch.float16).cuda().eval()
    texts = [conversation(t["ctx"]) for t in threads]
    embs = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            e = model(**enc).last_hidden_state[:, 0]  # CLS pooling (bge)
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1))
    E = torch.cat(embs)
    np.save(OUT / "emb.npy", E.cpu().numpy().astype(np.float16))
    K = max(KS)
    idx_all, sim_all = [], []
    for s in range(0, len(E), 4096):
        sims = E[s:s + 4096] @ E.T
        rows = torch.arange(s, min(s + 4096, len(E)), device=E.device)
        sims[torch.arange(len(rows)), rows] = -2.0  # exclude the thread itself
        v, i = sims.topk(K, dim=1)
        idx_all.append(i.cpu().numpy().astype(np.int32))
        sim_all.append(v.cpu().numpy().astype(np.float16))
    np.save(OUT / "nbr_idx.npy", np.concatenate(idx_all))
    np.save(OUT / "nbr_sim.npy", np.concatenate(sim_all))
    print(json.dumps({"threads": len(threads), "dim": int(E.shape[1]), "k_max": K}))


def consensus(responders) -> tuple[Counter, int]:
    c = Counter()
    for r in responders:
        c.update(tuple(x) for x in r)
    return c, len(responders)


def neighbour_prior(i, k, nbr_idx, nbr_sim, freq) -> dict:
    mu: Counter = Counter()
    w_sum = 0.0
    for j, s in zip(nbr_idx[i, :k], nbr_sim[i, :k]):
        w = max(float(s), 0.0)
        if w == 0.0:
            continue
        w_sum += w
        for item, f in freq[j].items():
            mu[item] += w * f
    if w_sum == 0.0:
        return {}
    return {item: v / w_sum for item, v in mu.most_common(TOP_PRIOR)}


def ndcg_recall(ranked, target, k=20):
    hits = [1.0 if x in target else 0.0 for x in ranked[:k]]
    dcg = sum(h / np.log2(p + 2) for p, h in enumerate(hits))
    idcg = sum(1 / np.log2(p + 2) for p in range(min(len(target), k)))
    return dcg / idcg, sum(hits) / len(target)


def phase_knn_xsub() -> None:
    """Top-200 neighbours excluding every thread of the same Reddit submission (GPU)."""
    import torch
    from datasets import load_from_disk
    cons = load_from_disk(str(REPO / "remark/data/crs/grpo_consensus/train"))
    keys = [submission_key(p[0]["content"]) for p in cons["prompt"]]
    ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    sub = torch.tensor([ids[k] for k in keys], device="cuda")
    E = torch.from_numpy(np.load(OUT / "emb.npy").astype(np.float32)).cuda()
    K = max(KS)
    idx_all, sim_all = [], []
    for s in range(0, len(E), 4096):
        sims = E[s:s + 4096] @ E.T
        sims[sub[s:s + 4096].unsqueeze(1) == sub.unsqueeze(0)] = -2.0  # same submission, incl. the thread itself
        v, i = sims.topk(K, dim=1)
        idx_all.append(i.cpu().numpy().astype(np.int32)); sim_all.append(v.cpu().numpy().astype(np.float16))
    np.save(OUT / "nbr_idx_xsub.npy", np.concatenate(idx_all)); np.save(OUT / "nbr_sim_xsub.npy", np.concatenate(sim_all))
    np.save(OUT / "submission_id.npy", sub.cpu().numpy().astype(np.int32))
    print(json.dumps({"threads": len(keys), "submissions": len(ids)}))


def phase_loro(n_eval: int, nbr: str = "_xsub") -> None:
    threads = [json.loads(line)["responders"] for line in open(OUT / "threads.jsonl")]
    nbr_idx, nbr_sim = np.load(OUT / f"nbr_idx{nbr}.npy"), np.load(OUT / f"nbr_sim{nbr}.npy").astype(np.float32)
    freq = []
    tot, n_resp = Counter(), 0
    for resp in threads:
        c, n = consensus(resp)
        freq.append({i: v / n for i, v in c.items()})
        tot.update(c)
        n_resp += n
    pop = {i: v / n_resp for i, v in tot.most_common(TOP_PRIOR)}
    rng = random.Random(42)
    multi = [i for i, r in enumerate(threads) if len(r) >= 2]
    sample = sorted(rng.sample(multi, min(n_eval, len(multi))))
    priors = {"none": None, "pop": "pop", **{f"knn{k}": k for k in KS}}
    grid = {(p, a): {"ndcg": [], "recall": [], "brier": [], "n_other": []} for p in priors for a in ALPHAS
            if (p == "none") == (a == 0.0)}
    grid[("prior_only", -1.0)] = {"ndcg": [], "recall": [], "brier": [], "n_other": []}
    for t in sample:
        resp = threads[t]
        h = rng.randrange(len(resp))
        target = {tuple(x) for x in resp[h]}
        if not target:
            continue
        c, n = consensus([r for q, r in enumerate(resp) if q != h])
        tie = {i: rng.random() for i in set(c) | target}
        mus = {"pop": pop, **{f"knn{k}": neighbour_prior(t, k, nbr_idx, nbr_sim, freq) for k in KS}}
        for (p, a), acc in grid.items():
            if p == "prior_only":
                est = dict(mus["knn50"])
            elif p == "none":
                est = {i: v / n for i, v in c.items()}
            else:
                mu = mus[p]
                est = {i: (c.get(i, 0) + a * mu.get(i, 0.0)) / (n + a) for i in set(c) | set(mu)}
            ranked = sorted(est, key=lambda i: (-est[i], tie.get(i, 0.5)))
            nd, rc = ndcg_recall(ranked, target)
            brier = sum((v - (i in target)) ** 2 for i, v in est.items()) + sum(1 for i in target if i not in est)
            acc["ndcg"].append(nd); acc["recall"].append(rc); acc["brier"].append(brier); acc["n_other"].append(n)
    rows = []
    for (p, a), acc in grid.items():
        n_other = np.array(acc["n_other"])
        row = {"prior": p, "alpha": a, "n": len(n_other)}
        for m in ("ndcg", "recall", "brier"):
            v = np.array(acc[m])
            row[m] = float(v.mean())
            for name, msk in {"n1": n_other == 1, "n2": n_other == 2, "n3_4": (n_other >= 3) & (n_other <= 4),
                              "n5plus": n_other >= 5}.items():
                row[f"{m}_{name}"] = float(v[msk].mean()) if msk.any() else None
        rows.append(row)
    base = next(r for r in rows if r["prior"] == "none")
    best = max((r for r in rows if r["prior"] not in ("none", "prior_only")), key=lambda r: r["ndcg"])
    gain = best["ndcg"] / base["ndcg"] - 1
    small = best["ndcg_n1"] / base["ndcg_n1"] - 1
    large = best["ndcg_n5plus"] / base["ndcg_n5plus"] - 1
    verdict = "PASS" if gain >= 0.05 and small > large else "PRUNE" if gain < 0.02 else "INCONCLUSIVE"
    RES.mkdir(parents=True, exist_ok=True)
    summary = {"n_threads": len(threads), "n_multi": len(multi), "n_eval": len(sample), "rows": rows,
               "best": best, "rel_gain_ndcg": gain, "rel_gain_n1": small, "rel_gain_n5plus": large, "verdict": verdict}
    (RES / f"loro_grid{nbr}.json").write_text(json.dumps(summary, indent=2))
    for r in sorted(rows, key=lambda r: -r["ndcg"]):
        print(f"{r['prior']:>10} a={r['alpha']:<5} ndcg {r['ndcg']:.4f} (n1 {r['ndcg_n1']:.4f} n5+ {r['ndcg_n5plus']:.4f})"
              f" recall {r['recall']:.4f} brier {r['brier']:.3f}")
    print(json.dumps({k: summary[k] for k in ("rel_gain_ndcg", "rel_gain_n1", "rel_gain_n5plus", "verdict")}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["embed", "knn_xsub", "loro"], required=True)
    ap.add_argument("--nbr", default="_xsub", help="neighbour file suffix: '' (v1, same-submission neighbours allowed) or _xsub")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=20000)
    a = ap.parse_args()
    if a.phase == "embed":
        phase_embed(a.batch)
    elif a.phase == "knn_xsub":
        phase_knn_xsub()
    else:
        phase_loro(a.n_eval, a.nbr)


if __name__ == "__main__":
    main()
