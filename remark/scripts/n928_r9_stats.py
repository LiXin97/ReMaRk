"""ProtoQA per-respondent information, a registered cleaning rule, overlap checks, and
proposed splits with answers.

Usage (CPU host, CPU only; plain repo env):
  env/bin/python remark/scripts/n928_r9_stats.py [--stopwords remark/data/protoqa/nltk_stopwords_english.txt]
Inputs : remark/data/protoqa/raw (git clone of iesl/protoqa-data)
Outputs: remark/data/protoqa/r9_stats.json
         remark/data/protoqa/splits/{train,validation,test_scraped,test_crowdsourced}.jsonl + manifest.json

Cleaning rule (registered 2026-10-02 before any training; applied to the scraped files only):
  C1 drop a question whose cluster counts sum to > 100 (the survey has 100 respondents; these boards list synonym
     expansions of one answer as separate entries with the same count, e.g. train_q9295: 18 entries x 62);
  C2 drop clusters with count <= 0 (answers the board shows with 0 points);
  C3 keep a question only if >= 2 clusters with count > 0 remain;
  C4 drop a train question whose normalised text equals, or is a near duplicate (content-token Jaccard >= 0.8) of, a
     validation / test / crowdsourced (dev or test) question;
  C5 (targets) every cluster keeps its original string and, if it contains "/", also the "/"-separated parts
     ("bread/yeast" -> ["bread/yeast", "bread", "yeast"]); this changes target data only, never the evaluator code.
dev.crowdsourced is kept as released (manually cleaned, multi-string clusters).
Splits: train = cleaned train.jsonl minus C4; dev.scraped (cleaned) shuffled with random.Random(42) over sorted ids:
first 400 -> validation, rest -> test_scraped; dev.crowdsourced -> test_crowdsourced.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "remark/data/protoqa"
RAW = BASE / "raw/data"
SEED, N_VAL, JACC = 42, 400, 0.8
FILES = {"train": "train/train.jsonl", "dev_scraped": "dev/dev.scraped.jsonl",
         "dev_crowdsourced": "dev/dev.crowdsourced.jsonl", "test_questions": "test/test.questions.jsonl",
         "omitted": "omitted/omitted.jsonl", "all_crowdsourced_questions": "all_crowdsourced.questions.jsonl"}


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.open()]


def qnorm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def dist(a) -> dict:
    a = np.asarray(a, dtype=float)
    return {"n": int(a.size), "min": float(a.min()), "p5": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)), "median": float(np.median(a)), "mean": round(float(a.mean()), 3),
            "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def file_stats(rows: list[dict], stop: frozenset) -> dict:
    cl = [list(r["answers"]["clusters"].values()) for r in rows]
    nc = np.array([len(c) for c in cl])
    tot = np.array([sum(x["count"] for x in c) for c in cl])
    npos = np.array([sum(1 for x in c if x["count"] > 0) for c in cl])
    strings = [s for c in cl for x in c for s in x["answers"]]
    raw_same = sum(1 for r, c in zip(rows, cl)
                   if r["answers"]["raw"] == {s: x["count"] for x in c for s in x["answers"]})
    return {
        "questions": len(rows),
        "fields": sorted({k for r in rows for k in r}),
        "answers_fields": sorted({k for r in rows for k in r["answers"]}),
        "num_answers_equals_sum_of_cluster_counts": int(sum(r["num"]["answers"] == t for r, t in zip(rows, tot))),
        "num_clusters_equals_len_clusters": int(sum(r["num"]["clusters"] == n for r, n in zip(rows, nc))),
        "raw_identical_to_clusters": raw_same,
        "strings_per_cluster_mean": round(len(strings) / nc.sum(), 3),
        "clusters_per_question": dist(nc),
        "clusters_per_question_hist": {int(k): int(v) for k, v in sorted(collections.Counter(nc.tolist()).items())},
        "respondents_per_question(sum_of_counts)": dist(tot),
        "total_eq_100": int((tot == 100).sum()), "total_lt_100": int((tot < 100).sum()),
        "total_gt_100": int((tot > 100).sum()), "total_lt_50": int((tot < 50).sum()),
        "total_in_[90,100]": int(((tot >= 90) & (tot <= 100)).sum()),
        "questions_ge2_clusters": int((nc >= 2).sum()),
        "questions_ge2_positive_clusters": int((npos >= 2).sum()),
        "questions_ge2_respondents": int((tot >= 2).sum()),
        "clusters_with_count_le_0": int(sum(1 for c in cl for x in c if x["count"] <= 0)),
        "questions_with_count_le_0_cluster": int(sum(1 for c in cl if any(x["count"] <= 0 for x in c))),
        "cluster_strings": len(strings),
        "strings_with_slash": sum("/" in s for s in strings),
        "strings_all_stopwords": sum(1 for s in strings if s.split() and all(t in stop for t in s.split())),
        "string_words": dist([len(s.split()) for s in strings]),
        "strings_over_50_chars": sum(len(s) > 50 for s in strings),
        "nonlowercase_strings": sum(s != s.lower() for s in strings),
    }


def clean_scraped(rows: list[dict]) -> tuple[list[dict], collections.Counter]:
    out, why = [], collections.Counter()
    for r in rows:
        cl = list(r["answers"]["clusters"].items())
        if sum(x["count"] for _, x in cl) > 100:
            why["C1_total_gt_100"] += 1
            continue
        pos = [(k, x) for k, x in cl if x["count"] > 0]
        why["C2_clusters_dropped"] += len(cl) - len(pos)
        if len(pos) < 2:
            why["C3_lt2_positive_clusters"] += 1
            continue
        clusters = {}
        for k, x in pos:
            alts = list(x["answers"])
            for s in x["answers"]:
                if "/" in s:
                    alts += [p.strip() for p in s.split("/") if p.strip() and p.strip() not in alts]
            clusters[k] = {"count": x["count"], "answers": alts, "original": x["answers"]}
        out.append({"id": r["metadata"]["id"], "source": r["metadata"]["source"],
                    "question": r["question"]["normalized"], "question_original": r["question"]["original"],
                    "clusters": clusters, "n_respondents": sum(x["count"] for x in clusters.values()),
                    "n_clusters": len(clusters)})
    return out, why


def keep_crowd(rows: list[dict]) -> list[dict]:
    return [{"id": r["metadata"]["id"], "source": r["metadata"]["source"], "question": r["question"]["normalized"],
             "question_original": r["question"]["original"],
             "clusters": {k: {"count": x["count"], "answers": list(x["answers"]), "original": list(x["answers"])}
                          for k, x in r["answers"]["clusters"].items()},
             "n_respondents": sum(x["count"] for x in r["answers"]["clusters"].values()),
             "n_clusters": len(r["answers"]["clusters"])} for r in rows]


def content_tokens(s: str, stop: frozenset) -> frozenset:
    return frozenset(t for t in qnorm(s).split() if t not in stop)


def near_dups(a: list[tuple[str, str]], b: list[tuple[str, str]], stop: frozenset, same: bool = False) -> dict:
    """Content-token Jaccard (>= 0.6 kept) between question texts a: [(id, text)] and b: [(id, text)] via a sparse
    intersection matrix; same=True compares a list with itself (each unordered pair once)."""
    from scipy import sparse
    ta = [content_tokens(t, stop) for _, t in a]
    tb = [content_tokens(t, stop) for _, t in b]
    vocab = {w: i for i, w in enumerate(sorted(set().union(*ta, *tb)))}

    def mat(ts):
        r = [i for i, t in enumerate(ts) for _ in t]
        c = [vocab[w] for t in ts for w in t]
        return sparse.csr_matrix((np.ones(len(r), dtype=np.int32), (r, c)), shape=(len(ts), len(vocab)))

    A, B = mat(ta), mat(tb)
    inter = (A @ B.T).tocoo()
    la, lb = np.array([len(t) for t in ta]), np.array([len(t) for t in tb])
    jac = inter.data / (la[inter.row] + lb[inter.col] - inter.data)
    keep = jac >= 0.6
    if same:
        keep &= inter.row < inter.col
    pairs = sorted(((a[i][0], b[j][0], round(float(x), 3)) for i, j, x in
                    zip(inter.row[keep], inter.col[keep], jac[keep])), key=lambda p: -p[2])
    exact_a = {qnorm(t) for _, t in a}
    exact = sum(1 for _, t in b if qnorm(t) in exact_a) if not same else None
    return {"pairs": pairs, "exact_norm": exact}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stopwords", default=str(BASE / "nltk_stopwords_english.txt"),
                    help="NLTK English stopword list (copy of nltk_data corpora/stopwords/english, 2026-10-02)")
    a = ap.parse_args()
    stop = frozenset(Path(a.stopwords).read_text().split())
    data = {k: jl(RAW / v) for k, v in FILES.items()}
    res: dict = {"data_git_commit": subprocess.run(["git", "-C", str(BASE / "raw"), "rev-parse", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip(),
                 "file_rows": {k: len(v) for k, v in data.items()},
                 "readme_counts": {"train": 8781, "dev_scraped": 979, "dev_crowdsourced": 51, "test_questions": 102},
                 "file_sha1": {k: hashlib.sha1((RAW / v).read_bytes()).hexdigest() for k, v in FILES.items()}}
    res["per_file"] = {k: file_stats(data[k], stop) for k in ("train", "dev_scraped", "dev_crowdsourced", "omitted")}

    # equal-count runs (a secondary marker of synonym explosion) among boards with total <= 100
    eq = 0
    for r in data["train"]:
        c = [x["count"] for x in r["answers"]["clusters"].values()]
        if sum(c) <= 100 and any(v >= 3 for k, v in collections.Counter(c).items() if k > 0):
            eq += 1
    res["train_total_le_100_with_>=3_clusters_sharing_a_count"] = eq

    # cleaning
    tr, why_tr = clean_scraped(data["train"])
    ds, why_ds = clean_scraped(data["dev_scraped"])
    res["cleaning"] = {"train": dict(why_tr), "dev_scraped": dict(why_ds),
                       "train_after_C1_C3": len(tr), "dev_scraped_after_C1_C3": len(ds)}

    # overlap (question text)
    T = [(r["id"], r["question"]) for r in tr]
    D = [(r["id"], r["question"]) for r in ds]
    C = [(r["metadata"]["id"], r["question"]["normalized"]) for r in data["dev_crowdsourced"]]
    Q = [(r["metadata"]["id"], r["question"]["normalized"]) for r in data["all_crowdsourced_questions"]]
    O = [(r["metadata"]["id"], r["question"]["normalized"]) for r in data["omitted"]]
    raw_T = [(r["metadata"]["id"], r["question"]["normalized"]) for r in data["train"]]
    raw_D = [(r["metadata"]["id"], r["question"]["normalized"]) for r in data["dev_scraped"]]
    ov = {}
    for name, x, y, same in (("train_vs_dev_scraped(raw)", raw_D, raw_T, False),
                             ("train_vs_crowdsourced_all154(raw)", Q, raw_T, False),
                             ("dev_scraped_vs_crowdsourced_all154(raw)", Q, raw_D, False),
                             ("train_internal(raw)", raw_T, raw_T, True),
                             ("dev_scraped_internal(raw)", raw_D, raw_D, True),
                             ("omitted_vs_train(raw)", O, raw_T, False)):
        nd = near_dups(x, y, stop, same)
        prs = nd["pairs"]
        ov[name] = {"exact_normalised_text_matches_in_second": nd["exact_norm"],
                    **{f"pairs_jaccard_ge_{t}": sum(1 for p in prs if p[2] >= t) for t in (0.6, 0.7, 0.8, 0.9, 1.0)},
                    f"first_side_questions_with_partner_jaccard_ge_{JACC}":
                        len({p[0] for p in prs if p[2] >= JACC}),
                    f"second_side_questions_with_partner_jaccard_ge_{JACC}":
                        len({p[1] for p in prs if p[2] >= JACC}),
                    "examples_ge_0.8": [p for p in prs if p[2] >= 0.8][:8]}
    ov["omitted_ids_in_train"] = len({i for i, _ in O} & {i for i, _ in raw_T})
    # answer agreement of near-duplicate train/dev pairs: same survey re-scraped, or a different survey?
    tmap = {r["metadata"]["id"]: r for r in data["train"]}
    dmap = {r["metadata"]["id"]: r for r in data["dev_scraped"]}
    agree = []
    for ia, ib, j in near_dups(raw_D, raw_T, stop)["pairs"]:
        if j >= 0.8:
            sa = {s for c in dmap[ia]["answers"]["clusters"].values() for s in c["answers"]}
            sb = {s for c in tmap[ib]["answers"]["clusters"].values() for s in c["answers"]}
            agree.append(len(sa & sb) / max(1, len(sa | sb)))
    ov["dev_train_pairs_ge_0.8_answer_string_jaccard"] = dist(agree) if agree else None
    ov["dev_train_pairs_ge_0.8_with_answer_jaccard_ge_0.5"] = sum(1 for x in agree if x >= 0.5)
    res["overlap"] = ov

    # splits
    ids = sorted(r["id"] for r in ds)
    random.Random(SEED).shuffle(ids)
    val_ids = set(ids[:N_VAL])
    val = [r for r in ds if r["id"] in val_ids]
    test_s = [r for r in ds if r["id"] not in val_ids]
    test_c = keep_crowd(data["dev_crowdsourced"])
    held = [(r["id"], r["question"]) for r in val + test_s] + Q
    nd = near_dups(held, T, stop)
    drop = {p[1] for p in nd["pairs"] if p[2] >= JACC} | {i for i, t in T if qnorm(t) in {qnorm(x) for _, x in held}}
    train = [r for r in tr if r["id"] not in drop]
    res["cleaning"]["C4_train_dropped_near_dup_of_heldout"] = len(drop)
    splits = {"train": train, "validation": val, "test_scraped": test_s, "test_crowdsourced": test_c}
    out = BASE / "splits"
    out.mkdir(exist_ok=True)
    sizes = {}
    for k, rows in splits.items():
        with (out / f"{k}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        nresp = np.array([r["n_respondents"] for r in rows])
        ncl = np.array([r["n_clusters"] for r in rows])
        sizes[k] = {"questions": len(rows), "respondent_rows": int(nresp.sum()),
                    "respondents_per_question_mean": round(float(nresp.mean()), 2),
                    "clusters_per_question_mean": round(float(ncl.mean()), 2),
                    "questions_ge2_clusters": int((ncl >= 2).sum()),
                    "questions_ge2_respondents": int((nresp >= 2).sum()),
                    "questions_total_lt_50": int((nresp < 50).sum())}
    res["splits"] = sizes
    manifest = {"script": "remark/scripts/n928_r9_stats.py",
                "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip(),
                "data_git_commit": res["data_git_commit"], "seed": SEED, "n_validation": N_VAL,
                "near_dup_jaccard": JACC, "sizes": sizes, "rule": __doc__.split("Cleaning rule")[1].strip()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (BASE / "r9_stats.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("file_rows", "cleaning", "splits")}, indent=1))
    print(json.dumps(res["overlap"], indent=1)[:4000])


if __name__ == "__main__":
    main()
