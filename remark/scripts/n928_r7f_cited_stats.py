"""Trace or recompute statistics the paper cites from internal notes only. CPU only.

For each statistic: the sentences in the notes file (env RESEARCH_LOG, optional) and the paper source that state it
(grep), the value recomputed from data (definition and inputs below), the source file if one exists, and whether the
recomputation matches the cited value at the cited precision.

Definitions
  max_sum_reward ("up to 47 per position"): for each Reddit-v2 training request (rows of data/raw/reddit_v2/
    processed_datasets/grpo/grpo_dataset/train sharing a prompt), c_i = number of its responders (rows) whose ground
    truth contains item i = (title, int(year)); the sum reward of a position is c_i of the matched item; statistic =
    max over requests and items of c_i. Cross-check: max of groundtruth_weights x n_responders in
    remark/data/crs/grpo_consensus/train.
  union_vs_sum ("89% / 76% / 24%"): pooled over training requests, with U_x the union items of request x:
    share of union items with c_i = 1 (sum_x #{i in U_x: c_i = 1} / sum_x |U_x|); share of the sum's mass on those
    items (sum_x #{c_i = 1} / sum_x sum_i c_i); the complement (mass on items named by >= 2 responders, counted once
    by the bounded sum); also the union's share of the sum's mass (sum_x |U_x| / sum_x sum_i c_i) and the per-request
    mean of the first share.
  responders ("17.2 / 5.35"): Reddit-v2 test split (sft_dataset/test), context = rows sharing a prompt, n = rows per
    context; row-weighted mean n = sum n^2 / sum n; per-request mean n = rows / contexts.
  submissions ("48,618"): distinct crs_neighbor_prior.submission_key (normalised opening post, the neighbour-prior definition) over
    the 72,233 training contexts (prompts of remark/data/crs/grpo_union/train); also contexts in multi-path
    submissions and the largest number of paths.
  books_popularity ("Books popularity R@20", two decimals in points): crs_popularity_baseline's definition applied to
    Reddit-Books: popularity = occurrences of (title, year) in the Books training ground truth
    (remark/outputs/crs/train_gt_popularity_reddit_books.json, asserted equal to a recount of sft_dataset/train);
    ranking by count, ties by title then year; per row drop titles in seen_titles; top 20; crs_eval.hits_for truncate
    with remark/data/crs/reddit_books/catalog.pkl; per-row mean R@20 (and per request). Computed in memory only.
    Variant: popularity counted on the per-responder GRPO rows (remark/data/crs/reddit_books/grpo/train).
  extraction_precision ("0.997"): remark/outputs/books/audit/judge.json, extraction verdicts of the GPT-5.6 judge on
    200 comments: correct / all judged items; the go/no-go file remark/outputs/books/go_nogo.json records the value
    and its CI.
Output: remark/outputs/crs/analysis/n928_r7f_cited_stats.json
Usage:  env/bin/python remark/scripts/n928_r7f_cited_stats.py
"""
from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import DATA, REPO, hits_for, recall_ndcg  # noqa: E402
from crs_neighbor_prior import submission_key  # noqa: E402
from crs_popularity_baseline import recommend  # noqa: E402

OUT = REPO / "remark/outputs/crs/analysis/n928_r7f_cited_stats.json"
NOTES = REPO / os.environ.get("RESEARCH_LOG", "notes/research_log.md")
TEX = REPO / "remark/paper/tex/main.tex"
ROWS = REPO / "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train"
CRS = REPO / "remark/data/crs"
BOOKS = CRS / "reddit_books"
POP_BOOKS = REPO / "remark/outputs/crs/train_gt_popularity_reddit_books.json"


def grep(pattern: str) -> list[dict]:
    out = []
    for f in (NOTES, TEX):
        if not f.exists():
            continue
        for i, line in enumerate(open(f, errors="replace"), 1):
            if re.search(pattern, line):
                out.append({"file": str(f.relative_to(REPO)), "line": i, "text": line.strip()[:300]})
    return out


def request_counts() -> tuple[list[Counter], list[int]]:
    rows = load_from_disk(str(ROWS)).select_columns(["prompt", "groundtruth_with_release_year"])
    by: dict[str, Counter] = {}
    n: dict[str, int] = defaultdict(int)
    for b in rows.iter(batch_size=20_000):
        for p, gts in zip(b["prompt"], b["groundtruth_with_release_year"]):
            k = p[0]["content"]
            by.setdefault(k, Counter()).update({(g[0], int(g[1])) for g in gts})
            n[k] += 1
    return list(by.values()), [n[k] for k in by]


def books_popularity(pop: Counter, split: str, catalog: set) -> dict:
    ranking = sorted(pop, key=lambda k: (-pop[k], k[0], k[1]))
    ds = load_from_disk(str(BOOKS / "sft_dataset" / split))
    r20, ctx = [], defaultdict(list)
    for i, (p, gts_raw, seen_l) in enumerate(zip(ds["prompt"], ds["groundtruth_with_release_year"], ds["seen_titles"])):
        seen = set(seen_l)
        recs = recommend(ranking, seen)
        gts = [(g[0], int(g[1])) for g in gts_raw]
        hits, _ = hits_for(recs, gts, seen, catalog, "truncate")
        r20.append(recall_ndcg(hits, len(gts), 20)[0])
        ctx[p[0]["content"]].append(i)
    r20 = np.array(r20)
    return {"rows": len(ds), "requests": len(ctx), "per_row_R@20": float(r20.mean()),
            "per_request_R@20": float(np.mean([r20[g].mean() for g in ctx.values()])),
            "per_row_R@20_points_2dp": round(100 * float(r20.mean()), 2)}


def main() -> None:
    res: dict = {"definition": __doc__.split("Definitions", 1)[1].split("Output:", 1)[0].strip(), "stats": {}}
    try:
        res["git"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True).stdout.strip()
    except OSError:
        pass
    st = res["stats"]
    # 1-2: responder counts over training requests
    counts, n_x = request_counts()
    cmax = max(max(c.values()) for c in counts)
    cons = load_from_disk(str(CRS / "grpo_consensus/train")).select_columns(["groundtruth_weights", "n_responders"])
    cons_max = max(round(max(float(x) for x in w) * int(n)) for w, n in zip(cons["groundtruth_weights"], cons["n_responders"]))
    st["up_to_47_per_position"] = {
        "cited": 47, "recomputed": int(cmax), "cross_check_consensus_w_times_n": int(cons_max),
        "matches": cmax == 47 == cons_max, "source": "recomputed (cited only as an audit fix in the notes); "
        "data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train, remark/data/crs/grpo_consensus/train",
        "requests": len(counts), "max_n_x": int(max(n_x)), "cited_in": grep(r"up to 47|max 47")}
    n_union = sum(len(c) for c in counts)
    n_single = sum(sum(1 for v in c.values() if v == 1) for c in counts)
    mass = sum(sum(c.values()) for c in counts)
    per_req = np.mean([sum(1 for v in c.values() if v == 1) / len(c) for c in counts if c])
    per_req_n2 = np.mean([sum(1 for v in c.values() if v == 1) / len(c) for c, n in zip(counts, n_x) if c and n >= 2])
    st["89_76_24_percent"] = {
        "cited": {"union_items_named_by_one": 0.89, "sum_mass_on_those": 0.76, "sum_mass_on_items_named_by_ge2": 0.24},
        "recomputed": {"union_items_named_by_one": n_single / n_union, "sum_mass_on_those": n_single / mass,
                       "sum_mass_on_items_named_by_ge2": 1 - n_single / mass, "union_share_of_sum_mass": n_union / mass,
                       "per_request_mean_share_named_by_one": float(per_req),
                       "per_request_mean_share_named_by_one_requests_n_ge_2": float(per_req_n2),
                       "union_items": n_union, "sum_mass": mass, "requests_with_empty_union": sum(1 for c in counts if not c)},
        "n9_correction_block": "88.8% of union items, 76.5% of the summed mass, union keeps 86.1% (per-request mean 91.2%)",
        "note": "the pooled shares (the paper's 89 / 76 / 24) reproduce; the notes' per-request mean 91.2% (not cited in the "
                "paper) does not reproduce over all requests, see per_request_mean_* for the two candidate definitions",
        "matches": round(100 * n_single / n_union) == 89 and round(100 * n_single / mass) == 76
        and round(100 * (1 - n_single / mass)) == 24,
        "source": "notes, correction block (computed from "
                  "data/crs/grpo_consensus/train; no file); recomputed here from the per-responder rows",
        "cited_in": grep(r"89\\?%|76\\?%|24\\?%|88\.8|76\.5")}
    # 3: responders per test request
    test = load_from_disk(str(DATA / "test")).select_columns(["prompt"])
    nc = np.array(list(Counter(p[0]["content"] for p in test["prompt"]).values()), dtype=float)
    st["17.2_5.35_responders"] = {
        "cited": {"row_weighted_mean_n": 17.2, "per_request_mean_n": 5.35},
        "recomputed": {"row_weighted_mean_n": float((nc ** 2).sum() / nc.sum()), "per_request_mean_n": float(nc.mean()),
                       "rows": int(nc.sum()), "requests": len(nc)},
        "matches": round(float((nc ** 2).sum() / nc.sum()), 1) == 17.2 and round(float(nc.mean()), 2) == 5.35,
        "source": "recomputed from data/raw/reddit_v2/processed_datasets/sft_dataset/test (cited in the notes "
                  "amendment text)", "cited_in": grep(r"17\.2\b|5\.35\b")}
    # 4: submissions
    up = [p[0]["content"] for p in load_from_disk(str(CRS / "grpo_union/train")).select_columns(["prompt"])["prompt"]]
    sft_ctx = set(p[0]["content"] for p in load_from_disk(str(DATA / "train")).select_columns(["prompt"])["prompt"])
    subs = Counter(submission_key(c) for c in up)
    st["48618_submissions"] = {
        "cited": 48618, "recomputed": len(subs), "matches": len(subs) == 48618,
        "contexts": len(up), "contexts_in_multi_path_submissions": int(sum(v for v in subs.values() if v > 1)),
        "max_paths_per_submission": int(max(subs.values())),
        "sft_train_contexts_equal_grpo_contexts": sft_ctx == set(up), "sft_train_contexts": len(sft_ctx),
        "sft_train_contexts_subset_of_grpo_contexts": sft_ctx <= set(up),
        "source": "notes (no file); recomputed with crs_neighbor_prior.submission_key over "
                  "remark/data/crs/grpo_union/train", "cited_in": grep(r"48,618|48618")}
    # 5: Books popularity baseline
    catalog = set(tuple(x) for x in pickle.load(open(BOOKS / "catalog.pkl", "rb")))
    pop = Counter({tuple(json.loads(k)): v for k, v in json.load(open(POP_BOOKS)).items()})
    recount = Counter()
    for gts in load_from_disk(str(BOOKS / "sft_dataset/train"))["groundtruth_with_release_year"]:
        for g in gts:
            recount[(g[0], int(g[1]))] += 1
    pop_grpo = Counter()
    for gts in load_from_disk(str(BOOKS / "grpo/train"))["groundtruth_with_release_year"]:
        for g in gts:
            pop_grpo[(g[0], int(g[1]))] += 1
    main_pop = {sp: books_popularity(pop, sp, catalog) for sp in ("validation", "test")}
    var_pop = {sp: books_popularity(pop_grpo, sp, catalog) for sp in ("validation", "test")}
    st["books_popularity_R@20"] = {
        "cited": {"test": 0.058, "validation": 0.070, "paper_table": "5.8 (test, one decimal)"},
        "recomputed": main_pop, "variant_popularity_from_grpo_rows": var_pop,
        "popularity_file_equals_sft_train_recount": recount == pop,
        "note": "both popularity sources round to the cited 0.058 / 0.070; the source used for the cited value is not "
                "recorded, so the two-decimal value is 5.77 under the crs_eval / crs_popularity_baseline popularity "
                "(primary) and 5.84 under the per-responder GRPO-row popularity",
        "matches_variant": round(var_pop["test"]["per_row_R@20"], 3) == 0.058
        and round(var_pop["validation"]["per_row_R@20"], 3) == 0.070,
        "matches": round(main_pop["test"]["per_row_R@20"], 3) == 0.058
        and round(main_pop["validation"]["per_row_R@20"], 3) == 0.070,
        "two_decimals_points_test": main_pop["test"]["per_row_R@20_points_2dp"],
        "source": "notes (Reddit-Books go/no-go) and outputs/books/go_nogo_report.md (text only; no script or "
                  "per-instance file; make_paper_figures.py hard-codes BOOKS_POP = 0.058); recomputed here",
        "cited_in": grep(r"0\.058|5\.8 on test|BOOKS_POP|popularity top-20")}
    # 6: extraction precision
    judge = json.load(open(REPO / "remark/outputs/books/audit/judge.json"))["extraction"]
    verdicts = Counter(v["verdict"] for c in judge for v in c["judge"]["verdicts"])
    gn = json.load(open(REPO / "remark/outputs/books/go_nogo.json"))["criteria"]["extraction_precision>=0.85"]
    prec = verdicts["correct"] / sum(verdicts.values())
    st["extraction_precision_0.997"] = {
        "cited": 0.997, "recomputed": prec, "verdicts": dict(verdicts), "comments": len(judge),
        "go_nogo_json": gn, "matches": round(prec, 3) == 0.997 and abs(gn["value"] - prec) < 1e-4,
        "source": "remark/outputs/books/audit/judge.json (verdicts); remark/outputs/books/go_nogo.json",
        "cited_in": grep(r"0\.997")}
    OUT.write_text(json.dumps(res, indent=1))
    for k, v in st.items():
        print(k, "matches" if v["matches"] else "DIFFERS", json.dumps(v["recomputed"])[:400])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
