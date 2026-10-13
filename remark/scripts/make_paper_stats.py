"""Collect every number the Experiments figures and tables use into one JSON (input to make_paper_figures.py).

Recomputes per-row test R@20 of each run from its per_instance.jsonl (crs_breakdown.per_row, Rank-GRPO protocol) to get
absolute 95% CIs (bootstrap over requests, row-weighted, 2,000 resamples, seed 42); paired differences, bucket and
robustness numbers are read from the analysis files written by crs_compare.py / crs_bucket_boot.py and friends, so the
figures show exactly the numbers in the text. Validation curves come from outputs/crs/val/*/summary.json.
Missing inputs (pending runs) are skipped and listed under "missing". Per-row arrays are cached (fig_cache/, ~90 KB each).

Usage: env/bin/python remark/scripts/make_paper_stats.py   ->   remark/outputs/crs/analysis/paper_fig_stats.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_breakdown import per_row  # noqa: E402
from crs_eval import DATA, REPO, RG  # noqa: E402

C = REPO / "remark/outputs/crs"
T, V, A = C / "test", C / "val", C / "analysis"
CACHE = A / "fig_cache"
BOOKS = REPO / "remark/data/crs/reddit_books"
MUSIC = REPO / "remark/data/crs/music"

# test runs: setting -> arm -> run dir under outputs/crs/test (seed 3407 unless the arm name says otherwise).
# "remark" = the final ReMaRk configuration (bounded sum, group scaling, request sampling; , user decision
# 2026-09-29 23:33 UTC); "registered" = the configuration the registered method-arm rule selected (bounded sum, batch
# scaling), whose registered primary comparison passed and is reported as the pre-registered result.
TEST = {
    "2B": {"sft": "test_2B_sft_step1500", "rankgrpo": "test_2B_single_group_s3407_step2000",
           "perresp": "test_2B_single_batch_s3407_step1750", "remark": "test_2B_union_group_step2000",
           "registered": "test_2B_union_s3407_step1500",
           "remark_s42": "test_2B_union_group_s42_p0c_step2000", "remark_s1234": "test_2B_union_group_s1234_p0c_step2000",
           "registered_s42": "test_2B_union_s42_step2000", "registered_s1234": "test_2B_union_s1234_step2000",
           "rankgrpo_s42": "test_2B_single_group_s42_step2000", "rankgrpo_s1234": "test_2B_single_group_s1234_step2000",
           # baselines (tab_main rows): Rank-GRPO G=16, multi-positive SFT, pooled-positive DPO (selected)
           "g16": "test_2B_single_group_s3407_p0c_G16_step2000", "multipos": "test_sft_multipos_2B_step1000",
           "dpo": "test_2B_dpo_union_lr5e-6_ep3_s3407_step250"},
    "4B": {"sft": "test_4B_sft_step1000", "rankgrpo": "test_4B_single_group_s3407_step2000",
           "perresp": "test_4B_single_batch_s3407_step2000", "remark": "test_4B_union_group_s3407_step2000",
           "registered": "test_4B_union_s3407_step2000",
           "registered_s42": "test_4B_union_s42_step2000", "rankgrpo_s42": "test_4B_single_group_s42_step1750",
           "remark_s42": "test_4B_union_group_s42_p0c_step2000",
           # third seed
           "remark_s1234": "test_4B_union_group_s1234_p0c_step2000", "rankgrpo_s1234": "test_4B_single_group_s1234_p0c_step2000",
           "g16": "test_4B_single_group_s3407_p0c_G16_step1750", "multipos": "test_sft_multipos_4B_step1500",
           "dpo": "test_4B_dpo_union_lr5e-6_ep3_s3407_step250"},
    "9B": {"sft": "test_9B_sft_step1000", "rankgrpo": "test_9B_single_group_s3407_step2000",
           "perresp": "test_9B_single_batch_s3407_step2000", "remark": "test_9B_union_group_s3407_p0c_step2000",
           "registered": "test_9B_union_s3407_step2000",
           "remark_s42": "test_9B_union_group_s42_p0c_step2000", "rankgrpo_s42": "test_9B_single_group_s42_p0c_step2000",
           # third seed
           "remark_s1234": "test_9B_union_group_s1234_p0c_step2000", "rankgrpo_s1234": "test_9B_single_group_s1234_p0c_step1250"},
    "books": {"sft": "test_books_sft_step1000", "rankgrpo": "test_books_single_group_p0d_step2000",
              "perresp": "test_books_single_batch_p0d_step2000", "remark": "test_books_union_group_p0d_step1500",
              "registered": "test_books_union_p0d_step2000",
              "remark_s42": "test_books_union_group_s42_p0d_step2000",
              "perresp_s42": "test_books_single_batch_s42_p0d_step2000",
              # third seed (same P0d configuration; run names carry the configuration suffix)
              "remark_s1234": "test_reddit_books_2B_union_group_s1234_p0c_cap384_ex_step2000",
              "perresp_s1234": "test_reddit_books_2B_single_batch_s1234_p0c_cap384_ex_step2000"},
    # Reddit-Books at 4B (; P0d arms, cap 384, exact items; SFT selected on validation @1500, tested once)
    "books4B": {"sft": "test_books4B_sft_step1500",
                "rankgrpo": "test_reddit_books_4B_single_group_s3407_p0c_cap384_ex_step2000",
                "perresp": "test_reddit_books_4B_single_batch_s3407_p0c_cap384_ex_step2000",
                "remark": "test_reddit_books_4B_union_group_s3407_p0c_cap384_ex_step2000",
                # (10-07): union + batch scaling, seed 3407, best-validation lock
                "registered": "test_reddit_books_4B_union_batch_s3407_p0c_cap384_ex_step1500"},
    # second model family: Llama-3.2-3B from the released Rank-GRPO SFT checkpoint, our pipeline with P0c
    # SFT = the released Rank-GRPO Llama-3.2-3B SFT checkpoint (outputs/crs/sft_llama3b_test, not under test/)
    "llama3b": {"sft": "../sft_llama3b_test", "rankgrpo": "test_llama3b_single_group_s3407_p0c_step2000",
                "remark": "test_llama3b_union_group_s3407_p0c_step1750",
                # (10-07): per-responder + batch scaling and union + batch scaling, seed 3407
                "perresp": "test_llama3b_single_batch_s3407_p0c_step2000",
                "registered": "test_llama3b_union_batch_s3407_p0c_step2000"},
    "llama8b": {"sft": "test_llama8b_sft_step1000", "rankgrpo": "test_llama8b_single_group_s3407_p0c_step2000",
                "remark": "test_llama8b_union_group_s3407_p0c_step1500",
                "perresp": "test_llama8b_single_batch_s3407_p0c_step2000",
                "registered": "test_llama8b_union_batch_s3407_p0c_step2000"},
    # Qwen3.5-27B pair (; ReMaRk restarted from scratch 10-01, both arms 2,000 steps, locked @2000, tested once)
    "27B": {"sft": "test_27B_sft_step1000", "rankgrpo": "test_27B_single_group_s3407_p0c_step2000",
            "remark": "test_27B_union_group_s3407_p0c_step2000"},
    # Qwen3.5-35B-A3B (MoE) pair (/; both arms 2,000 steps, locked @2000, tested once)
    "35BA3B": {"sft": "test_35BA3B_sft_step1500", "rankgrpo": "test_35BA3B_single_group_s3407_p0c_step2000",
               "remark": "test_35BA3B_union_group_s3407_p0c_step2000",
               # 10-08: batch-scaled arms on the retrained SFT (qwen35_35BA3B_r2 step 1500), s3407, locked @2000
               "perresp": "test_35BA3B_single_batch_s3407_p0c_step2000",
               "registered": "test_35BA3B_union_batch_s3407_p0c_step2000"},
    # E12 music, Qwen3.5-2B (/; SFT selected @1500 on validation; both arms --exact_items, locked on validation,
    # tested once)
    "music": {"sft": "test_music_sft_step1500", "rankgrpo": "test_music_2B_single_group_s3407_p0c_ex_step2000",
              "remark": "test_music_2B_union_group_s3407_p0c_ex_step1250",
              "perresp": "test_music_2B_single_batch_s3407_p0c_ex_step2000",
              "registered": "test_music_2B_union_batch_s3407_p0c_ex_step1000"},
    # reference systems without RL (/E6)
    "baselines_reddit": {"gpt56": "test_zeroshot_gpt56", "knn": "test_knn_prior_k200", "qwen27b": "test_zeroshot_qwen35_27B"},
    "baselines_books": {"gpt56": "test_zeroshot_gpt56_books", "knn": "test_knn_prior_k200_books",
                        "qwen27b": "test_zeroshot_qwen35_27B_books"},
    "baselines_music": {"gpt56": "test_zeroshot_gpt56_music", "knn": "test_knn_prior_k200_music",
                        "gpt56_yearconv": "test_zeroshot_gpt56_music_yearconv",  # year convention stated (Table 12)
                        "popularity": "popularity_train_top20_music"},
}
# validation curves: setting -> arm -> run name under val/grpo_qwen_<run>_step<k>; step 0 = the selected SFT checkpoint
VAL = {
    "2B": {"sft": "sft_qwen35_2B_step1500", "rankgrpo": "2B_single_group_s3407_p0c", "perresp": "2B_single_batch_s3407_p0c",
           "remark": "2B_union_group_s3407_p0c", "registered": "2B_union_batch_s3407_p0c",
           "rankgrpo_4k": "2B_single_group_s3407_p0c_ms4000", "registered_4k": "2B_union_batch_s3407_p0c_ms4000",
           "remark_4k": "2B_union_group_s3407_p0c_ms4000",
           "remark_rerun_4k": "2B_union_group_s3407_p0c_cap320_ms4000",
           "singlethr": "2B_singlethr_batch_s3407_p0c", "consensusrw": "2B_consensusrw_batch_s3407_p0c",
           "consensus": "2B_consensus_batch_s3407_p0c",
           # both arms trained to the registered plateau (8k cap; best 4,000 / 5,250)
           "rankgrpo_8k": "2B_single_group_s3407_p0c_ms8000", "remark_8k": "2B_union_group_s3407_p0c_ms8000",
           # the seed-42 plateau pair (best 4,500 / 7,250; Rank-GRPO collapses after 7,250)
           "rankgrpo_8k_s42": "2B_single_group_s42_p0c_ms8000", "remark_8k_s42": "2B_union_group_s42_p0c_ms8000"},
    "4B": {"sft": "sft_qwen35_4B_step1000", "rankgrpo": "4B_single_group_s3407_p0c", "perresp": "4B_single_batch_s3407_p0c",
           "remark": "4B_union_group_s3407_p0c", "registered": "4B_union_batch_s3407_p0c",
           "rankgrpo_4k": "4B_single_group_s3407_p0c_ms4000", "remark_4k": "4B_union_group_s3407_p0c_ms4000",
           # the 4B plateau pair (best 3,000 / 5,250)
           "rankgrpo_8k": "4B_single_group_s3407_p0c_ms8000", "remark_8k": "4B_union_group_s3407_p0c_ms8000"},
    "9B": {"sft": "sft_qwen35_9B_step1000", "rankgrpo": "9B_single_group_s3407_p0c", "perresp": "9B_single_batch_s3407_p0c",
           "remark": "9B_union_group_s3407_p0c", "registered": "9B_union_batch_s3407_p0c",
           "remark_rerun": "9B_union_group_s3407_p0c_cap320",
           # / the 9B plateau pair
           "rankgrpo_8k": "9B_single_group_s3407_p0c_ms8000", "remark_8k": "9B_union_group_s3407_p0c_ms8000"},
    "books": {"sft": "sft_reddit_books_2B_step1000", "rankgrpo": "reddit_books_2B_single_group_s3407_p0c_cap384_ex",
              "perresp": "reddit_books_2B_single_batch_s3407_p0c_cap384_ex",
              "remark": "reddit_books_2B_union_group_s3407_p0c_cap384_ex",
              "registered": "reddit_books_2B_union_batch_s3407_p0c_cap384_ex",
              # the Reddit-Books plateau pair (best 2,000 / 3,250)
              "rankgrpo_8k": "reddit_books_2B_single_group_s3407_p0c_cap384_ex_ms8000",
              "remark_8k": "reddit_books_2B_union_group_s3407_p0c_cap384_ex_ms8000"},
    # E12 music 2B and the music plateau pair (best 1,250 / 4,750); drawn in fig_curves_music
    "music": {"sft": "sft_music_2B_step1500", "rankgrpo": "music_2B_single_group_s3407_p0c_ex",
              "remark": "music_2B_union_group_s3407_p0c_ex",
              "rankgrpo_8k": "music_2B_single_group_s3407_p0c_ex_ms8000", "remark_8k": "music_2B_union_group_s3407_p0c_ex_ms8000"},
}
SIZES = ["2B", "4B", "9B"]
missing: list[str] = []


def load(path: Path):
    if not path.exists():
        missing.append(str(path.relative_to(REPO)))
        return None
    return json.load(open(path))


def groups_of(ds) -> list[list[int]]:
    ctx = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx[p[0]["content"]].append(i)
    return list(ctx.values())


def abs_stats(r20: np.ndarray, groups: list[list[int]]) -> dict:
    sums = np.array([r20[g].sum() for g in groups]); sizes = np.array([len(g) for g in groups])
    means = sums / sizes
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    row_boot = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    req_boot = means[draws].mean(axis=1)
    return {"row": [float(r20.mean()), *map(float, np.percentile(row_boot, [2.5, 97.5]))],
            "req": [float(means.mean()), *map(float, np.percentile(req_boot, [2.5, 97.5]))]}


def test_abs() -> dict:
    out = {}
    CACHE.mkdir(parents=True, exist_ok=True)
    reddit = load_from_disk(str(DATA / "test"))
    books = load_from_disk(str(BOOKS / "sft_dataset/test"))
    music = load_from_disk(str(MUSIC / "sft_dataset/test"))
    cat_r = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    cat_b = set(tuple(x) for x in pickle.load(open(BOOKS / "catalog.pkl", "rb")))
    cat_m = set(tuple(x) for x in pickle.load(open(MUSIC / "catalog.pkl", "rb")))
    g_r, g_b, g_m = groups_of(reddit), groups_of(books), groups_of(music)
    for setting, arms in TEST.items():
        ds, cat, grp = ((books, cat_b, g_b) if setting in ("books", "books4B", "baselines_books") else
                        (music, cat_m, g_m) if setting in ("music", "baselines_music") else (reddit, cat_r, g_r))
        out[setting] = {}
        for arm, run in arms.items():
            rd = T / run
            if not (rd / "per_instance.jsonl").exists():
                missing.append(f"test/{run}"); continue
            cf = CACHE / f"{run.lstrip('./')}.npy"
            if cf.exists():
                r20 = np.load(cf)
            else:
                r20 = per_row(rd, ds, cat)[0]; np.save(cf, r20)
            # runs outside test/ (the released Llama-3.2-3B SFT) are named by their path under the repo
            name = run if "/" not in run else str(Path(os.path.normpath(T / run)).relative_to(REPO))
            out[setting][arm] = {"run": name, **abs_stats(r20, grp)}
            print(setting, arm, f"{out[setting][arm]['row'][0]:.4f}")
    # released 15.8k-step Rank-GRPO checkpoint (Llama-3.2-3B) and the popularity list, Reddit-v2 test
    for name, rd in (("released", C / "rankgrpo_llama3b_test"), ("popularity", T / "popularity_train_top20")):
        if not (rd / "per_instance.jsonl").exists():
            missing.append(str(rd.relative_to(REPO))); continue
        cf = CACHE / f"ref_{name}.npy"
        r20 = np.load(cf) if cf.exists() else per_row(rd, reddit, cat_r)[0]
        np.save(cf, r20)
        out.setdefault("reference", {})[name] = {"run": str(rd.relative_to(REPO)), **abs_stats(r20, g_r)}
    if "popularity" in out.get("baselines_music", {}):  # music popularity list (crs_popularity_baseline.py --dataset music)
        out.setdefault("reference", {})["popularity_music"] = out["baselines_music"]["popularity"]
    return out


def val_curves() -> dict:
    out = {}
    for setting, arms in VAL.items():
        out[setting] = {}
        for arm, run in arms.items():
            if arm == "sft":
                s = load(V / run / "summary.json")
                if s:
                    out[setting]["sft"] = s["truncate_R@20"]
                continue
            pts = {}
            for d in V.glob(f"grpo_qwen_{run}_step*"):
                if d.name[len(f"grpo_qwen_{run}_step"):].isdigit() and (d / "summary.json").exists():
                    pts[int(d.name.rsplit("step", 1)[1])] = json.load(open(d / "summary.json"))["truncate_R@20"]
            if pts:
                out[setting][arm] = dict(sorted(pts.items()))
            else:
                missing.append(f"val/grpo_qwen_{run}_step*")
    return out


def pair(path: Path, a_sub: str, b_sub: str, metric: str = "R@20"):
    d = load(path)
    if not d:
        return None
    for p in d["pairs"]:
        if a_sub in p["a"] and b_sub in p["b"]:
            m = p[metric]
            return [m["diff"], m["ci_low"], m["ci_high"]]
    missing.append(f"{path.name}: {a_sub} - {b_sub}")
    return None


# Table 2 Delta baseline = the per-responder arm with the higher validation best : per-responder + batch
# scaling for Books 2B/4B, Llama-3.2-3B (25.61 vs 25.16) and Music (13.37 vs 12.91); Rank-GRPO elsewhere (Llama-8B 26.84 vs 26.50)
DELTA_PERRESP = ("books", "books4B", "llama3b", "music")


def deltas() -> dict:
    out = {"primary": {}, "registered": {}, "scale_union": {}, "buckets": {}, "decomposition": {}, "robust": {},
           "baselines": {}}
    # primary: ReMaRk (union + group) minus the better per-responder arm on validation ((1) comparisons)
    for s, stem in (("2B", "test_2B_union_group_{}.json"), ("4B", "test_4B_union_group_{}_context.json"),
                    ("9B", "test_9B_union_group_{}_context.json"), ("books", "test_books_union_group_{}.json"),
                    ("books4B", "test_books4B_union_group_vs_single_batch_{}.json"),
                    ("llama3b", "test_llama3b_n934_vs_single_batch_{}.json"),
                    ("llama8b", "test_llama8b_union_group_{}.json"), ("27B", "test_27B_pair_{}.json"),
                    ("35BA3B", "test_35BA3B_pair_{}.json"), ("music", "test_music_n934_vs_single_batch_{}.json")):
        a, b = TEST[s]["remark"], TEST[s]["perresp" if s in DELTA_PERRESP else "rankgrpo"]
        out["primary"][s] = {"row": pair(A / stem.format("row"), a, b), "req": pair(A / stem.format("unit"), a, b),
                             "ndcg": pair(A / stem.format("row"), a, b, "N@20")}
        # group minus batch scaling for the union (final vs registered configuration)
        if "registered" not in TEST[s] or s == "35BA3B":  # 35B batch arms use the retrained SFT (r2): no group-vs-batch pair
            continue
        r = TEST[s]["registered"]
        out["scale_union"][s] = {"row": pair(A / stem.format("row"), a, r), "req": pair(A / stem.format("unit"), a, r)}
    out["primary"]["books"]["vs_rankgrpo"] = pair(A / "test_books_union_group_row.json", TEST["books"]["remark"],
                                                  TEST["books"]["rankgrpo"])
    out["primary"]["books4B"]["vs_rankgrpo"] = pair(A / "test_books4B_union_group_vs_single_group_row.json",
                                                    TEST["books4B"]["remark"], TEST["books4B"]["rankgrpo"])
    pooled = load(A / "test_pooled_sizes_union_group.json")
    if pooled:
        out["primary"]["pooled"] = {"row": [pooled["pooled_mean_diff"], *pooled["ci95"]]}
    # the registered result: union + batch (the arm the registered rule selected), registered primary rule
    for s, stem in (("2B", "test_2B_primary_{}"), ("4B", "test_4B_primary_context_{}"), ("9B", "test_9B_primary_context_{}")):
        a, b = TEST[s]["registered"], TEST[s]["rankgrpo"]
        out["registered"][s] = {"row": pair(A / f"{stem.format('row')}.json", a, b),
                                "req": pair(A / f"{stem.format('unit')}.json", a, b),
                                "ndcg": pair(A / f"{stem.format('row')}.json", a, b, "N@20")}
    a, b = TEST["books"]["registered"], TEST["books"]["perresp"]
    out["registered"]["books"] = {"row": pair(A / "test_books_primary_row.json", a, b),
                                  "req": pair(A / "test_books_primary_unit.json", a, b),
                                  "ndcg": pair(A / "test_books_primary_row.json", a, b, "N@20")}
    pooled = load(A / "test_pooled_sizes.json")
    if pooled:
        out["registered"]["pooled"] = {"row": [pooled["pooled_mean_diff"], *pooled["ci95"]]}
    # responder-count buckets of ReMaRk - Rank-GRPO (all sizes) and of the reward-only contrast 6 - 3 (2B)
    bu = load(A / "test_union_group_primary_buckets.json")
    if bu:
        for r in bu:
            out["buckets"][next(s for s in SIZES if f"test_{s}_" in r["a"])] = r
    # decomposition along the 2B ablation arms (row numbers of the ablation table; 7 = ReMaRk):
    # 7 - 1 = (7 - 6) scale for the union + (6 - 3) reward + (3 - 2) sampling + (2 - 1) scale for the per-responder reward
    dec = load(A / "test_2B_decomposition_buckets.json")
    if dec:
        names = {("union", "singlethr"): "reward", ("singlethr", "single_batch"): "sampling",
                 ("single_batch", "single_group"): "scale"}
        for r in dec:
            for (ka, kb), nm in names.items():
                if ka in r["a"] and kb in r["b"]:
                    out["decomposition"][nm] = r
    tt = load(A / "paper_test_tables.json")
    if tt:
        for p in tt["datasets"]["reddit_test"]["pairs"]:
            nm = {("union", "singlethr"): "reward", ("singlethr", "single_batch"): "sampling",
                  ("single_batch", "single_group"): "scale", ("union", "single_group"): "total_registered",
                  ("union_group", "union"): "scale_union", ("union_group", "single_group"): "total"}.get((p["a"], p["b"]))
            if nm:
                out["decomposition"][f"{nm}_row"] = p["all"]["row"]
                out["decomposition"][f"{nm}_req"] = p["all"]["req"]
    # robustness of the primary comparison
    for s in SIZES:
        out["robust"].setdefault(s, {})["submission_per_row"] = pair(
            A / f"test_{s}_union_group_row_submission.json", TEST[s]["remark"], TEST[s]["rankgrpo"])
        ov = load(A / f"test_{s}_union_group_overlap_sensitivity.json")
        if ov:
            m = ov["comparison"]["excluding_grpo_overlap"]["context"]["R@20"]
            out["robust"][s]["overlap_excluded"] = [m["diff"], m["ci_low"], m["ci_high"]]
    for seed in ("s42", "s1234"):
        v = pair(A / f"test_2B_union_group_{seed}_pair_row.json", TEST["2B"][f"remark_{seed}"], TEST["2B"][f"rankgrpo_{seed}"])
        if v:
            out["robust"]["2B"][f"seed_{seed}"] = v
    for s in ("4B", "9B"):  # seed-42 pairs at the larger sizes
        if "remark_s42" in TEST[s] and "rankgrpo_s42" in TEST[s]:
            v = pair(A / f"test_{s}_union_group_s42_pair_row.json", TEST[s]["remark_s42"], TEST[s]["rankgrpo_s42"])
            if v:
                out["robust"][s]["seed_s42"] = v
    v = pair(A / "test_9B_union_group_s1234_pair_row.json", TEST["9B"]["remark_s1234"], TEST["9B"]["rankgrpo_s1234"])
    if v:
        out["robust"]["9B"]["seed_s1234"] = v
    v = pair(A / "test_4B_union_group_s1234_pair_row.json", TEST["4B"]["remark_s1234"], TEST["4B"]["rankgrpo_s1234"])
    if v:
        out["robust"]["4B"]["seed_s1234"] = v
    v = pair(A / "test_llama3b_s42_pair_row.json", "llama3b_union_group_s42", "llama3b_single_group_s42")
    if v:
        out["robust"].setdefault("llama3b", {})["seed_s42"] = v
    v = pair(A / "test_books4B_s42_pair_row.json", "reddit_books_4B_union_group_s42", "reddit_books_4B_single_batch_s42")
    if v:
        out["robust"].setdefault("books4B", {})["seed_s42"] = v
    # replicates (same configuration rerun, complete validation curves): difference against the original run
    for s, f in (("9B", "test_9B_rerun_vs_original_row.json"), ("2B", "test_2B_rerun4k_vs_original4k_row.json")):
        v = pair(A / f, "cap320", "p0c")
        if v:
            out["robust"].setdefault(s, {})["rerun_vs_original"] = v
    v = pair(A / "test_books_s42_pair_row.json", TEST["books"]["remark_s42"], TEST["books"]["perresp_s42"])
    if v:
        out["robust"].setdefault("books", {})["seed_s42"] = v
        out["robust"]["books"]["seed_s42_req"] = pair(A / "test_books_s42_pair_unit.json", TEST["books"]["remark_s42"],
                                                      TEST["books"]["perresp_s42"])
    v = pair(A / "test_books_s1234_pair_row.json", TEST["books"]["remark_s1234"], TEST["books"]["perresp_s1234"])
    if v:
        out["robust"].setdefault("books", {})["seed_s1234"] = v
        out["robust"]["books"]["seed_s1234_req"] = pair(A / "test_books_s1234_pair_unit.json", TEST["books"]["remark_s1234"],
                                                        TEST["books"]["perresp_s1234"])
    # robustness of the registered arm (kept for the appendix)
    reg = out["robust"]["registered"] = {}
    sub = load(A / "test_2B_submission_clustered.json")
    if sub:
        for x in sub["summary"]:
            if "union_s3407" in x["a"] and "single_group_s3407" in x["b"] and x["metric"] == "R@20":
                reg.setdefault("2B", {})["submission_" + x["weighting"]] = [x["diff"], *x["ci_submission"]]
    seeds = load(A / "seed_level_2B.json")
    if seeds:
        reg.setdefault("2B", {})["seeds"] = {"registered": seeds["remark_test_R20"], "rankgrpo": seeds["rankgrpo_test_R20"],
                                              "welch": [seeds["diff_of_means"], *seeds["welch_ci95"]]}
    v = pair(A / "test_4B_s42_row_context.json", TEST["4B"]["registered_s42"], TEST["4B"]["rankgrpo_s42"])
    if v:
        reg.setdefault("4B", {})["seed_s42"] = v
    # non-RL reference systems (GPT-5.6 zero-shot, E6 bge-kNN prior): ReMaRk minus each
    for ds, f in (("reddit", "n917_e5e6_reddit_union_group_{}.json"), ("books", "n917_e5e6_books_union_group_{}.json"),
                  ("music", "n917_e5e6_music_union_group_{}.json")):  # music: Table 2 references
        for w in ("row", "unit"):
            d = load(A / f.format(w))
            if d:
                for p in d["pairs"]:
                    key = p["a"].rsplit("/", 1)[-1] + " - " + p["b"].rsplit("/", 1)[-1]
                    out["baselines"].setdefault(ds, {}).setdefault(key, {})[w] = [p["R@20"][k] for k in ("diff", "ci_low", "ci_high")]
    return out


def extras() -> dict:
    """Readouts quoted in the text: dose-response ((2)), 4k-step pair, E2/E7/E9, top-k metrics (E6a)."""
    out = {}
    d = load(A / "dose_response.json")
    if d:
        out["dose"] = d
    for w in ("row", "unit"):
        d = load(A / f"val_4k_pair_{w}.json")
        if d:
            out[f"val_4k_{w}"] = [{"a": p["a"].rsplit("/", 1)[-1], "b": p["b"].rsplit("/", 1)[-1],
                                   "R@20": [p["R@20"][k] for k in ("a", "b", "diff", "ci_low", "ci_high")]}
                                  for p in d["pairs"]]
        d = load(A / f"n917_e2_{w}.json")
        if d:
            out[f"e2_{w}"] = [{"a": p["a"].rsplit("/", 1)[-1], "b": p["b"].rsplit("/", 1)[-1],
                               "R@20": [p["R@20"][k] for k in ("a", "b", "diff", "ci_low", "ci_high")]} for p in d["pairs"]]
    d = load(A / "n917_e7_judge.json")
    if d:
        out["e7_judge"] = {k: d[k] for k in ("a_run", "b_run", "remark_score", "ci95", "order_consistency",
                                             "requests_scored_both_orders")}
    d = load(REPO / "remark/outputs/books/audit/e9_recall.json")
    if d:
        out["e9_recall"] = {"strict": [d["recall"], *d["recall_ci95"]],
                            "normalized": [d["adjusted"]["recall"], *d["adjusted"]["recall_ci95"]]}
    d = load(A / "n917_e6_topk_metrics.json")
    if d:
        out["topk"] = {ds: {"pairs": d[ds]["pairs"], "arms": d[ds]["arms"]} for ds in ("reddit", "books") if ds in d}
    return out


def welch(a: list[float], b: list[float]) -> list[float]:
    """Difference of seed means with a Welch 95% CI (seeds as replicates)."""
    from scipy import stats
    a, b = np.array(a), np.array(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
    m = a.mean() - b.mean()
    return [float(m), float(m - h), float(m + h), float(df)]


def main() -> None:
    res = {"test": test_abs(), "val": val_curves(), "deltas": deltas(), "extras": extras()}
    t2 = res["test"]["2B"]
    if all(k in t2 for k in ("remark", "remark_s42", "rankgrpo", "rankgrpo_s42", "rankgrpo_s1234")):
        ug = [t2[k]["row"][0] for k in ("remark", "remark_s42", "remark_s1234") if k in t2]
        rg = [t2["rankgrpo"]["row"][0], t2["rankgrpo_s42"]["row"][0], t2["rankgrpo_s1234"]["row"][0]]
        res["deltas"]["robust"]["2B"]["seeds"] = {"remark": ug, "rankgrpo": rg, "welch": welch(ug, rg)[:3]}
    t9 = res["test"]["9B"]
    if all(k in t9 for k in ("remark", "remark_s42", "remark_s1234", "rankgrpo", "rankgrpo_s42", "rankgrpo_s1234")):
        ug = [t9[k]["row"][0] for k in ("remark", "remark_s42", "remark_s1234")]
        rg = [t9[k]["row"][0] for k in ("rankgrpo", "rankgrpo_s42", "rankgrpo_s1234")]
        res["deltas"]["robust"]["9B"]["seeds"] = {"remark": ug, "rankgrpo": rg, "welch": welch(ug, rg)[:3]}
    t4 = res["test"]["4B"]
    if all(k in t4 for k in ("remark", "remark_s42", "remark_s1234", "rankgrpo", "rankgrpo_s42", "rankgrpo_s1234")):
        ug = [t4[k]["row"][0] for k in ("remark", "remark_s42", "remark_s1234")]
        rg = [t4[k]["row"][0] for k in ("rankgrpo", "rankgrpo_s42", "rankgrpo_s1234")]
        res["deltas"]["robust"]["4B"]["seeds"] = {"remark": ug, "rankgrpo": rg, "welch": welch(ug, rg)[:3]}
    tb = res["test"]["books"]
    if all(k in tb for k in ("remark", "remark_s42", "remark_s1234", "perresp", "perresp_s42", "perresp_s1234")):
        ug = [tb[k]["row"][0] for k in ("remark", "remark_s42", "remark_s1234")]
        pb = [tb[k]["row"][0] for k in ("perresp", "perresp_s42", "perresp_s1234")]
        res["deltas"]["robust"]["books"]["seeds"] = {"remark": ug, "perresp": pb, "welch": welch(ug, pb)[:3]}
    t4 = res["test"]["4B"]
    if all(k in t4 for k in ("registered", "registered_s42", "rankgrpo", "rankgrpo_s42")):
        res["deltas"]["robust"]["registered"].setdefault("4B", {})["seeds"] = {
            "registered": [t4["registered"]["row"][0], t4["registered_s42"]["row"][0]],
            "rankgrpo": [t4["rankgrpo"]["row"][0], t4["rankgrpo_s42"]["row"][0]],
            "welch": welch([t4["registered"]["row"][0], t4["registered_s42"]["row"][0]],
                           [t4["rankgrpo"]["row"][0], t4["rankgrpo_s42"]["row"][0]])[:3]}
    res["missing"] = sorted(set(missing))
    (A / "paper_fig_stats.json").write_text(json.dumps(res, indent=1))
    print("wrote", A / "paper_fig_stats.json")
    if missing:
        print("missing (pending or not found):", *sorted(set(missing)), sep="\n  ")


if __name__ == "__main__":
    main()
