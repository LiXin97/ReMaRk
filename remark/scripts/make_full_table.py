"""Full test table of the paper (Table tab:main-full): per-row and per-request R@20 / N@20 and exact-20 for every arm.

Recomputes every absolute number from the run's per_instance.jsonl with the evaluation's own matcher (crs_eval.hits_for,
Rank-GRPO protocol, "truncate": mentioned or out-of-catalog items are misses), checks the per-row values against the
run's summary.json, and reads the paired differences with 95% CIs from the analysis files written by crs_compare.py
(row weighting = per row, unit weighting = per request; bootstrap over requests, 2,000 resamples, seed 42).
exact-20 is the row-weighted share of lists with at least 20 parsed items, as in crs_val_table.py (no list has more than
20 parsed items; the per-list share is also stored).

Usage: env/bin/python remark/scripts/make_full_table.py [--render_only]   (--render_only: re-typeset from the JSON)
  -> remark/outputs/crs/analysis/paper_full_table.json, remark/paper/tex/tables/tab_main_full.tex (Reddit-v2: references,
     Qwen3.5 2B-35B-A3B, stronger baselines at 2B/4B), tab_main_full_llama.tex (Reddit-v2, Llama-3.2-3B / 3.1-8B and the
     released checkpoint; split 10-09), tab_main_full_b.tex (Reddit-Books 2B/4B and
     music 2B with their references; Table 2 redesign), and
     tables/tab_books_val.tex (Books validation, from paper_books_val_table.json)
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import DATA, REPO, RG, hits_for, recall_ndcg  # noqa: E402
import paper_table_style as pts  # noqa: E402

C = REPO / "remark/outputs/crs"
T, A = C / "test", C / "analysis"
BOOKS = REPO / "remark/data/crs/reddit_books"
OUT_JSON = A / "paper_full_table.json"
OUT_TEX = REPO / "remark/paper/tex/tables/tab_main_full.tex"

MUSIC = REPO / "remark/data/crs/music"
OUT_TEX_B = OUT_TEX.parent / "tab_main_full_b.tex"  # second float: Reddit-Books and music (Table 2 redesign)
# 10-09 (author, appendix review): the Reddit-v2 table was taller than a page; split into Qwen3.5 (tab_main_full.tex, with the
# reference systems) and Llama (tab_main_full_llama.tex, with the released Llama-3.2-3B Rank-GRPO checkpoint)
OUT_TEX_L = OUT_TEX.parent / "tab_main_full_llama.tex"

# "remark" = the method (bounded sum, group scaling, request sampling); "registered" = its batch-scale variant, typeset as
# "bounded sum, batch" since (10-07, author: the full table matches Table 2's batch-scaling row). Row order in each block; an arm
# is typeset only where the setting has it. \best / \second rank every trained arm except plain SFT (stronger baselines
# included).
ARMS = [("sft", "SFT"), ("multipos", "SFT on pooled positives"),  # label as in Table 5 (10-09)
        ("dpo", "DPO, pooled positives"), ("rankgrpo", "Rank-GRPO"),
        ("g16", r"Rank-GRPO, $G{=}16$"), ("perresp", "per-responder, batch"), ("registered", "bounded sum, batch"),
        ("remark", r"\method{} (ours)")]
DATASET_OF = {"books": "books", "books4B": "books", "music": "music"}  # every other setting is Reddit-v2
BACKBONE = {"2B": "Qwen3.5-2B", "4B": "Qwen3.5-4B", "9B": "Qwen3.5-9B", "27B": "Qwen3.5-27B", "35BA3B": "Qwen3.5-35B-A3B",
            "llama3b": "Llama-3.2-3B", "llama8b": "Llama-3.1-8B", "books": "Qwen3.5-2B", "books4B": "Qwen3.5-4B",
            "music": "Qwen3.5-2B"}
RUNS = {
    "2B": {"sft": "test_2B_sft_step1500", "rankgrpo": "test_2B_single_group_s3407_step2000",
           "perresp": "test_2B_single_batch_s3407_step1750", "registered": "test_2B_union_s3407_step1500",
           "remark": "test_2B_union_group_step2000",
           # stronger baselines (selected on validation, tested once)
           "g16": "test_2B_single_group_s3407_p0c_G16_step2000", "multipos": "test_sft_multipos_2B_step1000",
           "dpo": "test_2B_dpo_union_lr5e-6_ep3_s3407_step250"},
    "4B": {"sft": "test_4B_sft_step1000", "rankgrpo": "test_4B_single_group_s3407_step2000",
           "perresp": "test_4B_single_batch_s3407_step2000", "registered": "test_4B_union_s3407_step2000",
           "remark": "test_4B_union_group_s3407_step2000",
           "g16": "test_4B_single_group_s3407_p0c_G16_step1750", "multipos": "test_sft_multipos_4B_step1500",
           "dpo": "test_4B_dpo_union_lr5e-6_ep3_s3407_step250"},
    "9B": {"sft": "test_9B_sft_step1000", "rankgrpo": "test_9B_single_group_s3407_step2000",
           "perresp": "test_9B_single_batch_s3407_step2000", "registered": "test_9B_union_s3407_step2000",
           "remark": "test_9B_union_group_s3407_p0c_step2000"},
    "27B": {"sft": "test_27B_sft_step1000", "rankgrpo": "test_27B_single_group_s3407_p0c_step2000",
            "remark": "test_27B_union_group_s3407_p0c_step2000",
            # 10-09: batch-scaled arms on the retrained SFT (qwen35_27B_r2 step 1500), s3407, locked @2000
            "perresp": "test_27B_single_batch_s3407_p0c_step2000",
            "registered": "test_27B_union_batch_s3407_p0c_step2000"},
    "35BA3B": {"sft": "test_35BA3B_sft_step1500", "rankgrpo": "test_35BA3B_single_group_s3407_p0c_step2000",
               "remark": "test_35BA3B_union_group_s3407_p0c_step2000",
               # 10-08: batch-scaled arms on the retrained SFT (qwen35_35BA3B_r2 step 1500), s3407, locked @2000
               "perresp": "test_35BA3B_single_batch_s3407_p0c_step2000",
               "registered": "test_35BA3B_union_batch_s3407_p0c_step2000"},
    # Llama-3.2-3B SFT = the released Rank-GRPO SFT checkpoint (evaluated under outputs/crs/, not test/)
    "llama3b": {"sft": "../sft_llama3b_test", "rankgrpo": "test_llama3b_single_group_s3407_p0c_step2000",
                "remark": "test_llama3b_union_group_s3407_p0c_step1750",
                # (10-07): batch-scaled arms, seed 3407, best-validation lock
                "perresp": "test_llama3b_single_batch_s3407_p0c_step2000",
                "registered": "test_llama3b_union_batch_s3407_p0c_step2000"},
    "llama8b": {"sft": "test_llama8b_sft_step1000", "rankgrpo": "test_llama8b_single_group_s3407_p0c_step2000",
                "remark": "test_llama8b_union_group_s3407_p0c_step1500",
                "perresp": "test_llama8b_single_batch_s3407_p0c_step2000",
                "registered": "test_llama8b_union_batch_s3407_p0c_step2000"},
    "books": {"sft": "test_books_sft_step1000", "rankgrpo": "test_books_single_group_p0d_step2000",
              "perresp": "test_books_single_batch_p0d_step2000", "registered": "test_books_union_p0d_step2000",
              "remark": "test_books_union_group_p0d_step1500"},
    "books4B": {"sft": "test_books4B_sft_step1500",
                "rankgrpo": "test_reddit_books_4B_single_group_s3407_p0c_cap384_ex_step2000",
                "perresp": "test_reddit_books_4B_single_batch_s3407_p0c_cap384_ex_step2000",
                "remark": "test_reddit_books_4B_union_group_s3407_p0c_cap384_ex_step2000",
                "registered": "test_reddit_books_4B_union_batch_s3407_p0c_cap384_ex_step1500"},
    "music": {"sft": "test_music_sft_step1500", "rankgrpo": "test_music_2B_single_group_s3407_p0c_ex_step2000",
              "remark": "test_music_2B_union_group_s3407_p0c_ex_step1250",
              "perresp": "test_music_2B_single_batch_s3407_p0c_ex_step2000",
              "registered": "test_music_2B_union_batch_s3407_p0c_ex_step1000"},
}
# paired differences of ReMaRk (row = per row, unit = per request; crs_compare.py): setting -> [(row file, unit file, base)]
# On Reddit-Books the better per-responder baseline on validation is the batch-scaled arm; both differences are shown.
PRIMARY = {"2B": [("test_2B_union_group_row.json", "test_2B_union_group_unit.json", "rankgrpo")],
           "4B": [("test_4B_union_group_row_context.json", "test_4B_union_group_unit_context.json", "rankgrpo")],
           "9B": [("test_9B_union_group_row_context.json", "test_9B_union_group_unit_context.json", "rankgrpo")],
           "27B": [("test_27B_pair_row.json", "test_27B_pair_unit.json", "rankgrpo")],
           "35BA3B": [("test_35BA3B_pair_row.json", "test_35BA3B_pair_unit.json", "rankgrpo")],
           # Llama-3B and music take the batch-scaled per-responder arm as the Delta baseline (higher validation best)
           "llama3b": [("test_llama3b_n934_vs_single_batch_row.json", "test_llama3b_n934_vs_single_batch_unit.json", "perresp"),
                       ("test_llama3b_union_group_row.json", "test_llama3b_union_group_unit.json", "rankgrpo")],
           "llama8b": [("test_llama8b_union_group_row.json", "test_llama8b_union_group_unit.json", "rankgrpo")],
           "books": [("test_books_union_group_row.json", "test_books_union_group_unit.json", "perresp"),
                     ("test_books_union_group_row.json", "test_books_union_group_unit.json", "rankgrpo")],
           "books4B": [("test_books4B_union_group_vs_single_batch_row.json",
                        "test_books4B_union_group_vs_single_batch_unit.json", "perresp"),
                       ("test_books4B_union_group_vs_single_group_row.json",
                        "test_books4B_union_group_vs_single_group_unit.json", "rankgrpo")],
           "music": [("test_music_n934_vs_single_batch_row.json", "test_music_n934_vs_single_batch_unit.json", "perresp"),
                     ("test_music_pair_row.json", "test_music_pair_unit.json", "rankgrpo")]}
REGISTERED = {"2B": ("test_2B_primary_row.json", "test_2B_primary_unit.json"),
              "4B": ("test_4B_primary_context_row.json", "test_4B_primary_context_unit.json"),
              "9B": ("test_9B_primary_context_row.json", "test_9B_primary_context_unit.json"),
              "books": ("test_books_primary_row.json", "test_books_primary_unit.json")}
# reference systems without RL training, per dataset (/E6; music and Books popularity ): key -> run dir
REFS = {"reddit": {"popularity": T / "popularity_train_top20", "knn": T / "test_knn_prior_k200",
                   "gpt56": T / "test_zeroshot_gpt56", "qwen27b": T / "test_zeroshot_qwen35_27B",
                   "released": C / "rankgrpo_llama3b_test"},
        "books": {"popularity": T / "popularity_train_top20_books", "knn": T / "test_knn_prior_k200_books",
                  "gpt56": T / "test_zeroshot_gpt56_books", "qwen27b": T / "test_zeroshot_qwen35_27B_books"},
        "music": {"popularity": T / "popularity_train_top20_music", "knn": T / "test_knn_prior_k200_music",
                  "gpt56": T / "test_zeroshot_gpt56_music",
                  # same dataset prompt plus a system message stating the year convention 
                  "gpt56_yearconv": T / "test_zeroshot_gpt56_music_yearconv"}}
REF_LABEL = {"popularity": "Popularity (train)", "knn": r"kNN prior ($k{=}200$)", "gpt56": "GPT-5.6 zero-shot",
             "qwen27b": "Qwen3.5-27B zero-shot", "gpt56_yearconv": "GPT-5.6, year convention stated"}
DATA_NAME = {"reddit": "Reddit-v2", "books": "Reddit-Books", "music": "Reddit-Music"}
# row-label marks explained in the caption: the 27B SFT checkpoint writes 19-line lists (exact-20 7.3%; author 2026-10-03)
LABEL_MARK = {("27B", "sft"): r"$^\ddagger$"}


def groups_of(ds) -> list[list[int]]:
    ctx = defaultdict(list)
    for i, p in enumerate(ds["prompt"]):
        ctx[p[0]["content"]].append(i)
    return list(ctx.values())


def run_stats(rd: Path, ds, catalog, groups) -> dict:
    r20, n20, full, over = [], [], [], []
    for r, it in zip(map(json.loads, open(rd / "per_instance.jsonl")), ds):
        recs = [(t, int(y)) for t, y in r["recs"]]
        full.append(len(recs) >= 20); over.append(len(recs) > 20)
        gts = [(g[0], int(g[1])) for g in it["groundtruth_with_release_year"]]
        hits, _ = hits_for(recs, gts, set(it["seen_titles"]), catalog, "truncate")
        rr, nn = recall_ndcg(hits, len(gts), 20)
        r20.append(rr); n20.append(nn)
    r20, n20 = np.array(r20), np.array(n20)
    out = {"run": rd.name, "row_R@20": float(r20.mean()), "row_N@20": float(n20.mean()),
           "req_R@20": float(np.mean([r20[g].mean() for g in groups])),
           "req_N@20": float(np.mean([n20[g].mean() for g in groups]))}
    # exact-20 per row, as in crs_val_table.py (Table tab:ablation), so the two tables agree; per list in *_per_list
    out["exact20"], out["over20"] = float(np.mean(full)), float(np.mean(over))
    out["exact20_per_list"] = float(np.mean([full[g[0]] for g in groups]))
    s = rd / "summary.json"
    if s.exists():
        sm = json.load(open(s))
        out["summary_R@20"], out["summary_N@20"] = sm["truncate_R@20"], sm["truncate_N@20"]
        assert abs(sm["truncate_R@20"] - out["row_R@20"]) < 1e-9, (rd, sm["truncate_R@20"], out["row_R@20"])
        assert abs(sm["truncate_N@20"] - out["row_N@20"]) < 1e-9, (rd, sm["truncate_N@20"], out["row_N@20"])
    return out


def delta(fname: str, a_run: str, b_run: str) -> dict:
    d = json.load(open(A / fname))
    for p in d["pairs"]:
        if p["a"].endswith(a_run) and p["b"].endswith(b_run):
            return {m: [p[m]["diff"], p[m]["ci_low"], p[m]["ci_high"]] for m in ("R@20", "N@20")}
    raise KeyError(f"{fname}: {a_run} - {b_run}")


def f4(x: float) -> str:
    return pts.num(x)  # R@20 / N@20 in %


def fd(v: list) -> str:
    s = lambda x: ("$-$" if x < 0 else "+") + f"{abs(x):.4f}".lstrip("0")
    return f"{s(v[0])} {{\\scriptsize[{s(v[1])},{s(v[2])}]}}"


def fd2(v: list, digits: int = 4) -> str:
    """Estimate over its CI on a second line (keeps columns narrow; no table shrinking)."""
    s = lambda x: ("$-$" if x < 0 else "+") + f"{abs(x):.{digits}f}".lstrip("0")
    c = lambda x: ("$-$" if x < 0 else "") + f"{abs(x):.3f}".lstrip("0")  # CI bounds: sign only when negative
    return rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}{s(v[0])}\\{{\scriptsize[{c(v[1])},{c(v[2])}]}}\end{{tabular}}"


def main() -> None:
    data = {"reddit": (DATA, RG / "reddit_catalog/gt_catalog.pkl"), "books": (BOOKS / "sft_dataset", BOOKS / "catalog.pkl"),
            "music": (MUSIC / "sft_dataset", MUSIC / "catalog.pkl")}
    dsets = {}
    for k, (d, c) in data.items():
        ds = load_from_disk(str(d / "test"))
        dsets[k] = (ds, set(tuple(x) for x in pickle.load(open(c, "rb"))), groups_of(ds))
    res: dict = {"note": __doc__.split("\n")[0], "settings": {}, "reference": {}, "datasets": {}, "missing": []}
    for k, (ds, _, grp) in dsets.items():
        res["datasets"][k] = {"rows": len(ds), "requests": len(grp)}
    for setting, arms in RUNS.items():
        ds, cat, grp = dsets[DATASET_OF.get(setting, "reddit")]
        st = res["settings"][setting] = {}
        for arm, run in arms.items():
            if (T / run / "per_instance.jsonl").exists():
                st[arm] = run_stats(T / run, ds, cat, grp)
            else:
                res["missing"].append(run)
        st["deltas"] = []
        for rowf, unitf, base in PRIMARY[setting]:
            dr, du = delta(rowf, arms["remark"], arms[base]), delta(unitf, arms["remark"], arms[base])
            st["deltas"].append({"vs": base, "row_R@20": dr["R@20"], "row_N@20": dr["N@20"],
                                 "req_R@20": du["R@20"], "req_N@20": du["N@20"]})
        if setting in REGISTERED:
            gr = delta(REGISTERED[setting][0], arms["registered"], arms[PRIMARY[setting][0][2]])
            gu = delta(REGISTERED[setting][1], arms["registered"], arms[PRIMARY[setting][0][2]])
            st["delta_registered"] = {"vs": PRIMARY[setting][0][2], "row_R@20": gr["R@20"], "row_N@20": gr["N@20"],
                                      "req_R@20": gu["R@20"], "req_N@20": gu["N@20"]}
        print(setting, {k: round(v["row_R@20"], 4) for k, v in st.items() if isinstance(v, dict) and "row_R@20" in v
                        and k != "delta_registered"})
    for dk, refs in REFS.items():
        ds, cat, grp = dsets[dk]
        for name, rd in refs.items():
            if (rd / "per_instance.jsonl").exists():
                res["reference"].setdefault(dk, {})[name] = run_stats(rd, ds, cat, grp)
            else:
                res["missing"].append(str(rd.relative_to(REPO)))
    if res["missing"]:
        print("MISSING (left out of the table):", res["missing"])
    OUT_JSON.write_text(json.dumps(res, indent=2))
    render(res)


def render(res: dict) -> None:
    """Typeset tab_main_full.tex (Reddit-v2) and tab_main_full_b.tex (Reddit-Books, music) from the computed numbers."""
    rf = lambda x: rf"\rf{{{x}}}"
    cols = ["row_R@20", "row_N@20", "req_R@20", "req_N@20"]
    ex = lambda v: f"{100 * v['exact20']:.1f}"

    def ref_rows(dk: str, released: bool = True) -> list[str]:
        ref = res["reference"].get(dk, {})
        L = []
        for name in ("popularity", "knn", "gpt56", "gpt56_yearconv", "qwen27b"):
            if name in ref:
                v = ref[name]
                L.append(f"-- & {REF_LABEL[name]} & " + " & ".join(rf(f4(v[c])) for c in cols) + " & -- \\\\")
        if dk == "reddit":
            L.append(r"-- & GPT-4o zero-shot$^\dagger$ & \rf{21.47} & \rf{11.97} & -- & -- & -- \\")
        if dk == "reddit" and released:
            rel = ref["released"]
            L.append(f"Llama-3.2-3B & Rank-GRPO, released (15.8k steps) & " + " & ".join(rf(f4(rel[c])) for c in cols)
                     + f" & {ex(rel)} \\\\")
        return L

    def block(setting: str) -> list[str]:
        st = res["settings"][setting]
        present = [(a, lab) for a, lab in ARMS if a in st]
        trained = [a for a, _ in present if a != "sft"]
        # rank at the displayed precision: arms that print the same value share the mark (10-09)
        mark = {c: pts.ranks({a: round(100 * st[a][c], 2) for a in trained}) for c in cols}
        tint = {"rankgrpo": r"\baserow ", "remark": r"\oursrow "}
        L = [r"\midrule"]
        for i, (arm, label) in enumerate(present):
            v = st[arm]
            cells = [pts.marked(f4(v[c]), mark[c].get(arm) if arm != "sft" else None) for c in cols]
            L.append(tint.get(arm, "") + (BACKBONE[setting] if i == 0 else "") + f" & {label}{LABEL_MARK.get((setting, arm), '')} & "
                     + " & ".join(cells)
                     + f" & {ex(v)} \\\\")
        for dl in st["deltas"]:
            vs = {"rankgrpo": "Rank-GRPO", "perresp": "per-responder, batch"}[dl["vs"]]
            L.append(rf" & \quad$\Delta$ vs.\ {vs} & " + " & ".join(pts.delta(dl[c]) for c in cols) + r" & \\")
        return L

    def header(dk: str, suffix: str = "") -> str:
        d = res["datasets"][dk]
        return rf"\multicolumn{{7}}{{@{{}}l}}{{\grp{{{DATA_NAME[dk]}{suffix}}} ({d['rows']:,} rows, {d['requests']:,} requests)}} \\"

    head = [r"% generated by remark/scripts/make_full_table.py from outputs/crs/test/*/per_instance.jsonl and the paired",
            r"% differences in outputs/crs/analysis (crs_compare.py); numbers in outputs/crs/analysis/paper_full_table.json",
            r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular}{@{}ll cc cc c@{}}", r"\toprule",
            r" & & \multicolumn{2}{c}{per row} & \multicolumn{2}{c}{per request} & \\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"\textbf{backbone} & \textbf{method} & R@20 & N@20 & R@20 & N@20 & exact-20 \\", r"\midrule"]
    tail = [r"\bottomrule", r"\end{tabular}"]
    A_ = [header("reddit", ", Qwen3.5")] + ref_rows("reddit", released=False)
    for s in ("2B", "4B", "9B", "27B", "35BA3B"):
        A_ += block(s)
    L_ = [header("reddit", ", Llama")]
    rel = res["reference"]["reddit"]["released"]
    L_.append(f"Llama-3.2-3B & Rank-GRPO, released (15.8k steps) & " + " & ".join(rf(f4(rel[c])) for c in cols)
              + f" & {ex(rel)} \\\\")
    for s in ("llama3b", "llama8b"):
        L_ += block(s)
    B_ = [header("books")] + ref_rows("books")
    for s in ("books", "books4B"):
        B_ += block(s)
    B_ += [r"\midrule", header("music")] + ref_rows("music") + block("music")
    OUT_TEX.write_text("\n".join(head + A_ + tail) + "\n")
    OUT_TEX_L.write_text("\n".join(head + L_ + tail) + "\n")
    OUT_TEX_B.write_text("\n".join(head + B_ + tail) + "\n")
    print("wrote", OUT_JSON, OUT_TEX, OUT_TEX_L, OUT_TEX_B)


def books_val_table() -> None:
    """Table tab:books from outputs/crs/analysis/paper_books_val_table.json (crs_val_table.py with
    configs/paper_books_val_table.json: P0d runs, each arm at its best validated step)."""
    d = json.load(open(A / "paper_books_val_table.json"))["datasets"]["books_p0d"]
    arms = d["arms"]
    # row numbers as in the Reddit-v2 ablation table (Table tab:ablation); arm 3 was not run on Reddit-Books
    rows = [(0, "sft", "SFT"), (1, "single_group", "Rank-GRPO"), (2, "single_batch", "per-resp., batch"),
            (4, r"consensusrw", r"mean, $\propto n$"), (5, "consensus", "mean, request"),
            (6, "union", r"bounded sum, batch"), (7, "union_group", r"\method{}")]
    num = {k: r for r, k, _ in rows}
    L = [r"% generated by remark/scripts/make_full_table.py from outputs/crs/analysis/paper_books_val_table.json",
         r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular}{@{}rl ccc@{}}", r"\toprule",
         r" & & R@20 & req.\ R@20 & R@20, $n{\ge}5$ \\", r"\midrule"]
    cols = ("R@20", "req_R@20", "R@20_n5")
    mark = {c: pts.ranks({k: arms[k][c] for _, k, _ in rows[1:]}) for c in cols}
    tint = {1: r"\baserow ", 7: r"\oursrow "}
    for r, k, lab in rows:
        v = arms[k]
        cells = [pts.marked(pts.num(v[c]), mark[c].get(k) if r else None) for c in cols]
        L.append(tint.get(r, "") + f"{r} & {lab} & " + " & ".join(cells) + " \\\\")  # exact-20 is in the P0 appendix text
    L.append(r"\midrule")
    for p in d["pairs"]:
        L.append(rf"\multicolumn{{2}}{{@{{}}l}}{{{num[p['a']]} $-$ {num[p['b']]}}} & {pts.delta(p['all']['row'])} & "
                 f"{pts.delta(p['all']['req'])} & {pts.delta(p['n>=5']['row'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT_TEX.parent / "tab_books_val.tex").write_text("\n".join(L) + "\n")
    print("wrote tab_books_val.tex")


if __name__ == "__main__":
    if "--render_only" in sys.argv:
        render(json.load(open(OUT_JSON)))
    else:
        main()
    books_val_table()
