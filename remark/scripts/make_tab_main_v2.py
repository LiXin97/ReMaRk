"""Write the compact main results table (Table 2, tab:main) for the paper.

Author script (2026-10-02). Reads outputs/crs/analysis/paper_fig_stats.json (maintained by make_paper_stats.py) and,
while the music entries are not in it yet, outputs/crs/analysis/test_music_pair_{row,unit}.json. Writes
paper/tex/tables/tab_main_v2.tex. Per-row test R@20 for every backbone in one row block; Delta per row and per request
with paired-bootstrap 95% CIs; size-independent reference systems span each dataset's columns. Missing cells print as
"--" and are listed on stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "outputs/crs/analysis/paper_fig_stats.json"
MUSIC = ROOT / "outputs/crs/analysis/test_music_pair_{}.json"
# Reddit-Books popularity list on test, recomputed with the evaluation code (as in make_paper_figures.py)
BOOKS_POP = ROOT / "outputs/crs/analysis/n928_r7f_cited_stats.json"
OUT = ROOT / "paper/tex/tables/tab_main_v2.tex"

# (stats key, header label); grouped as Reddit-v2 Qwen3.5 | Reddit-v2 Llama | Books | Music
COLS = [("2B", "2B"), ("4B", "4B"), ("9B", "9B"), ("27B", "27B"), ("35BA3B", "35B-A3B"),
        ("llama3b", "3B"), ("llama8b", "8B"),
        ("books", "2B"), ("books4B", "4B"),
        ("music", "2B")]
ARMS = [("sft", "SFT"), ("rankgrpo", r"\baserow Rank-GRPO"), ("perresp", "Per-resp., batch"),
        ("remark", r"\oursrow \method{} (ours)")]
TRAINED = ["rankgrpo", "perresp", "remark"]  # SFT is the start, not ranked
REFS = [("popularity", "Popularity"), ("knn", "kNN prior"), ("gpt56", "GPT-5.6 zero-shot")]
SPANS = [("reddit", 7), ("books", 2), ("music", 1)]
# cells with a caption footnote: (dataset, reference) -> marker
NOTES: dict = {}  # no table footnote marks; music GPT-5.6 year convention in Appendix app:music, analysis/n928_music_year_diag.json

missing: list[str] = []

LOCKS = ROOT / "outputs/crs/locks"
# validation curves (every 250 steps) of the seed-3407 pair per column: Rank-GRPO, ReMaRk; ("val", key) reads
# paper_fig_stats.json val[key], ("lock", name) reads outputs/crs/locks/<name>.json
CURVES = {
    "2B": (("val", "2B", "rankgrpo"), ("val", "2B", "remark")),
    "4B": (("val", "4B", "rankgrpo"), ("val", "4B", "remark")),
    "9B": (("val", "9B", "rankgrpo"), ("val", "9B", "remark")),  # the tested run (first checkpoint 750)
    "27B": (("lock", "27B_single_group_s3407_p0c"), ("lock", "27B_union_group_s3407_p0c")),
    "35BA3B": (("lock", "35BA3B_single_group_s3407_p0c"), ("lock", "35BA3B_union_group_s3407_p0c")),
    "llama3b": (("lock", "llama3b_single_group_s3407_p0c"), ("lock", "llama3b_union_group_s3407_p0c")),
    "llama8b": (("lock", "llama8b_single_group_s3407_p0c"), ("lock", "llama8b_union_group_s3407_p0c")),
    "books": (("val", "books", "rankgrpo"), ("val", "books", "remark")),
    "books4B": (("lock", "reddit_books_4B_single_group_s3407_p0c_cap384_ex"),
                ("lock", "reddit_books_4B_union_group_s3407_p0c_cap384_ex")),
    "music": (("lock", "music_2B_single_group_s3407_p0c_ex"), ("lock", "music_2B_union_group_s3407_p0c_ex")),
}


def curve(st: dict, src: tuple) -> dict[int, float]:
    if src[0] == "val":
        c = st["val"][src[1]][src[2]]
    else:
        c = json.loads((LOCKS / f"{src[1]}.json").read_text())["val_R@20"]
    return {int(k): v for k, v in c.items()}


C1C2 = ROOT / "outputs/crs/analysis/n931_c1c2.json"
# column -> (n931_c1c2 setting, baseline): the Delta baseline of Table 2 (the per-responder arm better on validation;
# batch scaling for Llama-3B, Books and Music since )
C2KEY = {"2B": ("2B", "rankgrpo"), "4B": ("4B", "rankgrpo"), "9B": ("9B", "rankgrpo"), "27B": ("27B", "rankgrpo"),
         "35BA3B": ("35BA3B", "rankgrpo"), "llama3b": ("llama3b", "perresp_batch"), "llama8b": ("llama8b", "rankgrpo"),
         "books": ("books2B", "perresp_batch"), "books4B": ("books4B", "perresp_batch"), "music": ("music2B", "perresp_batch")}


def step_share(col: str) -> str:
    """Share of the baseline's 2,000 steps that ReMaRk needs to reach its step-2,000 validation R@20 (interpolated),
    with the request-bootstrap 95% CI (scripts/n931_c1c2.py)."""
    if col not in C2KEY:
        missing.append(f"share {col}")
        return "--"
    k, b = C2KEY[col]
    c = json.loads(C1C2.read_text())["settings"][k][b]["c2"]
    lo, hi = c["ratio_ci"]
    return rf"\dstack{{{c['ratio']:.2f}}}{{[{lo:.2f},{hi:.2f}]}}"


def steps_to_match(st: dict, col: str) -> str:
    """First validated step at which ReMaRk reaches Rank-GRPO's step-2,000 validation R@20 ("<=" if that is
    ReMaRk's first validated checkpoint and it is later than step 250)."""
    if col not in CURVES:
        missing.append(f"steps {col}")
        return "--"
    rg, rm = (curve(st, s) for s in CURVES[col])
    target = rg[2000]
    hit = [k for k in sorted(rm) if rm[k] >= target]
    if not hit:
        return "$>$2,000"
    k = hit[0]
    pre = r"$\le$" if k == min(rm) and k > 250 else ""
    return f"{pre}{k:,}"


def music_entry() -> tuple[dict, dict]:
    """Music test values and deltas from the pair files (a = ReMaRk, b = per-responder, group, row)."""
    test, delta = {}, {}
    for w, key in (("row", "row"), ("unit", "req")):
        p = json.loads(Path(str(MUSIC).format(w)).read_text())["pairs"][0]
        r = p["R@20"]
        test.setdefault("remark", {})[key] = [r["a"]]
        test.setdefault("rankgrpo", {})[key] = [r["b"]]
        delta[key] = [r["diff"], r["ci_low"], r["ci_high"]]
    return test, delta


# configuration fixed before any test run (bounded sum, batch scaling, request sampling; option A, user 2026-10-05):
# column -> (analysis file, ReMaRk run, baseline run = the Delta baseline of that column)
A = ROOT / "outputs/crs/analysis"
PREFIXED = {"2B": ("test_2B_primary_row.json", "test_2B_union_s3407_step1500", "test_2B_single_group_s3407_step2000"),
            "4B": ("test_4B_primary_context_row.json", "test_4B_union_s3407_step2000", "test_4B_single_group_s3407_step2000"),
            "9B": ("test_9B_primary_context_row.json", "test_9B_union_s3407_step2000", "test_9B_single_group_s3407_step2000"),
            "books": ("test_books_primary_row.json", "test_books_union_p0d_step2000", "test_books_single_batch_p0d_step2000"),
            # (10-07): union + batch scaling minus the column's Delta baseline (per-responder arm with the higher
            # validation best: batch for Llama-3B / Music / Books 4B, Rank-GRPO for Llama-8B)
            "llama3b": ("test_llama3b_n934_vs_single_batch_row.json", "test_llama3b_union_batch_s3407_p0c_step2000",
                        "test_llama3b_single_batch_s3407_p0c_step2000"),
            "llama8b": ("test_llama8b_n934_row.json", "test_llama8b_union_batch_s3407_p0c_step2000",
                        "test_llama8b_single_group_s3407_p0c_step2000"),
            "books4B": ("test_books4B_n934_vs_single_batch_row.json",
                        "test_reddit_books_4B_union_batch_s3407_p0c_cap384_ex_step1500",
                        "test_reddit_books_4B_single_batch_s3407_p0c_cap384_ex_step2000"),
            "music": ("test_music_n934_vs_single_batch_row.json", "test_music_2B_union_batch_s3407_p0c_ex_step1000",
                      "test_music_2B_single_batch_s3407_p0c_ex_step2000")}


# 27B (10-09, SFT r2 step 1500) and 35B-A3B batch cells (10-08, decision 10-07): batch arms start from a retrained SFT (r2 step 1500), so "Per-resp., batch"
# is the new-SFT per-responder batch arm (not eligible as the Delta baseline: different SFT start) and "Delta, batch scaling"
# is union batch minus the column's Delta baseline (Rank-GRPO), as in every other column (10-09, consistency fix)
# (both from outputs/crs/analysis/n936_tab5_35b.json, points x100).
T35 = json.loads((ROOT / "outputs/crs/analysis/n936_tab5_35b.json").read_text())["pairs"]


def prefixed_cell(col: str) -> str:
    if col in ("35BA3B", "27B"):
        x = T35[("35B" if col == "35BA3B" else "27B") + "/union_batch_minus_old_rankgrpo"]["row"]
        return dcell([x["diff"] / 100, x["ci_low"] / 100, x["ci_high"] / 100])
    if col not in PREFIXED:
        return "--"
    f, a, b = PREFIXED[col]
    for p in json.loads((A / f).read_text())["pairs"]:
        if p["a"].rstrip("/").endswith(a) and p["b"].rstrip("/").endswith(b):
            x = p["R@20"]
            return dcell([x["diff"], x["ci_low"], x["ci_high"]])
    missing.append(f"prefixed {col}")
    return "--"


def fmt(x: float) -> str:
    return f"{100 * x:.2f}"


def dcell(d: list[float] | None) -> str:
    if d is None:
        return "--"
    v, lo, hi = (100 * t for t in d)
    sign = "+" if v >= 0 else "$-$"
    tag = r"\up" if lo > 0 else r"\ns"
    lo_s = f"{lo:.2f}" if lo >= 0 else f"$-${-lo:.2f}"
    return rf"\dstack{{{tag}{{{sign}{abs(v):.2f}}}}}{{[{lo_s},{hi:.2f}]}}"


def main() -> None:
    st = json.loads(STATS.read_text())
    test, deltas = st["test"], st["deltas"]["primary"]
    if "music" not in test:
        mt, md = music_entry()
        test = dict(test, music=mt)
        deltas = dict(deltas, music=md)

    def val(col: str, arm: str) -> float | None:
        if col in ("35BA3B", "27B") and arm == "perresp":
            return T35[("35B" if col == "35BA3B" else "27B") + "/single_batch_minus_old_rankgrpo"]["row"]["a"] / 100
        v = test.get(col, {}).get(arm)
        if v is None or "row" not in v:
            return None
        return v["row"][0]

    lines = [
        "% generated by remark/scripts/make_tab_main_v2.py from outputs/crs/analysis/paper_fig_stats.json"
        " (+ test_music_pair_*.json); do not edit by hand",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\renewcommand{\dstack}[2]{\begin{tabular}[t]{@{}c@{}}#1\\[-1.5pt]{\scriptsize\color{refgray}#2}\end{tabular}}",
        r"\begin{tabular}{@{}l ccccc cc cc c@{}}",
        r"\toprule",
        r" & \multicolumn{7}{c}{Reddit-v2} & \multicolumn{2}{c}{Reddit-Books} & Music \\",
        r"\cmidrule(lr){2-8}\cmidrule(lr){9-10}\cmidrule(l){11-11}",
        r" & \multicolumn{5}{c}{Qwen3.5} & \multicolumn{2}{c}{Llama} & \multicolumn{2}{c}{Qwen3.5} & Qwen3.5 \\",
        r"\cmidrule(lr){2-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(l){11-11}",
        " & " + " & ".join(h for _, h in COLS) + r" \\",
        r"\midrule",
    ]
    # rank trained arms per column
    rank: dict[tuple[str, str], int] = {}
    for c, _ in COLS:
        vals = sorted(((val(c, a), a) for a in TRAINED if val(c, a) is not None), reverse=True)
        for i, (_, a) in enumerate(vals[:2]):
            rank[(c, a)] = i
    for arm, label in ARMS:
        cells = []
        for c, _ in COLS:
            v = val(c, arm)
            if v is None:
                if arm != "perresp":
                    missing.append(f"{c}/{arm}")
                cells.append("--")
                continue
            s = fmt(v)
            if rank.get((c, arm)) == 0:
                s = rf"\best{{{s}}}"
            elif rank.get((c, arm)) == 1:
                s = rf"\second{{{s}}}"
            cells.append(s)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace[1pt]")
    for key, label in (("row", r"$\Delta$ per row"), ("req", r"$\Delta$ per request")):
        cells = []
        for c, _ in COLS:
            d = deltas.get(c, {}).get(key)
            if d is None:
                missing.append(f"delta {c}/{key}")
            cells.append(dcell(d))
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.append(r"steps to match & " +" & ".join(step_share(c) for c, _ in COLS) + r" \\")
    lines.append(r"\addlinespace[2pt]")
    lines.append(r"$\Delta$, batch scaling & " + " & ".join(prefixed_cell(c) for c, _ in COLS) + r" \\")
    lines += [r"\midrule", r"\multicolumn{11}{@{}l}{\grp{Reference (no training)}} \\"]
    ref = {
        "reddit": {**test.get("baselines_reddit", {}), "popularity": test.get("reference", {}).get("popularity")},
        "books": dict(test.get("baselines_books", {})),
        "music": dict(test.get("baselines_music", {})),
    }
    ref["books"].setdefault("popularity", test.get("reference", {}).get("popularity_books"))
    ref["music"].setdefault("popularity", test.get("reference", {}).get("popularity_music"))
    if ref["books"].get("popularity") is None:
        bp = json.loads(BOOKS_POP.read_text())["stats"]["books_popularity_R@20"]["recomputed"]["test"]
        ref["books"]["popularity"] = {"row": [bp["per_row_R@20"]], "req": [bp["per_request_R@20"]]}
    for key, label in REFS:
        cells = []
        for ds, span in SPANS:
            v = ref[ds].get(key)
            if v is None:
                missing.append(f"ref {ds}/{key}")
                cells.append(rf"\multicolumn{{{span}}}{{c}}{{--}}")
                continue
            else:
                x = v["row"][0]
            mark = NOTES.get((ds, key), "")
            cells.append(rf"\multicolumn{{{span}}}{{c}}{{\rf{{{fmt(x)}}}{mark}}}" if span > 1 else rf"\rf{{{fmt(x)}}}{mark}")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")
    print("missing:", ", ".join(missing) if missing else "none")


if __name__ == "__main__":
    main()
