"""Ablation tables of the ReMaRk paper (Qwen3.5-2B, Reddit-v2), generated from the analysis JSONs.

Inputs:  outputs/crs/analysis/paper_test_tables.json (test arms and paired differences; crs_test_tables.py),
         outputs/crs/analysis/paper_val_tables.json and paper_val_tables_ug.json (validation; row 7 from the latter).
Outputs: paper/tex/tables/tab_ablation_test.tex       main text: the 8 arms (differences: Figure 4 and the appendix),
         paper/tex/tables/tab_ablation_test_diffs.tex  appendix: every test difference quoted in the text,
         paper/tex/tables/tab_ablation_val.tex         appendix: validation arms (with exact-20) and differences.
Differences: paired bootstrap over requests (2,000 resamples, seed 42) as stored in the JSONs; a difference a - b that is
stored as b - a is negated (estimate and CI bounds swap sign, which is exact for percentile intervals).
Usage: python3 remark/scripts/make_ablation_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_table_style as pts  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
A = REPO / "remark/outputs/crs/analysis"
TAB = REPO / "remark/paper/tex/tables"

# row number -> (arm key, aggregation, scale, sampling)
ARMS = [(0, "sft", r"\multicolumn{3}{@{}l}{SFT (no RL)}", None, None),
        (1, "single_group", "per-responder", "group", "row (Rank-GRPO)"),  # method names: sampling column 
        (2, "single_batch", "per-responder", "batch", "row"),
        (3, "singlethr", "per-responder", "batch", "request"),
        (4, "consensusrw", "mean", "batch", "row"),  # same draw as rows 1-2 (was "$\propto n$")
        (5, "consensus", "mean", "batch", "request"),
        (6, "union", "bounded sum", "batch", "request"),
        (7, "union_group", "bounded sum", "group", r"request (\method)"),
        (8, "singlethr_group", "per-responder", "group", "request"),
        (9, "consensus_group", "mean", "group", "request")]  # , s3407 (added 10-09)
KEY = {r: k for r, k, *_ in ARMS}
# differences: (a row, b row, label)
MAIN_DIFFS = [(7, 1, r"\method{} $-$ Rank-GRPO"), (6, 3, "reward: marginalized"), (3, 2, "sampling: by request"),
              (7, 6, r"scale: group vs.\ batch")]
ALL_DIFFS = [(7, 1, r"\method{} vs.\ Rank-GRPO"), (7, 6, "scale, bounded sum"),
             (6, 1, r"bounded sum, batch scale vs.\ Rank-GRPO"), (4, 2, "reward, row sampling"),
             (5, 3, "reward, request sampling"), (6, 3, "reward, request sampling, bounded sum"),
             (5, 4, "sampling, mean"), (3, 2, "sampling, per-responder"),
             (6, 5, r"bounded sum vs.\ mean, request sampling"), (6, 4, "two estimators of the row objective"),
             (2, 1, "scale, per-responder"), (7, 5, r"\method{} vs.\ mean, request sampling"),
             (7, 8, "reward, group scale"), (8, 1, "sampling, group scale"), (8, 3, "scale, request sampling"),
             (7, 9, r"bounded sum vs.\ mean, group scale"), (9, 5, "scale, mean")]  # row 9 (10-09)
COLS = [("all", "row"), ("all", "req"), ("n=1", "row"), ("n>=5", "row")]
ARM_COLS = ["R@20", "req_R@20", "R@20_n1", "R@20_n5"]
HEAD = [r" & \textbf{aggregation} & \textbf{scale} & \textbf{sampling} & R@20 & req.\ R@20 & R@20, $n{=}1$ & R@20, $n{\ge}5$"]


def m(s: str) -> str:
    return s.replace("-", "$-$")


def stacked(v: list[float]) -> str:
    """Difference coloured by its CI, CI in grey on a second line (paper_table_style)."""
    return pts.delta(v)


def find(pairs: list[dict], a: str, b: str) -> dict | None:
    """The pair a - b, negating a stored b - a."""
    for p in pairs:
        if p["a"] == a and p["b"] == b:
            return {k: v for k, v in p.items() if k not in ("a", "b")}
        if p["a"] == b and p["b"] == a:
            neg = {}
            for k, v in p.items():
                if isinstance(v, dict):
                    neg[k] = {w: [-x[0], -x[2], -x[1]] for w, x in v.items()}
            return neg
    return None


def arm_rows(arms: dict, with_exact: bool) -> list[str]:
    trained = [k for r, k, *_ in ARMS if r > 0 and k in arms]
    # rank at the displayed precision, so arms that print the same value share the mark (rows 7 and 9, 10-09)
    mark = {c: pts.ranks({k: round(100 * arms[k][c], 2) for k in trained}) for c in ARM_COLS}
    tint = {1: r"\baserow ", 7: r"\oursrow "}
    out = []
    for r, k, agg, scale, samp in ARMS:
        if k not in arms:
            out.append(f"{r} & {agg} & & & " + " & ".join([r"\tbd"] * (len(ARM_COLS) + with_exact)) + r" \\")
            continue
        cells = [pts.marked(pts.num(arms[k][c]), mark[c].get(k) if r > 0 else None) for c in ARM_COLS]
        if with_exact:
            cells.append(f"{100 * arms[k]['exact20']:.1f}")
        lead = f"{r} & {agg}" if scale is None else f"{r} & {agg} & {scale} & {samp}"
        out.append(tint.get(r, "") + lead + " & " + " & ".join(cells) + r" \\")
    return out


def diff_rows(pairs: list[dict], diffs: list[tuple], ncols_extra: int = 0) -> list[str]:
    out = []
    for a, b, lab in diffs:
        p = find(pairs, KEY[a], KEY[b])
        cells = [stacked(p[g][w]) if p and g in p else r"\tbd" for g, w in COLS] + [""] * ncols_extra
        out.append(rf"\multicolumn{{4}}{{@{{}}l}}{{{lab} ({a}$-${b})}} & " + " & ".join(cells) + r" \\")
    return out


def write(name: str, lines: list[str], src: str) -> None:
    (TAB / name).write_text(f"% generated by remark/scripts/make_ablation_tables.py from {src}; do not edit by hand\n"
                            + "\n".join(lines) + "\n")
    print("wrote", name)


def main() -> None:
    t = json.load(open(A / "paper_test_tables.json"))["datasets"]["reddit_test"]
    # main text: the arms only. The differences of Figure 4 used to be a second block here; they are in the figure
    # and, in full, in tab_ablation_test_diffs.tex, and the space is needed for the 8-page limit (2026-10-02).
    L = [r"\renewcommand{\arraystretch}{1.1}", r"\begin{tabular}{@{}r lll cccc@{}}", r"\toprule", HEAD[0] + r" \\", r"\midrule"]
    L += arm_rows(t["arms"], with_exact=False)
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation_test.tex", L, "outputs/crs/analysis/paper_test_tables.json")
    # appendix: every test difference
    L = [r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular}{@{}r lll cccc@{}}", r"\toprule",
         r"\multicolumn{4}{@{}l}{difference (rows of Table~\ref{tab:ablation})} & R@20 & req.\ R@20 & R@20, $n{=}1$ & "
         r"R@20, $n{\ge}5$ \\", r"\midrule"]
    L += diff_rows(t["pairs"], ALL_DIFFS) + [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation_test_diffs.tex", L, "outputs/crs/analysis/paper_test_tables.json")
    # appendix: validation twin (row 7 and its pairs from paper_val_tables_ug.json)
    v = json.load(open(A / "paper_val_tables.json"))["datasets"]["reddit"]
    ug = json.load(open(A / "paper_val_tables_ug.json"))["datasets"]["reddit"]
    arms = dict(v["arms"]); arms["union_group"] = ug["arms"]["union_group"]
    for k in ("singlethr_group", "consensus_group"):  # row 8; row 9 (10-09)
        if k in ug["arms"]:
            arms[k] = ug["arms"][k]
    pairs = list(v["pairs"]) + list(ug["pairs"])
    L = [r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular}{@{}r lll cccc c@{}}", r"\toprule", HEAD[0] + r" & exact-20 \\", r"\midrule"]
    L += arm_rows(arms, with_exact=True)
    val_diffs = [d for d in ALL_DIFFS if find(pairs, KEY[d[0]], KEY[d[1]])]
    L += [r"\midrule"] + diff_rows(pairs, val_diffs, ncols_extra=1) + [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation_val.tex", L, "outputs/crs/analysis/paper_val_tables.json, paper_val_tables_ug.json")


if __name__ == "__main__":
    main()
