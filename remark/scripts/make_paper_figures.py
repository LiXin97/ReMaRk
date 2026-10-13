"""Experiments figures (vector PDF) and generated tables for the ReMaRk paper, from paper_fig_stats.json.

Run make_paper_stats.py first (env/bin/python; it needs `datasets`), then this script with the system python3 (it needs
matplotlib, which the project env lacks). Every value is read from the stats file; nothing is typed in here except the
GPT-4o zero-shot number quoted from the Rank-GRPO paper. Pending results are skipped; rerun both scripts when they land.

"ReMaRk" is the method (bounded sum, group scaling, request sampling). The key "registered" holds its batch-scale
variant (bounded sum, batch scaling), shown only where the advantage scale is ablated, under one label: "w/ batch scale".

Usage: python3 remark/scripts/make_paper_figures.py
Outputs: remark/paper/tex/figures/fig_{main,attrib,curves,val_arms,dose}.pdf,
         remark/paper/tex/tables/tab_{main,robust,topk}.tex
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
S = json.load(open(REPO / "remark/outputs/crs/analysis/paper_fig_stats.json"))
FIG = REPO / "remark/paper/tex/figures"
TAB = REPO / "remark/paper/tex/tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
TEXTW, COLW = 6.3, 3.03  # inches (ACL two-column)
# Reddit-Books popularity baseline, test R@20 per row / per request ((f): recomputed with the evaluation code's
# popularity definition, outputs/crs/analysis/n928_r7f_cited_stats.json; recorded 0.058 per row only)
_R7F = json.loads((REPO / "remark/outputs/crs/analysis/n928_r7f_cited_stats.json").read_text())["stats"]["books_popularity_R@20"]["recomputed"]["test"]
BOOKS_POP = (_R7F["per_row_R@20"], _R7F["per_request_R@20"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "legend.handletextpad": 0.4, "legend.columnspacing": 1.0, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
    "legend.frameon": False, "axes.titlepad": 4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# one colour per method in every figure (Okabe-Ito, colour-blind safe)
COL = {"sft": "#8C8C8C", "rankgrpo": "#0072B2", "perresp": "#E69F00", "remark": "#D55E00", "registered": "#E9A27C",
       "consensusrw": "#009E73", "consensus": "#CC79A7", "singlethr": "#56B4E9"}
# attribution components get colours no method uses (fig_attrib only)
COMP = {"total": "#262626", "reward": "#6A3D9A", "sampling": "#17757D", "scale": "#8A8A6E", "scale_union": "#B5B58F"}
MARK = {"sft": "s", "rankgrpo": "o", "perresp": "^", "remark": "D", "registered": "d"}
NAME = {"sft": "SFT", "rankgrpo": "Rank-GRPO", "perresp": "per-responder, batch", "remark": "ReMaRk (ours)",
        "registered": "ReMaRk w/ batch scale"}
SIZES = ["2B", "4B", "9B"]
T, D, VAL = S["test"], S["deltas"], S["val"]


def ci_err(v):
    return [[v[0] - v[1]], [v[2] - v[0]]]


def zero(ax):
    ax.axhline(0, color="#555555", lw=0.6, zorder=1)


def pct(ax, axis="y"):
    """Tick labels in % / percentage points (data stay in [0, 1])."""
    f = FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else f"{100 * v:.2f}".rstrip("0").rstrip("."))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


# ------------------------------------------------------------------ Figure: main results at a glance
# settings shown as rows of the forest plot (only those with results are drawn)
FOREST = [("2B", "Qwen3.5-2B"), ("4B", "Qwen3.5-4B"), ("9B", "Qwen3.5-9B"), ("27B", "Qwen3.5-27B"),
          ("35BA3B", "Qwen3.5-35B-A3B"), ("llama3b", "Llama-3.2-3B"), ("llama8b", "Llama-3.1-8B"),
          ("pooled", "pooled (2B, 4B, 9B)"), ("books", "Reddit-Books (2B)"), ("books4B", "Reddit-Books (4B)"),
          ("music", "Reddit-Music (2B)")]  # music added 10-07 (author: caption says "every setting")
REQ_COL = "#6B6B6B"


def fig_main() -> None:
    """(a) Forest plot of the gain over the better per-responder baseline, every setting and seed; (b) the gain by the
    request's number of responders."""
    rows = [(k, lab) for k, lab in FOREST if k in D["primary"]]
    fig, axs = plt.subplots(1, 2, figsize=(TEXTW, max(1.55, 0.45 + 0.23 * len(rows))),
                            gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.3})
    ax = axs[0]
    seeds = {k: [D["primary"][k]["row"][0]] + [v[0] for kk, v in D["robust"].get(k, {}).items()
                                                if kk.startswith("seed_s") and not kk.endswith("_req") and isinstance(v, list)]
             for k, _ in rows}
    for i, (k, lab) in enumerate(rows):
        p = D["primary"][k]; y = i
        ax.axhspan(y - 0.42, y + 0.42, color="#F7F7F7" if i % 2 == 0 else "white", zorder=0, lw=0)
        r = p["row"]
        ax.errorbar(r[0], y - 0.13, xerr=ci_err(r), fmt="D", color=COL["remark"], ms=3.4, lw=1.1, capsize=1.6, zorder=4)
        if len(seeds[k]) > 1:
            ax.plot(seeds[k], [y - 0.13] * len(seeds[k]), "o", ms=2.6, color=COL["remark"], alpha=0.45, mew=0, zorder=3)
        if p.get("req"):
            q = p["req"]
            ax.errorbar(q[0], y + 0.17, xerr=ci_err(q), fmt="o", color=REQ_COL, mfc="white", ms=3.0, lw=0.8, capsize=1.4,
                        zorder=4)
        ax.text(r[2] + 0.0012, y - 0.13, f"{100 * r[0]:+.2f}", va="center", ha="left", color=COL["remark"], fontsize=6.8,
                fontweight="bold", zorder=5)
    n_main = sum(1 for k, _ in rows if k not in ("pooled", "books", "books4B", "music"))
    if n_main < len(rows):
        ax.axhline(n_main - 0.5, color="#BBBBBB", lw=0.5, ls=(0, (3, 2)))
    ax.axvline(0, color="#555555", lw=0.7, zorder=1)
    ax.set_yticks(range(len(rows)), [lab for _, lab in rows]); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel(r"$\Delta$ test R@20 over the better per-responder baseline (points)", labelpad=1.5); pct(ax, "x")
    ax.set_title("(a) gain of ReMaRk (95% CI)", loc="left", pad=2)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5); ax.grid(axis="y", visible=False)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.004)  # room for the legend right of the Reddit-Books CIs
    ax.legend(handles=[Line2D([], [], color=COL["remark"], marker="D", ms=4, lw=1.3, label="per row (seed 3407)"),
                       Line2D([], [], color=COL["remark"], marker="o", ms=2.6, alpha=0.45, ls="none", label="other seeds"),
                       Line2D([], [], color=REQ_COL, marker="o", mfc="white", ms=3.6, lw=0.9, label="per request")],
              loc="lower right", borderaxespad=0.2, handlelength=1.2, labelspacing=0.15, framealpha=1.0, fontsize=6.6,
              borderpad=0.3)
    # (b) gain by responder count
    ax = axs[1]
    buckets = ["n=1", "n=2-4", "n=5-19", "n>=20"]
    shades = {"2B": "#FDAE6B", "4B": "#E6550D", "9B": "#8C2D04"}
    marks = {"2B": "o", "4B": "s", "9B": "D"}
    for j, s in enumerate(SIZES):
        b = D["buckets"].get(s)
        if not b:
            continue
        for i, k in enumerate(buckets):
            v = [b[k]["diff"], *b[k]["ci"]]
            ax.errorbar(i + (j - 1) * 0.24, v[0], yerr=ci_err(v), fmt=marks[s], color=shades[s], ms=3.0, lw=0.9,
                        capsize=1.3, label=s if i == 0 else None)
    zero(ax)
    ax.set_xticks(range(4), ["1", "2–4", "5–19", "≥20"])
    ax.set_xlabel("responders per request", labelpad=1.5)
    ax.set_ylabel(r"$\Delta$ vs. Rank-GRPO (points)"); pct(ax)
    ax.set_title("(b) gain by number of responders", loc="left", pad=2)
    ax.legend(loc="upper right", ncol=3, borderaxespad=0.2, handlelength=0.9, columnspacing=0.7, handletextpad=0.3,
              fontsize=6.6)
    fig.savefig(FIG / "fig_main.pdf"); plt.close(fig)


# ------------------------------------------------------------------ Figure: where the gain comes from
def _waterfall(ax, dec, w, title, labels=True):
    """Horizontal waterfall, top to bottom: Rank-GRPO (row 1) -> batch scale (row 2) -> request sampling (row 3) ->
    union reward (row 6) -> group scale (row 7) = ReMaRk. Each step: the paired difference with its 95% CI."""
    steps = [("scale", "+ batch scale"), ("sampling", "+ request sampling"), ("reward", "+ union reward"),
             ("scale_union", "+ group scale")]
    names = ["Rank-GRPO"] + [lab for _, lab in steps] + ["= ReMaRk"]
    ax.plot([0], [0], marker="D", ms=4, color=COL["rankgrpo"], zorder=4)
    level = 0.0
    for i, (k, _) in enumerate(steps, start=1):
        v = dec[f"{k}_{w}"]; d = v[0]
        ax.plot([level, level], [i - 1 + 0.32, i - 0.32], color="#888888", lw=0.6, ls=(0, (2, 1.5)), zorder=2)
        ax.barh(i, d, left=level, height=0.56, color=COMP[k] if d >= 0 else "#B5B5B5", edgecolor="none", zorder=3)
        ax.errorbar(level + d, i, xerr=[[d - v[1]], [v[2] - d]], color="#333333", lw=0.7, capsize=1.5, zorder=4)
        ax.text(max(level + v[2], level) + 0.0015, i, f"{100 * d:+.2f}", va="center", ha="left", fontsize=7)
        level += d
    n = len(steps) + 1; tot = dec[f"total_{w}"]
    ax.plot([level, level], [n - 1 + 0.32, n - 0.32], color="#888888", lw=0.6, ls=(0, (2, 1.5)), zorder=2)
    ax.barh(n, tot[0], height=0.6, color=COL["remark"], zorder=3)
    ax.errorbar(tot[0], n, xerr=[[tot[0] - tot[1]], [tot[2] - tot[0]]], color="#333333", lw=0.7, capsize=1.5, zorder=4)
    ax.text(tot[2] + 0.0015, n, f"{100 * tot[0]:+.2f}", va="center", ha="left", fontsize=7.5, fontweight="bold",
            color=COL["remark"])
    ax.axvline(0, color="#555555", lw=0.6, zorder=1)
    ax.set_yticks(range(n + 1), names if labels else [""] * (n + 1)); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    pct(ax, "x"); ax.set_title(title, loc="left", pad=2)
    ax.grid(axis="x", color="#E6E6E6", lw=0.5); ax.grid(axis="y", visible=False)


def fig_attrib() -> None:
    dec = D["decomposition"]
    need = ["total", "reward", "sampling", "scale_union", "scale"]
    if not all(f"{k}_{w}" in dec for k in need for w in ("row", "req")):
        print("fig_attrib: decomposition incomplete, skipped"); return
    fig, axs = plt.subplots(1, 3, figsize=(TEXTW, 1.6), gridspec_kw={"width_ratios": [1.15, 0.85, 1.0], "wspace": 0.12})
    _waterfall(axs[0], dec, "row", "(a) per row")
    _waterfall(axs[1], dec, "req", "(b) per request", labels=False)
    lo = min(axs[0].get_xlim()[0], axs[1].get_xlim()[0]); hi = max(axs[0].get_xlim()[1], axs[1].get_xlim()[1])
    for a in axs[:2]:
        a.set_xlim(lo, hi + 0.006)
        a.set_xlabel(r"$\Delta$ test R@20 (points, 2B)", labelpad=1.5)
    ax = axs[2]
    buckets = ["n=1", "n=2-4", "n=5-19", "n>=20"]
    for j, (nm, c, m, lab) in enumerate((("reward", COMP["reward"], "D", "union reward"),
                                          ("sampling", COMP["sampling"], "o", "request sampling"))):
        b = dec.get(nm)
        if not b:
            continue
        for i, k in enumerate(buckets):
            v = [b[k]["diff"], *b[k]["ci"]]
            ax.errorbar(i + (j - 0.5) * 0.28, v[0], yerr=ci_err(v), fmt=m, color=c, ms=3.0, lw=0.9, capsize=1.3,
                        label=lab if i == 0 else None)
    zero(ax)
    ax.set_xticks(range(4), ["1", "2–4", "5–19", "≥20"])
    ax.set_xlabel("responders per request"); pct(ax)
    ax.yaxis.tick_right(); ax.yaxis.set_label_position("right"); ax.set_ylabel("points, per row")
    ax.spines["right"].set_visible(True); ax.spines["left"].set_visible(False)
    ax.set_title("(c) by number of responders", loc="left", pad=2)
    ax.set_xlabel("responders per request", labelpad=1.5)
    ax.legend(loc="upper right", borderaxespad=0.2, handlelength=0.9, labelspacing=0.15, fontsize=6.6)
    fig.savefig(FIG / "fig_attrib.pdf"); plt.close(fig)


# ------------------------------------------------------------------ Figure: validation curves per setting
def curve(ax, setting: str, arm: str, **kw) -> None:
    v = VAL.get(setting, {})
    if arm not in v:
        return
    xs, ys = [0] + [int(k) for k in v[arm]], [v.get("sft")] + list(v[arm].values())
    # start at the SFT point only if the run's first checkpoint (step 250) was validated; a run resumed after a
    # preemption whose early checkpoints were lost (9B ReMaRk: first validated step 750) starts at its first point
    if ys[0] is None or xs[1] > 250:
        xs, ys = xs[1:], ys[1:]
    ax.plot(xs, ys, **kw)


def fig_curves() -> None:
    titles = {"2B": "Reddit-v2, 2B", "4B": "Reddit-v2, 4B", "9B": "Reddit-v2, 9B", "books": "Reddit-Books, 2B"}
    long4 = "remark_4k" in VAL.get("4B", {})  # 4B 4,000-step pair
    # left/right: the axes fill the text width (the one-row legend no longer sets the cropped width)
    long8 = "remark_8k" in VAL.get("2B", {})  # 2B pair trained to the registered plateau
    s42 = "remark_8k_s42" in VAL.get("2B", {})  # second 2B plateau pair, seed 42: its own panel
    fig, axs = plt.subplots(1, 5 if s42 else 4, figsize=(TEXTW, 1.85),
                            gridspec_kw={"width_ratios": [2.1 if long8 else 1.5] + ([1.7] if s42 else [])
                                         + [2.0 if "remark_8k" in VAL.get("4B", {}) else 1.5 if long4 else 1,
                                            1.8 if "remark_8k" in VAL.get("9B", {}) else 1,
                                            1.8 if "remark_8k" in VAL.get("books", {}) else 1],
                                         "wspace": 0.36,
                                         "left": 0.065, "right": 0.995})
    if s42:  # panel: seed-42 plateau pair only; the y-axis is clipped, Rank-GRPO's collapse is annotated
        ax42, axs = axs[1], [axs[0]] + list(axs[2:])
        for arm in ("rankgrpo_8k_s42", "remark_8k_s42"):
            c = COL[arm.split("_")[0]]
            curve(ax42, "2B", arm, color=c, ls=(0, (1, 1.1)), lw=1.0)
            k, v = max(VAL["2B"][arm].items(), key=lambda kv: (kv[1], -int(kv[0])))
            ax42.plot(int(k), v, marker="*", ms=6.5, color=c, mec="black", mew=0.4, ls="none", zorder=5)
        lo = min(v for a in ("rankgrpo_8k_s42", "remark_8k_s42") for k, v in VAL["2B"][a].items() if v > 0.15) - 0.006
        hi = max(v for a in ("rankgrpo_8k_s42", "remark_8k_s42") for v in VAL["2B"][a].values()) + 0.005
        pct(ax42); ax42.set_ylim(lo, hi)
        rg = VAL["2B"]["rankgrpo_8k_s42"]
        drop = [int(k) for k, v in rg.items() if v < lo]
        if drop:
            ax42.annotate(f"collapse ({100 * min(rg[str(d)] for d in drop):.1f})", xy=(drop[0], lo + 0.001),
                          xytext=(drop[0] - 3600, lo + 0.009), fontsize=7, color=COL["rankgrpo"],
                          arrowprops=dict(arrowstyle="->", color=COL["rankgrpo"], lw=0.6))
        ax42.set_xticks([0, 2000, 4000, 6000, 8000], ["0", "2k", "4k", "6k", "8k"])
        ax42.set_title("2B, seed 42", loc="left")  # full "Reddit-v2, 2B, seed 42" ran into the 4B title; ax42.set_xlabel("RL step")
    for ax, s in zip(axs, ["2B", "4B", "9B", "books"]):
        # the per-responder batch-scale arm is not drawn: it overlaps Rank-GRPO and the text does not discuss it
        # (writing review 2026-10-02, item 10); it stays in the main table and in the appendix figure
        for arm in ("rankgrpo", "remark"):
            curve(ax, s, arm, color=COL[arm], marker=MARK[arm], ms=2.6, lw=1.0 if arm != "remark" else 1.5,
                  label=NAME[arm])
        # reruns of the same configuration: complete curves where a preemption cost the original its early
        # checkpoints (9B ReMaRk starts at 750, the 2B 4k run at 1750)
        curve(ax, s, "remark_rerun", color=COL["remark"], lw=0.8, alpha=0.55)
        if s == "2B" or (s == "4B" and long4):
            curve(ax, s, "rankgrpo_4k", color=COL["rankgrpo"], ls=(0, (3, 2)), lw=0.9,
                  label="Rank-GRPO, 4k steps" if s == "2B" else None)
            if "remark_4k" not in VAL.get(s, {}):
                curve(ax, s, "registered_4k", color=COL["registered"], ls=(0, (3, 2)), lw=0.9,
                      label="ReMaRk w/ batch scale, 4k steps")
            curve(ax, s, "remark_4k", color=COL["remark"], ls=(0, (3, 2)), lw=0.9,
                  label="ReMaRk, 4k steps" if s == "2B" else None)
            # reruns are explained in the caption ("Faint: ..."), not in the legend, which must fit on one row
            curve(ax, s, "remark_rerun_4k", color=COL["remark"], ls=(0, (3, 2)), lw=0.8, alpha=0.55)
            best = max(VAL[s]["remark"].values())
            ax.axhline(best, color=COL["remark"], lw=0.5, ls=":")
            ax.set_xticks([0, 1000, 2000, 3000, 4000], ["0", "1k", "2k", "3k", "4k"])
            if "remark_8k" in VAL.get(s, {}):  # 2B: ; 4B: 
                # both arms to the plateau (stop after 1,500 steps without a new best); star = selected step
                for arm, lab in (("rankgrpo_8k", "Rank-GRPO, to plateau"), ("remark_8k", "ReMaRk, to plateau")):
                    c = COL[arm.split("_")[0]]
                    curve(ax, s, arm, color=c, ls=(0, (1, 1.1)), lw=1.0, label=lab if s == "2B" else None)
                    if arm in VAL[s]:
                        k, v = max(VAL[s][arm].items(), key=lambda kv: (kv[1], -int(kv[0])))
                        ax.plot(int(k), v, marker="*", ms=6.5, color=c, mec="black", mew=0.4, ls="none", zorder=5)
                ax.set_xticks([0, 2000, 4000, 6000], ["0", "2k", "4k", "6k"])
        elif s in ("9B", "books") and "remark_8k" in VAL.get(s, {}):  # (Books), (9B): plateau pairs
            plateau_curves(ax, s)
            if s == "9B":  # narrow panel: 2k-spaced tick labels to 8k run together
                ax.set_xticks([0, 4000, 8000], ["0", "4k", "8k"])
        else:
            ax.set_xticks([0, 1000, 2000], ["0", "1k", "2k"])
        ax.set_title(titles[s], loc="left"); ax.set_xlabel("RL step"); pct(ax)
        if "sft" in VAL.get(s, {}):
            ax.plot(0, VAL[s]["sft"], marker=MARK["sft"], color=COL["sft"], ms=3, ls="none", zorder=4)
    axs[0].set_ylabel("validation R@20 (%)")
    h, l = axs[0].get_legend_handles_labels()  # legend is taken from the 2B panel
    h.append(Line2D([], [], color=COL["sft"], marker="s", ls="none", ms=3)); l.append("SFT (step 0)")
    # two legend rows: a one-row legend made the cropped PDF ~545pt wide, so \textwidth scaled every font to ~84%
    fig.legend(h, l, loc="lower center", ncol=(len(h) + 1) // 2, bbox_to_anchor=(0.5, 0.99), handlelength=1.8,
               columnspacing=1.0, borderaxespad=0.0)
    fig.savefig(FIG / "fig_curves.pdf"); plt.close(fig)


def plateau_curves(ax, s: str) -> None:
    """Both arms trained to the registered plateau (dotted; star = selected step), as in the 2B/4B panels."""
    for arm in ("rankgrpo_8k", "remark_8k"):
        c = COL[arm.split("_")[0]]
        curve(ax, s, arm, color=c, ls=(0, (1, 1.1)), lw=1.0)
        if arm in VAL[s]:
            k, v = max(VAL[s][arm].items(), key=lambda kv: (kv[1], -int(kv[0])))
            ax.plot(int(k), v, marker="*", ms=6.5, color=c, mec="black", mew=0.4, ls="none", zorder=5)
    hi = max(int(k) for a in ("rankgrpo_8k", "remark_8k") for k in VAL[s].get(a, {}))
    ticks = [t for t in (0, 2000, 4000, 6000, 8000) if t <= hi + 500]
    ax.set_xticks(ticks, [f"{t // 1000}k" if t else "0" for t in ticks])


def fig_curves_music() -> None:
    """Appendix: music 2B, the 2,000-step pair and both arms trained to the registered plateau."""
    if "remark_8k" not in VAL.get("music", {}):
        return
    # width chosen so the cropped PDF is about one column wide (fonts print at their set size, as in fig_curves)
    fig, ax = plt.subplots(figsize=(3.36, 1.5))
    for arm in ("rankgrpo", "remark"):
        curve(ax, "music", arm, color=COL[arm], marker=MARK[arm], ms=2.4, lw=1.0 if arm != "remark" else 1.5, label=NAME[arm])
    for arm, lab in (("rankgrpo_8k", "Rank-GRPO, to plateau"), ("remark_8k", "ReMaRk, to plateau")):
        curve(ax, "music", arm, color=COL[arm.split("_")[0]], ls=(0, (1, 1.1)), lw=1.0, label=lab)
    plateau_curves(ax, "music")
    if "sft" in VAL["music"]:
        ax.plot(0, VAL["music"]["sft"], marker=MARK["sft"], color=COL["sft"], ms=3, ls="none", zorder=4)
    ax.set_xlabel("RL step"); ax.set_ylabel("validation R@20, % (music)"); pct(ax)
    fig.legend(*ax.get_legend_handles_labels(), ncol=2, loc="lower center", bbox_to_anchor=(0.55, 0.99),
               handlelength=1.8, columnspacing=0.8, labelspacing=0.2, fontsize=7, borderaxespad=0.0)
    fig.savefig(FIG / "fig_curves_music.pdf"); plt.close(fig)


def fig_val_arms() -> None:
    """Appendix: the 2B ablation arms on validation (row numbers as in the ablation table)."""
    arms = [("rankgrpo", "Rank-GRPO (1)", "-", "o"), ("perresp", "per-resp., row (2)", "--", "^"),
            ("singlethr", "per-resp., request (3)", "--", "v"), ("consensusrw", "mean, row (4)", "-", "s"),
            ("consensus", "mean, request (5)", "-", "P"), ("registered", "bounded sum, batch (6)", "--", "d"),
            ("remark", "ReMaRk (7)", "-", "D")]
    fig, ax = plt.subplots(figsize=(COLW, 2.0))
    for arm, lab, ls, m in arms:
        curve(ax, "2B", arm, color=COL[arm], ls=ls, marker=m, ms=2.4, lw=1.5 if arm == "remark" else 0.9, label=lab)
    ax.set_xlabel("RL step"); ax.set_ylabel("validation R@20, % (2B)"); pct(ax)
    ax.set_xticks([0, 500, 1000, 1500, 2000], ["0", "500", "1k", "1.5k", "2k"])
    fig.legend(*ax.get_legend_handles_labels(), ncol=2, loc="lower center", bbox_to_anchor=(0.55, 0.99),
               handlelength=1.8, columnspacing=0.8, labelspacing=0.2, fontsize=7, borderaxespad=0.0)
    fig.savefig(FIG / "fig_val_arms.pdf"); plt.close(fig)


def fig_dose() -> None:
    """Dose-response ((2)): validation R@20 against the share of responder noise removed, 1 - rho(m)."""
    d = S.get("extras", {}).get("dose")
    if not d:
        print("fig_dose: dose_response.json missing, skipped"); return
    import numpy as np
    rho = d["rho"]; ms = ["1", "2", "4", "all"]
    fig, axs = plt.subplots(1, 2, figsize=(COLW, 1.75), gridspec_kw={"wspace": 0.5})
    for ax, key, title in ((axs[0], "best/all", "(a) all requests"), (axs[1], "best/n=1", "(b) single-responder")):
        r = d["readouts"][key]
        xs = np.array([1 - rho[m] for m in ms]); mean = np.array([r["seed_means"][m] for m in ms])
        for m in ms:
            pts = [v for k, v in r["per_seed"].items() if k.split("/")[0] == m]
            ax.plot([1 - rho[m]] * len(pts), pts, ls="none", marker="o", ms=2.4, mfc="white", mec=COL["consensusrw"],
                    mew=0.7, zorder=3)
        ax.plot(xs, mean, ls="none", marker="D", ms=3.6, color=COL["consensusrw"], zorder=4)
        b = r["slope"]; a0 = mean.mean() - b * xs.mean(); gx = np.array([0, 1])
        ax.plot(gx, a0 + b * gx, color=COL["consensusrw"], lw=0.9, zorder=2)
        lo, hi = r["slope_ci"]
        ax.text(0.03, 0.97, f"slope {100 * b:+.2f}\n[{100 * lo:+.2f}, {100 * hi:+.2f}]".replace("-", "\u2212"), transform=ax.transAxes,
                va="top", fontsize=7)
        ax.set_title(title, loc="left"); pct(ax)
        ax.set_xticks([1 - rho[m] for m in ms], [f"{m}" for m in ms])
        # m=4 and m=all sit 0.16 apart: nudge their labels apart so they do not touch
        from matplotlib.transforms import ScaledTranslation
        for lab, dx in zip(ax.get_xticklabels()[2:], (-1.5, 2.5)):  # points
            lab.set_transform(lab.get_transform() + ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans))
        ax.set_xlim(-0.08, 1.08)
        ax.margins(y=0.18)
    axs[0].set_ylabel("validation R@20, % (2B)")
    fig.supxlabel(r"responders in the reward, $m$ (spaced by noise removed, $1-\rho(m)$)", fontsize=7.5, y=-0.1)
    fig.savefig(FIG / "fig_dose.pdf"); plt.close(fig)


# ------------------------------------------------------------------ tables
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_table_style as pts  # noqa: E402


def f4(x):
    return pts.num(x)  # R@20 in %


def dci(v, digits=4):
    if not v:
        return r"\tbd"
    short = lambda x: f"{x:+.3f}".replace("0.", ".", 1)
    return (f"{v[0]:+.{digits}f}" + r" {\scriptsize[" + short(v[1]) + "," + short(v[2]) + "]}").replace("-", "$-$")


def dci2(v, digits=2):
    """Paired difference (percentage points) with its CI on a second line, coloured by the CI (paper_table_style)."""
    return pts.delta(v)


def tab_main() -> None:
    """Main table (full width), two blocks: (top) the trained Qwen3.5 arms, columns = metric x dataset x backbone size;
    (bottom) reference systems, which have no Qwen3.5 size, under their own header. R@20 in %, per row and per request."""
    sets = ["2B", "4B", "9B", "books"] + (["books4B"] if "books4B" in T else [])
    nb = len(sets) - 3  # Reddit-Books columns (2B, and 4B once tested)
    ws = ["row", "req"]
    bl = {"reddit": T.get("baselines_reddit", {}), "books": T.get("baselines_books", {})}
    ref = T.get("reference", {})
    rf = lambda x: x if x in ("--", r"\tbd") else rf"\rf{{{x}}}"

    def get(d, k, w):
        return f4(d[k][w][0]) if k in d and w in d[k] else "--"

    def ref_row(label, cells):  # cells: [(reddit, books) per metric]
        out = []
        for reddit, books in cells:
            out += [rf"\multicolumn{{3}}{{c}}{{{rf(reddit)}}}", rf"\multicolumn{{{nb}}}{{c}}{{{rf(books)}}}" if nb > 1 else rf(books)]
        return f"{label} & " + " & ".join(out) + r" \\"

    k = len(sets)  # columns per block
    bh = rf"\multicolumn{{{nb}}}{{c}}{{Reddit-Books}}" if nb > 1 else "Reddit-Books"
    sizes = " & ".join(["2B", "4B", "9B", "2B", "4B"][:k])
    # CI sub-rows of this table in refgray, darker than main.tex's \dstack (cigray), for print. Local to the table*
    # float that \input's this file, so the other generated tables keep the preamble's \dstack. The size stays
    # \scriptsize (7 pt in acl.sty): a \footnotesize (9 pt) CI is 39.7 pt wide ([-0.15,1.52]: 46.9 pt), so ten CI
    # columns plus the label column exceed \textwidth even at \tabcolsep = 0 (measured 2026-10-02; at 4 pt the table
    # is 89 pt overfull).
    dstack_main = (r"\renewcommand{\dstack}[2]{\begin{tabular}[t]{@{}c@{}}#1\\[-1.5pt]{\scriptsize\color{refgray}#2}"
                   r"\end{tabular}}")
    # the stronger-baseline label Rank-GRPO_{G=16} is ~9.5 pt wider than the old widest label; 3.6 pt column spacing
    # (main.tex: 4 pt; local to the table* float) keeps the table at \textwidth (measured 2026-10-02)
    lines = [r"\renewcommand{\arraystretch}{1.12}", r"\setlength{\tabcolsep}{3.6pt}", dstack_main,
             r"\begin{tabular}{@{}l " + "c" * k + " " + "c" * k + "@{}}", r"\toprule",
             rf" & \multicolumn{{{k}}}{{c}}{{\textbf{{R@20 per row}} (primary)}} & \multicolumn{{{k}}}{{c}}{{\textbf{{R@20 per request}}}} \\",
             rf"\cmidrule(lr){{2-{k + 1}}}\cmidrule(l){{{k + 2}-{2 * k + 1}}}",
             rf" & \multicolumn{{3}}{{c}}{{Reddit-v2}} & {bh} & \multicolumn{{3}}{{c}}{{Reddit-v2}} & {bh} \\",
             rf"\cmidrule(lr){{2-4}}\cmidrule(lr){{5-{k + 1}}}\cmidrule(lr){{{k + 2}-{k + 4}}}\cmidrule(l){{{k + 5}-{2 * k + 1}}}",
             rf"\grp{{Qwen3.5 + RL}} & {sizes} & {sizes} \\", r"\midrule"]
    # rows (2B and 4B only; "--" elsewhere): multi-positive baselines and Rank-GRPO with G = 16
    extra = {"multipos", "dpo", "g16"}
    arms = [("sft", "SFT"), ("rankgrpo", "Rank-GRPO"), ("perresp", "Per-resp., batch"), ("remark", r"\method{} (ours)")]
    extra_arms = [("g16", r"Rank-GRPO$_{G=16}$"), ("multipos", r"Multi-pos.\ SFT"), ("dpo", "DPO, pooled")]
    mark = {(s, w): pts.ranks({a: T[s][a][w][0] for a in ("rankgrpo", "perresp", "remark") if a in T[s]})
            for s in sets for w in ws}
    tint = {"rankgrpo": r"\baserow ", "remark": r"\oursrow "}
    for a, lab in arms:
        cells = []
        for w in ws:
            for s in sets:
                if a not in T[s]:
                    cells.append("--" if a in extra else r"\tbd"); continue
                cells.append(pts.marked(f4(T[s][a][w][0]), mark[(s, w)].get(a) if a not in ("sft",) and a not in extra else None))
        lines.append(tint.get(a, "") + f"{lab} & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace[1pt]")
    lines.append(r"$\Delta$ (95\% CI) & " + " & ".join(dci2(D["primary"].get(s, {}).get(w)) for w in ws for s in sets) + r" \\")
    # stronger baselines (/E11; 2B and 4B only), a separate group: not part of the Delta row above
    lines += [r"\midrule", rf"\multicolumn{{{2 * k + 1}}}{{@{{}}l}}{{\grp{{Stronger baselines}}}} \\"]
    for a, lab in extra_arms:
        cells = [f4(T[s][a][w][0]) if a in T[s] else "--" for w in ws for s in sets]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    # reference systems: no Qwen3.5 size, so their own header with one value per dataset
    lines += [r"\midrule[\heavyrulewidth]",
              rf"\grp{{Reference}} & \multicolumn{{3}}{{c}}{{Reddit-v2}} & {bh} & \multicolumn{{3}}{{c}}{{Reddit-v2}} & {bh} \\", r"\midrule"]
    if "popularity" in ref:
        lines.append(ref_row("Popularity", [(f4(ref["popularity"]["row"][0]), f4(BOOKS_POP[0])),
                                            (f4(ref["popularity"]["req"][0]), f4(BOOKS_POP[1]))]))
    lines.append(ref_row("kNN prior", [(get(bl["reddit"], "knn", w), get(bl["books"], "knn", w)) for w in ws]))
    if "released" in ref:
        lines.append(ref_row(r"Rank-GRPO$^\ddagger$", [(f4(ref["released"]["row"][0]), "--"), (f4(ref["released"]["req"][0]), "--")]))
    lines.append(rf"\multicolumn{{{2 * k + 1}}}{{@{{}}l}}{{\textit{{zero-shot LLMs}}}} \\")
    lines.append(ref_row(r"\quad GPT-4o$^\dagger$", [("21.47", "--"), ("--", "--")]))
    lines.append(ref_row(r"\quad GPT-5.6", [(get(bl["reddit"], "gpt56", w), get(bl["books"], "gpt56", w)) for w in ws]))
    if "qwen27b" in bl["reddit"]:
        lines.append(ref_row(r"\quad Qwen3.5-27B", [(get(bl["reddit"], "qwen27b", w), get(bl["books"], "qwen27b", w)) for w in ws]))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_main.tex").write_text("% generated by remark/scripts/make_paper_figures.py from "
                                      "outputs/crs/analysis/paper_fig_stats.json; do not edit by hand\n"
                                      + "\n".join(lines) + "\n")


def tab_robust() -> None:
    R, P, SU, RG = D["robust"], D["primary"], D["scale_union"], D["registered"]
    sets = SIZES + ["books"]

    def cell(s, key):
        v = R.get(s, {}).get(key)
        return dci2(v) if v else "--"

    def seedcell(s):
        v = R.get(s, {}).get("seeds")
        return dci2(v["welch"]) if v else "--"

    body = [
        (r"\grp{Resampling unit}", None),
        ("requests (primary)", [dci2(P[s]["row"]) for s in sets]),
        ("Reddit submissions", [cell(s, "submission_per_row") for s in SIZES] + ["--"]),
        ("without the 42 train-overlap requests", [cell(s, "overlap_excluded") for s in SIZES] + ["--"]),
        (r"\grp{Metric}", None),
        ("R@20 per request", [dci2(P[s]["req"]) for s in sets]),
        ("N@20 per row", [dci2(P[s]["ndcg"]) for s in sets]),
        (r"\grp{Seeds}", None),
        ("seed 42 pair", [cell(s, "seed_s42") if R.get(s, {}).get("seed_s42") else r"\tbd" for s in sets]),
        ("seed 1234 pair", [cell(s, "seed_s1234") if s in ("2B", "4B", "9B", "books") else "--" for s in sets]),
        ("three seeds each (Welch)", [seedcell(s) if s in ("2B", "4B", "9B", "books") else "--" for s in sets]),
        # 10-09 (author, appendix review): the "Rerun, same configuration" group is run history and left out
        (r"\grp{Advantage scale}", None),
        ("group $-$ batch, bounded sum", [dci2(SU[s]["row"]) for s in sets]),
    ]
    lines = [r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular}{@{}l cccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{\textbf{Reddit-v2}} & \textbf{Reddit-Books} \\", r"\cmidrule(lr){2-4}\cmidrule(l){5-5}",
             r" & 2B & 4B & 9B & 2B \\", r"\midrule"]
    for i, (lab, cs) in enumerate(body):
        if cs is None:
            if i:
                lines.append(r"\addlinespace[3pt]")
            lines.append(rf"\multicolumn{{5}}{{@{{}}l}}{{{lab}}} \\")
            continue
        lines.append(r"\quad " + lab + " & " + " & ".join(cs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_robust.tex").write_text("% generated by remark/scripts/make_paper_figures.py from "
                                        "outputs/crs/analysis/paper_fig_stats.json; do not edit by hand\n"
                                        + "\n".join(lines) + "\n")


def tab_topk() -> None:
    """Appendix: ReMaRk minus the per-responder baseline at other cutoffs (, per row)."""
    tk = S.get("extras", {}).get("topk")
    if not tk:
        print("tab_topk: n917_e6_topk_metrics.json missing, skipped"); return
    want = [("reddit", "2B_union_group", "2B_rankgrpo", "Reddit-v2, 2B"), ("reddit", "4B_union_group", "4B_rankgrpo", "Reddit-v2, 4B"),
            ("reddit", "9B_union_group", "9B_rankgrpo", "Reddit-v2, 9B"),
            ("books", "books_union_group", "books_perresp_batch", "Reddit-Books, 2B")]
    ms = ["R@5", "R@10", "R@20", "N@20", "Hit@20"]
    found = [next((q for q in tk.get(ds, {}).get("pairs", []) if q["a"] == a and q["b"] == b), None) for ds, a, b, _ in want]
    # metrics as rows, settings as columns: fits one column at \small
    lines = [r"\renewcommand{\arraystretch}{1.08}", r"\begin{tabular}{@{}l cccc@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{\textbf{Reddit-v2}} & \textbf{Reddit-Books} \\", r"\cmidrule(lr){2-4}\cmidrule(l){5-5}",
             r" & 2B & 4B & 9B & 2B \\", r"\midrule"]
    for m in ms:
        lines.append(f"{m} & " + " & ".join(dci2(p[m]["row"], 4) if p else r"\tbd" for p in found) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_topk.tex").write_text("% generated by remark/scripts/make_paper_figures.py from "
                                      "outputs/crs/analysis/paper_fig_stats.json (n917_e6_topk_metrics.json); do not edit by hand\n"
                                      + "\n".join(lines) + "\n")


def main() -> None:
    for f in (fig_main, fig_attrib, fig_curves, fig_curves_music, fig_val_arms, fig_dose, tab_main, tab_robust, tab_topk):
        f(); print("wrote", f.__name__)
    import n932_e3_compute  # fig_gpuh.pdf and tab_compute.tex (own inputs: train logs, val, test pairs)
    n932_e3_compute.main(); print("wrote fig_gpuh, tab_compute")
    if S.get("missing"):
        print("pending inputs (skipped):", *S["missing"], sep="\n  ")


if __name__ == "__main__":
    main()
