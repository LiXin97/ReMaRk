"""Compute-matched comparison of ReMaRk and Rank-GRPO (Reddit-v2 test), the GPU-hour figure and table.

GPU-hours of a run up to step S = 4 GPUs x (sum over logged intervals ending at a step <= S of 10 x step_time, plus the
steps not covered by any log at the run's mean logged step_time) / 3600. Sources per run: trainer_state.json log_history of
every checkpoint under grpo_qwen_selected/<run>/ (resumed segments offset by checkpoint step - global_step), then the node
logs outputs/crs/train_logs/*/grpo_<run>[_seg<k>].log (a _seg<k> file starts at step k; plain logs start at OFFSET[run] or 0).
The share of steps filled with the mean is reported per cell.
Cells (difference = ReMaRk - Rank-GRPO, test R@20 per row, paired bootstrap over requests, 2,000, seed 42; crs_compare):
  1k vs 2k   ReMaRk step 1,000 (*_ms1500 runs) vs Rank-GRPO step 2,000 (Table 2)          test_<S>_g5_match_pair_row.json
  2k vs 4k   ReMaRk step 2,000 (Table 2) vs Rank-GRPO step 4,000 (reruns *_ms4000_lr1e-6)    test_<S>_e3_2k4k_pair_row.json
  best/best  plateau pairs, validation-selected steps (2B, 4B, 9B)                    test_<S>_r1*/g4 files
Figure fig_gpuh.pdf (main text): validation-selected score (best checkpoint so far) against training GPU-hours, each arm up
to its selected step; 2B = mean of seeds 3407  and 42  with a min-max band (see best_so_far for the rule); 4B, 9B
single seed. Figure fig_gpuh_raw.pdf (appendix): raw validation R@20 against cumulative training GPU-hours (validation and test excluded) of the seed-3407
plateau pairs: 2B, 4B  and, once both arms are locked, 9B as a third panel. Also called from
make_paper_figures.py main().
Usage: cd remark/scripts && python3 n932_e3_compute.py   (system python3: matplotlib; crs_compare runs in env/bin/python; computes the 2k-vs-4k comparison once both tests exist)
Outputs: outputs/crs/analysis/n932_e3_compute.json, paper/tex/figures/fig_gpuh.pdf, paper/tex/tables/tab_compute.tex
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n928_step_cost import entries_from_log  # noqa: E402
import make_paper_figures as mpf  # noqa: E402  (style: rcParams, COL, MARK, COLW)

plt = mpf.plt
REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
L, SEL, V, T, A = C / "train_logs", C / "grpo_qwen_selected", C / "val", C / "test", C / "analysis"
OFFSET = {("2B_single_group_s3407_p0c_ms8000", "run_logs/grpo_2B_single_group_s3407_p0c_ms8000.log"): 6250}  #

# setting -> cell -> (ReMaRk run, step, Rank-GRPO run, step, comparison file stem or None)
CELLS = {
    "2B": {"1k_2k": ("2B_union_group_s3407_p0c_ms1500", 1000, "2B_single_group_s3407_p0c", 2000, "test_2B_g5_match_pair"),
           "2k_4k": ("2B_union_group_s3407_p0c", 2000, "2B_single_group_s3407_p0c_ms4000_lr1e-6", 4000, "test_2B_e3_2k4k_pair"),
           "best": ("2B_union_group_s3407_p0c_ms8000", 4000, "2B_single_group_s3407_p0c_ms8000", 5250, "test_2B_r1_plateau_pair")},
    "4B": {"1k_2k": ("4B_union_group_s3407_p0c_ms1500", 1000, "4B_single_group_s3407_p0c", 2000, "test_4B_g5_match_pair"),
           "2k_4k": ("4B_union_group_s3407_p0c", 2000, "4B_single_group_s3407_p0c_ms4000_lr1e-6", 4000, "test_4B_e3_2k4k_pair"),
           "best": ("4B_union_group_s3407_p0c_ms8000", 3000, "4B_single_group_s3407_p0c_ms8000", 5250, "test_4B_r1c_plateau_pair")},
    "9B": {"1k_2k": ("9B_union_group_s3407_p0c_ms1500", 1000, "9B_single_group_s3407_p0c", 2000, "test_9B_g5_match_pair"),
           "best": ("9B_union_group_s3407_p0c_ms8000", 2750, "9B_single_group_s3407_p0c_ms8000", None, "test_9B_g4_plateau_pair")},
}
# 2k-vs-4k test dirs (the ReMaRk side is Table 2's 2,000-step test)
E3_TESTS = {"2B": ("test_2B_union_group_step2000", "test_2B_single_group_s3407_p0c_ms4000_lr1e-6_step4000"),
            "4B": ("test_4B_union_group_s3407_step2000", "test_4B_single_group_s3407_p0c_ms4000_lr1e-6_step4000")}
LLAMA = {"remark": 0.2356, "remark_test": "test_llama3b_union_group_s3407_p0c_step1750", "remark_run": "llama3b_union_group_s3407_p0c",
         "remark_step": 1750, "official": 0.2365}
# panels: 2B, 4B, 9B (drawn once both arms are locked)
FIG_PAIRS = {"2B": ("2B_union_group_s3407_p0c_ms8000", "2B_single_group_s3407_p0c_ms8000"),
             "4B": ("4B_union_group_s3407_p0c_ms8000", "4B_single_group_s3407_p0c_ms8000"),
             "9B": ("9B_union_group_s3407_p0c_ms8000", "9B_single_group_s3407_p0c_ms8000")}


def step_times(run: str) -> dict[int, float]:
    logged: dict[int, float] = {}
    for ck in sorted(SEL.glob(f"{run}/checkpoint-*")):
        ts = ck / "trainer_state.json"
        if ts.exists():
            t = json.load(open(ts))
            off = int(ck.name.split("-")[1]) - t["global_step"]
            for x in t["log_history"]:
                if "step_time" in x:
                    logged[int(x["step"]) + off] = float(x["step_time"])
    for p in sorted(L.glob(f"*/grpo_{run}.log")) + sorted(L.glob(f"*/grpo_{run}_seg*.log")):
        if "discarded" in str(p) or "aborted" in p.name:
            continue
        m = re.search(r"_seg(\d+)\.log$", p.name)
        off = int(m.group(1)) if m else OFFSET.get((run, f"{p.parent.name}/{p.name}"), 0)
        for i, x in enumerate(entries_from_log(p)):
            logged[off + 10 * (i + 1)] = x["step_time"]
    return logged


def gpuh(run: str, S: int, cache: dict = {}) -> dict | None:  # noqa: B006
    if run not in cache:
        cache[run] = step_times(run)
    lg = cache[run]
    if not lg:
        return None
    mean = float(np.mean(list(lg.values())))
    have = [k for k in lg if k <= S]
    est = max(0, S - 10 * len(have))
    return {"steps": S, "gpu_hours": 4 * (10 * sum(lg[k] for k in have) + est * mean) / 3600,
            "share_estimated": est / S, "mean_s_per_step": mean}


def val_curve(run: str) -> dict[int, float]:
    out = {}
    for p in V.glob(f"grpo_qwen_{run}_step*/summary.json"):
        out[int(p.parent.name.rsplit("_step", 1)[1])] = json.load(open(p))["truncate_R@20"]
    return dict(sorted(out.items()))


def compare(stem: str, a: str, b: str) -> None:
    for w in ("row", "unit"):
        out = A / f"{stem}_{w}.json"
        if not out.exists():
            subprocess.run([str(REPO / "env/bin/python"), str(REPO / "remark/scripts/crs_compare.py"),
                            "--a", f"remark/outputs/crs/test/{a}", "--b", f"remark/outputs/crs/test/{b}",
                            "--split", "test", "--weighting", w, "--out", str(out)], check=True, cwd=REPO)


def read_diff(stem: str) -> dict | None:
    out = {}
    for w in ("row", "unit"):
        f = A / f"{stem}_{w}.json"
        if not f.exists():
            return None
        r = json.load(open(f))["pairs"][0]["R@20"]
        out[w] = {"a": r["a"], "b": r["b"], "diff": r["diff"], "ci": [r["ci_low"], r["ci_high"]],
                  "a_dir": json.load(open(f))["pairs"][0]["a"], "b_dir": json.load(open(f))["pairs"][0]["b"]}
    return out


def lock_step(run: str) -> int | None:
    f = C / "locks" / f"{run}.json"
    return json.load(open(f))["best_step"] if f.exists() else None


def fig_gpuh_raw(res: dict) -> None:
    """Appendix: raw validation curves (solid) + best so far (dashed), seed 3407, 4B drift annotated."""
    pairs = {k: v for k, v in FIG_PAIRS.items() if all(lock_step(r) for r in v)}
    fig, axs = plt.subplots(1, len(pairs), figsize=(mpf.COLW, 1.55 if len(pairs) == 2 else 1.45), sharey=False)
    for ax, (size, (rm, rg)) in zip(axs, pairs.items()):
        for arm, run in (("rankgrpo", rg), ("remark", rm)):
            cv = val_curve(run)
            xs = [0.0] + [gpuh(run, s)["gpu_hours"] for s in cv]
            ys = [sft_point(size)] + [100 * v for v in cv.values()]  # step 0 = the common SFT checkpoint
            ax.plot(xs, ys, color=mpf.COL[arm], lw=1.1)
            # best so far: the score that selecting the checkpoint by validation would give at that compute
            ax.plot(xs, np.maximum.accumulate(ys), color=mpf.COL[arm], lw=0.7, ls=(0, (2.2, 1.4)), alpha=0.9)
            if (size, arm) == ("4B", "remark") and 2500 in cv:  # 10-03 14:55: transient year drift after a KL spike
                ax.annotate("year drift\nafter KL spike", xy=(gpuh(run, 2500)["gpu_hours"], 100 * cv[2500]),
                            xytext=(0.97, 0.06), textcoords="axes fraction", ha="right", va="bottom", fontsize=6, linespacing=1.0,
                            color="#444444", arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#777777",
                                                         "shrinkA": 1, "shrinkB": 2})
            b = lock_step(run)
            if b in cv:
                ax.plot([gpuh(run, b)["gpu_hours"]], [100 * cv[b]], marker=mpf.MARK[arm], color=mpf.COL[arm], ms=3.6,
                        mec="white", mew=0.5, zorder=5)
            res["figure_raw"][f"{size}_{arm}"] = {"run": run, "best_step": b, "points": [[x, y] for x, y in zip(xs, ys)]}
        ax.set_title(f"Qwen3.5-{size}")
        ax.set_xlabel("RL GPU-hours" if len(pairs) == 2 else "GPU-h")
        ax.set_xlim(left=0)
    axs[0].set_ylabel("Val. R@20")
    fig.tight_layout(w_pad=0.6)
    Line2D = mpf.Line2D
    h = [Line2D([], [], color=mpf.COL["remark"], lw=1.3, label="ReMaRk"),
         Line2D([], [], color=mpf.COL["rankgrpo"], lw=1.3, label="Rank-GRPO"),
         Line2D([], [], color="#555555", lw=1.1, label="raw"),
         Line2D([], [], color="#555555", lw=0.8, ls=(0, (2.2, 1.4)), label="best so far")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, handlelength=1.6,
               columnspacing=0.8)
    fig.savefig(mpf.FIG / "fig_gpuh_raw.pdf")
    plt.close(fig)


# common SFT start of both arms (runner7h reads outputs/crs/sft/qwen35_<size>/selected.json at launch; unchanged since
# 09-28, before every run here; the live trainer shows --sft_path .../qwen35_9B/checkpoint-1000)
SFT_VAL = {"2B": "sft_qwen35_2B_step1500", "4B": "sft_qwen35_4B_step1000", "9B": "sft_qwen35_9B_step1000"}


def sft_point(size: str) -> float:
    return 100 * json.load(open(V / SFT_VAL[size] / "summary.json"))["truncate_R@20"]


# main-text figure: 2B averages two seeds (s3407, s42); 4B and 9B are single-seed (s3407)
MAIN_PANELS = {"2B": {"remark": ["2B_union_group_s3407_p0c_ms8000", "2B_union_group_s42_p0c_ms8000"],
                      "rankgrpo": ["2B_single_group_s3407_p0c_ms8000", "2B_single_group_s42_p0c_ms8000"]},
               "4B": {"remark": ["4B_union_group_s3407_p0c_ms8000"], "rankgrpo": ["4B_single_group_s3407_p0c_ms8000"]},
               "9B": {"remark": ["9B_union_group_s3407_p0c_ms8000"], "rankgrpo": ["9B_single_group_s3407_p0c_ms8000"]}}


def best_so_far(runs: list[str]) -> dict:
    """Per step t on the 250 grid up to the latest selected step T among the seeds: each seed's best validation R@20 over
    its validated steps <= t (its validation-selected score at t) and its training GPU-h to t. A seed whose run already
    stopped (plateau rule) before t keeps its final value (= its selected checkpoint's score) and its GPU-h at the stop.
    Returns x = mean GPU-h, y = mean best so far, band = min/max over seeds."""
    cvs = {r: val_curve(r) for r in runs}
    sel = {r: lock_step(r) for r in runs}
    stop = {r: max(cvs[r]) for r in runs}
    T = max(sel.values())
    s0 = sft_point(runs[0].split("_")[0])
    out = {"x": [0.0], "y": [s0], "lo": [s0], "hi": [s0], "steps": [0], "T": T, "selected": sel, "stop": stop}
    for t in range(250, T + 1, 250):
        vals, gh = [], []
        for r in runs:
            have = [v for s, v in cvs[r].items() if s <= min(t, stop[r])]
            if not have:
                break
            vals.append(max(s0, 100 * max(have))); gh.append(gpuh(r, min(t, stop[r]))["gpu_hours"])
        else:
            out["steps"].append(t); out["x"].append(float(np.mean(gh))); out["y"].append(float(np.mean(vals)))
            out["lo"].append(min(vals)); out["hi"].append(max(vals))
    return out


def fig_gpuh(res: dict) -> None:
    """Main text: validation-selected score (best checkpoint so far) against training GPU-h, to each arm's selected step."""
    panels = {k: v for k, v in MAIN_PANELS.items() if all(lock_step(r) for runs in v.values() for r in runs)}
    fig, axs = plt.subplots(1, len(panels), figsize=(mpf.COLW, 1.55 if len(panels) == 2 else 1.45), sharey=False)
    for ax, (size, arms) in zip(axs, panels.items()):
        for arm in ("rankgrpo", "remark"):
            c = best_so_far(arms[arm])
            ax.plot(c["x"], c["y"], color=mpf.COL[arm], lw=1.2)
            if len(arms[arm]) > 1:
                ax.fill_between(c["x"], c["lo"], c["hi"], color=mpf.COL[arm], alpha=0.15, lw=0)
            ax.plot([c["x"][-1]], [c["y"][-1]], marker=mpf.MARK[arm], color=mpf.COL[arm], ms=3.6, mec="white", mew=0.5,
                    zorder=5)
            res["figure"][f"{size}_{arm}"] = {"runs": arms[arm], **c}
        ax.set_title(f"Qwen3.5-{size}" + (" (2 seeds)" if len(arms["remark"]) > 1 else ""))
        ax.set_xlabel("RL GPU-hours" if len(panels) == 2 else "GPU-h")
        ax.set_xlim(left=0)
    axs[0].set_ylabel("Val. R@20 (best ckpt so far)")
    fig.tight_layout(w_pad=0.6)
    Line2D = mpf.Line2D
    h = [Line2D([], [], color=mpf.COL["remark"], marker=mpf.MARK["remark"], ms=3.6, lw=1.3, label="ReMaRk"),
         Line2D([], [], color=mpf.COL["rankgrpo"], marker=mpf.MARK["rankgrpo"], ms=3.6, lw=1.3, label="Rank-GRPO")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2, handlelength=1.8)
    fig.savefig(mpf.FIG / "fig_gpuh.pdf")
    plt.close(fig)
    src = {}
    for runs in MAIN_PANELS["2B"].values():
        for r in runs:
            st = step_times(r)
            src[r] = {"selected_step": lock_step(r), "stop_step": max(val_curve(r)),
                      "logged_steps": 10 * len(st), "logged_range": [min(st), max(st)] if st else None,
                      "to_selected": gpuh(r, lock_step(r)), "to_stop": gpuh(r, max(val_curve(r))),
                      "log_files": sorted(str(p.relative_to(L)) for p in L.glob(f"*/grpo_{r}*.log")),
                      "trainer_state": sorted(str(p.relative_to(SEL)) for p in SEL.glob(f"{r}/checkpoint-*/trainer_state.json"))}
    res["gpuh_sources_2B"] = src


def fmt_cell(d: dict | None, ga: dict | None, gb: dict | None) -> tuple[str, str]:
    if d is None:
        return "pending", "--"
    r = d["row"]
    lo, hi = (100 * x for x in r["ci"])
    v = 100 * r["diff"]
    sgn = lambda x: f"{x:+.2f}".replace("-", "$-$")  # noqa: E731
    num = lambda x: f"{x:.2f}".replace("-", "$-$")  # noqa: E731
    mac = "\\up" if lo > 0 else ("\\dn" if hi < 0 else "\\ns")
    g = f"{ga['gpu_hours']:.0f} / {gb['gpu_hours']:.0f}" if ga and gb else "--"
    return f"\\dstack{{{mac}{{{sgn(v)}}}}}{{[{num(lo)},{num(hi)}]}}", g


def table(res: dict) -> None:
    lines = ["% generated by remark/scripts/n932_e3_compute.py; do not edit by hand",
             "\\renewcommand{\\arraystretch}{1.08}", "\\begin{tabular}{@{}l cc cc cc@{}}", "\\toprule",
             " & \\multicolumn{2}{c}{ReMaRk 1k vs Rank-GRPO 2k} & \\multicolumn{2}{c}{ReMaRk 2k vs Rank-GRPO 4k}"
             " & \\multicolumn{2}{c}{best vs best} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}",
             " & $\\Delta$R@20 & GPU-h & $\\Delta$R@20 & GPU-h & $\\Delta$R@20 & GPU-h \\\\", "\\midrule"]
    for size in ("2B", "4B", "9B"):
        row = [f"Qwen3.5-{size}"]
        for cell in ("1k_2k", "2k_4k", "best"):
            c = res["cells"][size].get(cell)
            if c is None:
                row += ["--", "--"]
            else:
                row += list(fmt_cell(c["diff"], c["gpuh_remark"], c["gpuh_rankgrpo"]))
        lines.append(" & ".join(row) + " \\\\")
    lm = res["llama"]
    g = f"{lm['gpuh_remark']['gpu_hours']:.0f} / --" if lm.get("gpuh_remark") else "--"
    lines += ["\\addlinespace[3pt]",
              f"Llama-3.2-3B$^\\dagger$ & \\multicolumn{{6}}{{l}}{{ReMaRk 2k-step run {100 * lm['remark']:.2f} vs released"
              f" Rank-GRPO (15.8k steps) {100 * lm['official']:.2f}; GPU-h {g}}} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (mpf.TAB / "tab_compute.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    res: dict = {"note": __doc__.split("\n")[0], "cells": {}, "figure": {}, "figure_raw": {}, "missing": []}
    for size, (a, b) in E3_TESTS.items():
        if (T / a / "summary.json").exists() and (T / b / "summary.json").exists():
            compare(f"test_{size}_e3_2k4k_pair", a, b)
        else:
            res["missing"].append(f"{size} 2k-vs-4k test: {b}")
    for size, cells in CELLS.items():
        res["cells"][size] = {}
        for cell, (rm, sa, rg, sb, stem) in cells.items():
            if sb is None:
                sb = lock_step(rg)
            d = read_diff(stem)
            if d is None:
                res["missing"].append(f"{size} {cell}: {stem}")
            res["cells"][size][cell] = {"remark_run": rm, "remark_step": sa, "rankgrpo_run": rg, "rankgrpo_step": sb,
                                        "diff": d, "gpuh_remark": gpuh(rm, sa),
                                        "gpuh_rankgrpo": gpuh(rg, sb) if sb else None}
    res["llama"] = dict(LLAMA, gpuh_remark=gpuh(LLAMA["remark_run"], LLAMA["remark_step"]))
    fig_gpuh(res)
    fig_gpuh_raw(res)
    table(res)
    (A / "n932_e3_compute.json").write_text(json.dumps(res, indent=1))
    for size, cells in res["cells"].items():
        for cell, c in cells.items():
            d = c["diff"]["row"] if c["diff"] else None
            print(size, cell, f"{100 * d['diff']:+.2f} [{100 * d['ci'][0]:.2f},{100 * d['ci'][1]:.2f}]" if d else "pending",
                  "GPU-h", *(f"{g['gpu_hours']:.1f} (est {g['share_estimated']:.0%})" if g else "--"
                             for g in (c["gpuh_remark"], c["gpuh_rankgrpo"])))
    print("missing:", *res["missing"], sep="\n  ")


if __name__ == "__main__":
    main()
