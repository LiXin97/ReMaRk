"""Figure fig:noise (main text) = (a) test gain of ReMaRk over Rank-GRPO by the request's number of responders
and (b, c) the dose-response, side by side at text width (forest plot moved to the appendix, bucket panel and
dose-response merged). Reuses the data, colours and helpers of make_paper_figures.py (panel code adapted from its
fig_main (b) and fig_dose); reads outputs/crs/analysis/paper_fig_stats.json only.
Usage: python3 remark/scripts/make_fig_noise.py   (system python3: matplotlib)
Output: remark/paper/tex/figures/fig_noise.pdf
(b, c) are drawn in the ReMaRk colour (the earlier green clashed with the significance green of the tables).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

import make_paper_figures as mpf


def main() -> None:
    D, d = mpf.D, mpf.S["extras"]["dose"]
    fig, axs = plt.subplots(1, 3, figsize=(mpf.TEXTW, 1.55), gridspec_kw={"width_ratios": [1.25, 1, 1], "wspace": 0.42})
    # (a) gain by responder count
    ax = axs[0]
    buckets = ["n=1", "n=2-4", "n=5-19", "n>=20"]
    shades = {"2B": "#FDAE6B", "4B": "#E6550D", "9B": "#8C2D04"}
    marks = {"2B": "o", "4B": "s", "9B": "D"}
    for j, s in enumerate(mpf.SIZES):
        b = D["buckets"].get(s)
        if not b:
            continue
        for i, k in enumerate(buckets):
            v = [b[k]["diff"], *b[k]["ci"]]
            ax.errorbar(i + (j - 1) * 0.24, v[0], yerr=mpf.ci_err(v), fmt=marks[s], color=shades[s], ms=3.0, lw=0.9,
                        capsize=1.3, label=s if i == 0 else None)
    mpf.zero(ax)
    ax.set_xticks(range(4), ["1", "2–4", "5–19", "≥20"])
    ax.set_xlabel("responders per request", labelpad=1.5)
    ax.set_ylabel(r"$\Delta$ test R@20 vs. Rank-GRPO"); mpf.pct(ax)
    ax.set_title("(a) gain by number of responders", loc="left", pad=2)
    ax.legend(loc="upper right", ncol=3, borderaxespad=0.2, handlelength=0.9, columnspacing=0.7, handletextpad=0.3,
              fontsize=6.6)
    # (b, c) dose-response
    rho = d["rho"]; ms = ["1", "2", "4", "all"]
    for ax, key, title in ((axs[1], "best/all", "(b) noise removed: all requests"),
                           (axs[2], "best/n=1", "(c) single-responder requests")):
        r = d["readouts"][key]
        xs = np.array([1 - rho[m] for m in ms]); mean = np.array([r["seed_means"][m] for m in ms])
        for m in ms:
            pts = [v for k, v in r["per_seed"].items() if k.split("/")[0] == m]
            ax.plot([1 - rho[m]] * len(pts), pts, ls="none", marker="o", ms=2.4, mfc="white", mec=mpf.COL["remark"],
                    mew=0.7, zorder=3)
        ax.plot(xs, mean, ls="none", marker="D", ms=3.6, color=mpf.COL["remark"], zorder=4)
        b = r["slope"]; a0 = mean.mean() - b * xs.mean(); gx = np.array([0, 1])
        ax.plot(gx, a0 + b * gx, color=mpf.COL["remark"], lw=0.9, zorder=2)
        lo, hi = r["slope_ci"]
        ax.text(0.03, 0.97, f"slope {100 * b:+.2f}\n[{100 * lo:+.2f}, {100 * hi:+.2f}]".replace("-", "−"),
                transform=ax.transAxes, va="top", fontsize=7)
        ax.set_title(title, loc="left", pad=2); mpf.pct(ax)
        ax.set_xticks([1 - rho[m] for m in ms], [f"{m}" for m in ms])
        for lab, dx in zip(ax.get_xticklabels()[2:], (-1.5, 2.5)):  # m=4 and m=all sit close: nudge labels apart
            lab.set_transform(lab.get_transform() + ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans))
        ax.set_xlim(-0.08, 1.08); ax.margins(y=0.18)
        ax.set_xlabel(r"responders in reward, $m$", labelpad=1.5)
    axs[1].set_ylabel("validation R@20, % (2B)")
    fig.savefig(mpf.FIG / "fig_noise.pdf"); plt.close(fig)
    print(mpf.FIG / "fig_noise.pdf")


if __name__ == "__main__":
    main()
