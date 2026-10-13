"""Shared cell formatting for the paper's generated tables.

The LaTeX macros used here are defined once in main.tex (preamble, "table style"), so the look of every table changes in
one place: \\best / \\second (best and second-best trained arm per column), \\up / \\dn / \\ns (a difference whose 95% CI
is above zero / below zero / includes zero), \\dstack{estimate}{CI} (CI on a second line in grey), \\rf (reference
systems, grey), \\grp (group header), \\oursrow / \\baserow (row tints for ReMaRk and Rank-GRPO).
"""
from __future__ import annotations

SCALE = 100  # the paper reports R@20 / N@20 / Hit@20 in % and differences in percentage points


def m(s: str) -> str:
    """Typeset minus signs."""
    return s.replace("-", "$-$")


def num(x: float, d: int = 2) -> str:
    return f"{SCALE * x:.{d}f}"


def nz(x: float, d: int = 2, plus: bool = False) -> str:
    """Scaled value with d decimals; a value that rounds to zero is typeset unsigned ("0.00", never "-0.00"/"+0.00")."""
    s = f"{SCALE * x:+.{d}f}" if plus else f"{SCALE * x:.{d}f}"
    return s.lstrip("+-") if float(s) == 0 else s


def ci_bounds(lo: float, hi: float, d: int = 2) -> str:
    f = lambda x: m(nz(x, d))  # sign only when negative; no space after the comma
    return f"[{f(lo)},{f(hi)}]"


def delta(v: list[float] | None, d: int = 2, stacked: bool = True) -> str:
    """Paired difference [estimate, lo, hi]: coloured by its CI, CI in grey (second line when stacked)."""
    if not v:
        return r"\tbd"
    est, lo, hi = v[0], v[1], v[2]
    mac = r"\up" if lo > 0 else (r"\dn" if hi < 0 else r"\ns")
    top = f"{mac}{{{m(nz(est, d, plus=True))}}}"
    return rf"\dstack{{{top}}}{{{ci_bounds(lo, hi)}}}" if stacked else rf"{top} {{\scriptsize\color{{cigray}}{ci_bounds(lo, hi)}}}"


def ranks(values: dict[str, float]) -> dict[str, str]:
    """'best' / 'second' marks for the highest and second-highest value (ties share the mark)."""
    order = sorted(set(values.values()), reverse=True)
    out = {}
    for k, v in values.items():
        if v == order[0]:
            out[k] = "best"
        elif len(order) > 1 and v == order[1]:
            out[k] = "second"
    return out


def marked(s: str, mark: str | None) -> str:
    return {"best": rf"\best{{{s}}}", "second": rf"\second{{{s}}}"}.get(mark or "", s)
