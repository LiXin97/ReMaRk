"""ProtoQA arms over three seeds (3407 = the registered ProtoQA runs, 42, 1234).

Per seed: test_scraped per-respondent recall@10 of each arm's locked checkpoint and the paired differences (d) - (a) and
(b) - (a) (bootstrap over questions, 2,000 resamples, seed 42; n928_r9_compare's draws). Pooled over seeds: difference of
the arms' mean test scores with a Welch 95% CI (t with Welch-Satterthwaite df), as the registered seed-level comparisons;
also the mean of the per-seed paired differences. Optional 500-step runs (tag _ms500) and arm (c) when present.
Output: outputs/protoqa/analysis/n931_g1_seeds.json
Usage: env/bin/python remark/scripts/n931_g1_seeds.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
O = REPO / "remark/outputs/protoqa"
E = O / "eval"
ARMS = {"a": "single", "b": "rb", "c": "mean", "d": "union"}


def test_dir(arm: str, seed: int, tag: str = "") -> Path | None:
    d = E / (f"r9test_2B_{arm}" if seed == 3407 and not tag else f"g1test_2B_{arm}_s{seed}{tag}")
    return d if (d / "per_instance.jsonl").exists() else None


def per_q(d: Path) -> dict[str, float]:
    return {r["qid"]: r["resp_recall@10"] for r in map(json.loads, open(d / "per_instance.jsonl"))}


def paired(da: Path, db: Path) -> list[float]:
    a, b = per_q(da), per_q(db)
    q = sorted(a)
    diff = np.array([a[x] - b[x] for x in q])
    idx = np.random.default_rng(42).integers(0, len(q), size=(2000, len(q)))
    return [float(diff.mean()), *map(float, np.percentile(diff[idx].mean(1), [2.5, 97.5]))]


def welch(x: list[float], y: list[float]) -> list[float]:
    x, y = np.array(x), np.array(y)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1)) if se > 0 else np.inf
    t = stats.t.ppf(0.975, df) if np.isfinite(df) else 1.96
    d = x.mean() - y.mean()
    return [float(d), float(d - t * se), float(d + t * se), float(df)]


def main() -> None:
    out = {"note": __doc__.split("\n")[0], "seeds": {}, "pooled": {}, "ms500": {}}
    scores = {k: {} for k in ARMS}
    for seed in (3407, 42, 1234):
        s = out["seeds"][seed] = {}
        dirs = {k: test_dir(arm, seed) for k, arm in ARMS.items()}
        for k, d in dirs.items():
            if d:
                scores[k][seed] = float(json.load(open(d / "summary.json"))["resp_recall@10"])
                s[k] = scores[k][seed]
        for k in ("b", "c", "d"):
            if dirs[k] and dirs["a"]:
                s[f"{k}-a"] = paired(dirs[k], dirs["a"])
    for k in ("b", "c", "d"):
        common = sorted(set(scores[k]) & set(scores["a"]))
        if len(common) >= 2:
            out["pooled"][f"{k}-a"] = {"seeds": common, "welch": welch([scores[k][s] for s in common], [scores["a"][s] for s in common]),
                                       "mean_paired": float(np.mean([out["seeds"][s][f"{k}-a"][0] for s in common]))}
    for k in ("b", "d"):
        dk, da = test_dir(ARMS[k], 3407, "_ms500"), test_dir("single", 3407, "_ms500")
        if dk and da:
            out["ms500"][f"{k}-a"] = paired(dk, da)
            out["ms500"]["scores"] = {kk: float(json.load(open(test_dir(ARMS[kk], 3407, "_ms500") / "summary.json"))["resp_recall@10"])
                                      for kk in ("a", "b", "d") if test_dir(ARMS[kk], 3407, "_ms500")}
    (O / "analysis").mkdir(exist_ok=True)
    (O / "analysis/n931_g1_seeds.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
