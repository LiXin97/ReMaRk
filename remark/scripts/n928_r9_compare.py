"""Paired bootstrap between two n928_r9_eval.py runs on the same split (unit = question; 2,000 resamples,
seed 42; draws as crs_compare.py: rng = default_rng(seed), integers(0, n, (n_boot, n))).

Usage: python remark/scripts/n928_r9_compare.py --a <eval dir A> --b <eval dir B> [--out x.json]
Reports mean(A) - mean(B) with the 95% percentile CI for resp_recall@k, resp_recall_anymatch@k and the official
metrics. Needs only numpy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ["resp_recall@10", "resp_recall@5", "resp_recall@3", "resp_recall@1", "resp_recall_anymatch@10",
           "official/Max Answers - 1", "official/Max Answers - 3", "official/Max Answers - 5",
           "official/Max Answers - 10", "official/Max Incorrect - 5"]


def load(d: Path) -> dict[str, dict]:
    return {r["qid"]: r for r in map(json.loads, (d / "per_instance.jsonl").open())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A, B = load(Path(a.a)), load(Path(a.b))
    sa = json.loads((Path(a.a) / "summary.json").read_text())
    sb = json.loads((Path(a.b) / "summary.json").read_text())
    assert sa["split"] == sb["split"] and set(A) == set(B), "runs must cover the same split and questions"
    qids = sorted(A)
    draws = np.random.default_rng(a.seed).integers(0, len(qids), size=(a.n_boot, len(qids)))
    res = {"a": a.a, "b": a.b, "split": sa["split"], "questions": len(qids), "n_boot": a.n_boot, "seed": a.seed,
           "metrics": {}}
    for m in METRICS:
        xa = np.array([A[q][m] for q in qids], dtype=float)
        xb = np.array([B[q][m] for q in qids], dtype=float)
        d = xa - xb
        boot = d[draws].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res["metrics"][m] = {"a": float(xa.mean()), "b": float(xb.mean()), "diff": float(d.mean()),
                             "ci_low": float(lo), "ci_high": float(hi), "p_le_0": float((boot <= 0).mean())}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    for m, v in res["metrics"].items():
        print(f"{m:28s} A {v['a']:.4f}  B {v['b']:.4f}  diff {v['diff']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
