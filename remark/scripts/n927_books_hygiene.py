"""Length-hygiene check of every locked Reddit-Books P0d arm in the paper (appendix "Check of the Books reruns").

For each arm: the minimum exact-20 share over its training logging windows (rank/frac_exact_len, every log of the run
on every node), the maximum malformed-line share over those windows, the exact-20 share on validation at the locked
step and on test (lists with exactly 20 parsed items in per_instance.jsonl), and the test share of lists with fewer
than 20 parsed items. The one-word-20th-line exploit of the earlier Books runs shows as a 20th line that
does not parse: under P0d's exact-items rule it lowers exact-20 in training and leaves 19 parsed items at test, and it
raises the malformed-line share; the three columns together are the check.

Usage: env/bin/python remark/scripts/n927_books_hygiene.py
Output: outputs/crs/analysis/n927_books_hygiene.json
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
C = REPO / "remark/outputs/crs"
ARMS = [  # (label, training run name, locked step if the arm predates locks/<run>.json, test dir if named differently)
    ("2B union+group s3407", "reddit_books_2B_union_group_s3407_p0c_cap384_ex", 1500, "test_books_union_group_p0d_step1500"),
    ("2B union+group s42", "reddit_books_2B_union_group_s42_p0c_cap384_ex", 2000, "test_books_union_group_s42_p0d_step2000"),
    ("2B per-responder batch s3407", "reddit_books_2B_single_batch_s3407_p0c_cap384_ex", 2000, "test_books_single_batch_p0d_step2000"),
    ("2B per-responder batch s42", "reddit_books_2B_single_batch_s42_p0c_cap384_ex", 2000, "test_books_single_batch_s42_p0d_step2000"),
    ("2B Rank-GRPO (single+group) s3407", "reddit_books_2B_single_group_s3407_p0c_cap384_ex", 2000, "test_books_single_group_p0d_step2000"),
    ("2B union+batch s3407", "reddit_books_2B_union_batch_s3407_p0c_cap384_ex", 2000, "test_books_union_p0d_step2000"),
    ("4B union+group s3407", "reddit_books_4B_union_group_s3407_p0c_cap384_ex", None, None),
    ("4B union+group s42", "reddit_books_4B_union_group_s42_p0c_cap384_ex", None, None),
    ("4B per-responder batch s3407", "reddit_books_4B_single_batch_s3407_p0c_cap384_ex", None, None),
    ("4B per-responder batch s42", "reddit_books_4B_single_batch_s42_p0c_cap384_ex", None, None),
    ("4B Rank-GRPO (single+group) s3407", "reddit_books_4B_single_group_s3407_p0c_cap384_ex", None, None),
]


def windows(run: str) -> list[list[dict]]:
    """Logging windows of every log of the run (one list per log file, in order; a log starts at the run's first
    step or at a resume point)."""
    out = []
    for log in sorted((C / "train_logs").rglob(f"grpo_{run}*.log")):
        txt = log.read_text(errors="ignore")
        ws = []
        for m in re.finditer(r"\{'loss'.*?\}", txt):
            try:
                ws.append(ast.literal_eval(m.group(0)))
            except (ValueError, SyntaxError):
                continue
        if ws:
            out.append(ws)
    return out


def exact_share(d: Path) -> tuple[float | None, float | None, int]:
    p = d / "per_instance.jsonl"
    if not p.exists():
        return None, None, 0
    n = e = short = 0
    for line in open(p):
        k = len(json.loads(line).get("recs", []))
        n += 1; e += k == 20; short += k < 20
    return (e / n, short / n, n) if n else (None, None, 0)


def main() -> None:
    res = []
    for label, run, fixed_step, test_name in ARMS:
        logs = windows(run)
        allw = [x for ws in logs for x in ws]
        ex = [float(x["rank/frac_exact_len"]) for x in allw if "rank/frac_exact_len" in x]
        # start-up: the first 10 logging windows (100 steps) of each log; the SFT policy starts below 95% exact-20
        late = [float(x["rank/frac_exact_len"]) for ws in logs for x in ws[10:] if "rank/frac_exact_len" in x]
        first = [float(ws[0]["rank/frac_exact_len"]) for ws in logs if "rank/frac_exact_len" in ws[0]]
        reach = None
        if logs:
            for i, x in enumerate(logs[0]):
                if float(x.get("rank/frac_exact_len", 0)) >= 0.95:
                    reach = 10 * (i + 1); break
        mal = [float(x["rank/malformed_line_frac"]) for x in allw if "rank/malformed_line_frac" in x]
        lock = C / "locks" / f"{run}.json"
        step = json.loads(lock.read_text())["best_step"] if lock.exists() else fixed_step
        val = exact_share(C / "val" / f"grpo_qwen_{run}_step{step}") if step else (None, None, 0)
        tests = [C / "test" / test_name] if test_name else sorted((C / "test").glob(f"test_{run}_step*"))
        tst = exact_share(tests[0]) if tests and tests[0].exists() else (None, None, 0)
        res.append({"arm": label, "run": run, "locked_step": step, "train_windows": len(ex), "logs": len(logs),
                    "train_exact20_first_window": round(first[0], 4) if first else None,
                    "train_first_step_at_95": reach,
                    "train_exact20_min_after_100_steps": round(min(late), 4) if late else None,
                    "train_exact20_min_all": round(min(ex), 4) if ex else None,
                    "train_malformed_max": round(max(mal), 5) if mal else None,
                    "val_exact20": round(val[0], 4) if val[0] is not None else None,
                    "test_exact20": round(tst[0], 4) if tst[0] is not None else None,
                    "test_short_lists": round(tst[1], 4) if tst[1] is not None else None,
                    "test_dir": tests[0].name if tests else None})
    have = [r for r in res if r["train_exact20_min_after_100_steps"] is not None]
    summary = {
        "arms": len(res), "arms_with_logs": len(have),
        "min_train_exact20_after_100_steps": min(r["train_exact20_min_after_100_steps"] for r in have) if have else None,
        "latest_step_reaching_95": max((r["train_first_step_at_95"] or 0) for r in have) if have else None,
        "min_val_exact20": min((r["val_exact20"] for r in res if r["val_exact20"] is not None), default=None),
        "min_test_exact20": min((r["test_exact20"] for r in res if r["test_exact20"] is not None), default=None),
        "max_train_malformed": max((r["train_malformed_max"] for r in have if r["train_malformed_max"] is not None), default=None),
    }
    out = {"summary": summary, "arms": res,
           "exploit_rule": "a 20th line that does not parse: lowers training exact-20 under P0d's exact-items rule, "
                           "raises the malformed-line share, and leaves 19 parsed items at test"}
    (C / "analysis/n927_books_hygiene.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    for r in res:
        print(f"{r['arm']:34s} @{r['locked_step']}  start {r['train_exact20_first_window']} >=.95@{r['train_first_step_at_95']}"
              f"  min>100 {r['train_exact20_min_after_100_steps']}  malf {r['train_malformed_max']}"
              f"  val {r['val_exact20']}  test {r['test_exact20']}  short {r['test_short_lists']}")


if __name__ == "__main__":
    main()
