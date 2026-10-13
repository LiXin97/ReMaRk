"""Assemble r/musicsuggestions threads from the Arctic Shift pulls, reusing books_threads.build_month.

Same responder definition and filters as Reddit-Books (submitter, deleted/removed, moderators, bots dropped; one
responder per distinct author, their top-level comments merged in time order; comments read from months M..M+2).
Only the paths differ: raw = remark/outputs/music/raw/musicsuggestions, out = remark/outputs/music/threads.

Usage: env/bin/python remark/scripts/music_threads.py --months 2021-01:2025-06
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import books_threads as bt
from books_pull import month_range

REPO = Path(__file__).resolve().parents[2]
bt.RAW = REPO / "remark/outputs/music/raw/musicsuggestions"
OUT = REPO / "remark/outputs/music/threads"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for ym in month_range(a.months):
        if not (bt.RAW / "posts" / f"{ym}.done").exists() or not (bt.RAW / "comments" / f"{ym}.done").exists():
            print(f"{ym}: pull not finished, skipped")
            continue
        st: Counter = Counter()
        th = bt.build_month(ym, st)
        with (OUT / f"{ym}.jsonl").open("w") as f:
            for t in th:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        st["comments_months_available"] = sum((bt.RAW / "comments" / f"{m}.done").exists() for m in bt.next_months(ym, 2))
        stats[ym] = dict(st)
        print(ym, st["threads"], st["threads_with_responder"], st["responders"], flush=True)
    (OUT / f"threads_stats_{a.months.replace(':', '_')}.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
