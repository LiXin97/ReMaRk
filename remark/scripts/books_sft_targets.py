"""GPT-5.6 teacher lists (20 books) for Reddit-Books training threads (the role of Rank-GRPO's teacher lists).

The user message is exactly the dataset prompt (books header + "Here is the conversation: USER: ...\\n"). A system
message asks for " by <author>" after each line's year (used only to ground the line to Goodreads) and gives the
thread's month, asking for books published before it (the catalog is Goodreads as crawled in late 2017); the SFT
completion written by books_build.py is the grounded "Title (Year)" list (no authors, catalog titles, seen books
and duplicates removed, grounded order kept, no filling; kept only if >= 15 items survive).

Usage (API only):
  env/bin/python remark/scripts/books_sft_targets.py --tag full [--limit 200]
Input: remark/outputs/books/ground/grounded_<tag>.jsonl (threads with >= 1 grounded responder), threads/*.jsonl
Output: remark/outputs/books/sft/api_cache.jsonl, teacher_<tag>.jsonl ({sid, lines: [[title, year, author]]})
"""

from __future__ import annotations

import argparse
import calendar
import json
import random
import re
import sys
import time

from books_common import OUTB, REPO, del_markup, log_usage, prompt_text, split_of

sys.path.insert(0, str(REPO / "remark"))
from lib.llm import batch_chat  # noqa: E402

VERSION = "v2"  # v2 adds the thread date to the system message (v1: 28% of pilot lists < 15 grounded items)
SFT = OUTB / "sft"
SYSTEM = ("You are a book recommender. Follow the user's instructions, but end every line with \" by <author>\" "
          "after the year, e.g. \"The Hobbit (1937) by J.R.R. Tolkien\". Give exactly 20 lines, one book per line, "
          "each a real, specific book (not a series), different from the books the user says they have read. "
          "The conversation took place in {date}: recommend only books published before {date}.")
LINE = re.compile(r"^(.+?)\s+\((\d{4})\)\s*(?:[-–—,:]?\s*by\s+(.+?))?\s*$", re.I)


def month_name(ym: str) -> str:
    y, m = ym.split("-")
    return f"{calendar.month_name[int(m)]} {y}"


def parse_teacher(text: str) -> list[list]:
    out = []
    for line in (text or "").splitlines():
        line = del_markup(line)
        m = LINE.match(line)
        if m:
            out.append([m.group(1).strip().strip('"').strip("*").strip(), int(m.group(2)),
                        (m.group(3) or "").strip().strip("*").strip() or None])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    grounded = {}
    for line in (OUTB / "ground" / f"grounded_{a.tag}.jsonl").open():
        r = json.loads(line)
        if split_of(r["month"]) == "train" and any(b["g"]["status"] == "ok" for bs in r["responders"].values() for b in bs):
            grounded[r["sid"]] = r["month"]
    ctx = {}
    for ym in sorted(set(grounded.values())):
        for line in (OUTB / "threads" / f"{ym}.jsonl").open():
            t = json.loads(line)
            if t["sid"] in grounded:
                ctx[t["sid"]] = t["context"]
    sids = sorted(ctx)
    random.Random(42).shuffle(sids)  # calls run in this order, so a partial run covers a random subset of threads
    if a.limit:
        sids = sids[:a.limit]
    reqs = [{"id": f"t{VERSION}:{s}", "max_tokens": 4000, "meta": {"sid": s},
             "messages": [{"role": "system", "content": SYSTEM.replace("{date}", month_name(grounded[s]))},
                          {"role": "user", "content": prompt_text(ctx[s])}]}
            for s in sids]
    print(f"{len(reqs)} teacher calls", flush=True)
    t0 = time.time()
    res = batch_chat(reqs, SFT / "api_cache.jsonl", workers=a.workers)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    n20 = 0
    with (SFT / f"teacher_{a.tag}{'_limit' if a.limit else ''}.jsonl").open("w") as f:
        for s in sids:
            rec = res.get(f"t{VERSION}:{s}")
            lines = parse_teacher(rec["text"]) if rec else []
            n20 += len(lines) >= 20
            f.write(json.dumps({"sid": s, "lines": lines, "ok": rec is not None}, ensure_ascii=False) + "\n")
    usage = log_usage("sft_targets", SFT / "api_cache.jsonl")
    print(json.dumps({"threads": len(sids), "with_20_parsed_lines": n20, "api_usage_cumulative": usage}, indent=1))


if __name__ == "__main__":
    main()
