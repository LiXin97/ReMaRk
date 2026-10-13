"""Assemble r/suggestmeabook threads (submission + top-level responders) from the Arctic Shift pulls.

Responder = distinct author of >= 1 kept top-level comment (parent_id t3_<submission>). Dropped comments: by the
submitter, author or body [deleted]/[removed], AutoModerator, moderator-distinguished, bots (author ends in "bot",
known bot names, or body says "I am a bot"). A responder's top-level comments in one thread are merged in time order.
Dropped submissions: AutoModerator / moderator-distinguished posts. Submissions whose body is [removed]/[deleted] are
kept (Reddit-v2 keeps them; the title carries the request) and counted. Threads with 0 responders are kept in the
output with responders = [] (they are counted, never sent to the extractor).

Context rendering follows Reddit-v2 exactly: "USER: {title}. {selftext}\n" (literal ". " join, body newlines kept).
Comments of a submission may arrive in the following months, so comments are read from months M..M+2 when pulled.

Raw month files may be gzipped (<month>.jsonl.gz) once their pull is finished.
Usage: env/bin/python remark/scripts/books_threads.py --months 2014-05:2017-10
Output: remark/outputs/books/threads/YYYY-MM.jsonl + threads_stats_<months>.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

from books_pull import OUT as RAW, month_range

OUT = Path(__file__).resolve().parents[2] / "remark/outputs/books/threads"
DEAD = {"[deleted]", "[removed]", ""}
BOT_NAMES = {"automoderator", "remindmebot", "goodreads-rebot", "goodreadsbot", "bookbot", "wikitextbot",
             "timezone_bot", "converter-bot", "totesmessenger", "sneakpeekbot", "imdb_bot", "hashtagbot"}
BOT_BODY = re.compile(r"\bI(?: a|')m a bot\b|\bbeep,? boop\b|this action was performed automatically", re.I)


def is_bot(author: str, body: str) -> bool:
    a = author.lower()
    return a in BOT_NAMES or a.endswith("bot") or a.endswith("_bot") or a.endswith("-bot") or bool(BOT_BODY.search(body))


def read_jsonl(path: Path) -> list[dict]:
    """Reads <month>.jsonl, or <month>.jsonl.gz (finished raw pulls are gzipped to save quota)."""
    if not path.exists():
        path = path.with_name(path.name + ".gz")
        if not path.exists():
            return []
    out = []
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def next_months(ym: str, k: int) -> list[str]:
    y, m = map(int, ym.split("-"))
    out = []
    for _ in range(k):
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append(f"{y:04d}-{m:02d}")
    return out


def build_month(ym: str, st: Counter) -> list[dict]:
    posts = read_jsonl(RAW / "posts" / f"{ym}.jsonl")
    pids = {p["id"] for p in posts}
    by_post: dict[str, list[dict]] = {}
    for m in [ym, *next_months(ym, 2)]:
        for c in read_jsonl(RAW / "comments" / f"{m}.jsonl"):
            lid = c.get("link_id", "")[3:]
            if lid in pids and c.get("parent_id", "").startswith("t3_"):
                by_post.setdefault(lid, []).append(c)
    threads = []
    for p in sorted(posts, key=lambda x: (int(x["created_utc"]), x["id"])):
        st["posts"] += 1
        author = p.get("author") or "[deleted]"
        if author.lower() == "automoderator" or p.get("distinguished") == "moderator":
            st["posts_dropped_mod_or_automod"] += 1
            continue
        body = p.get("selftext") or ""
        st["posts_body_removed_or_deleted"] += body.strip() in ("[removed]", "[deleted]")
        st["posts_submitter_deleted"] += author == "[deleted]"
        resp: dict[str, list[dict]] = {}
        seen_ids = set()
        for c in sorted(by_post.get(p["id"], []), key=lambda x: (int(x["created_utc"]), x["id"])):
            if c["id"] in seen_ids:
                continue
            seen_ids.add(c["id"])
            st["top_level_comments"] += 1
            ca, cb = c.get("author") or "[deleted]", (c.get("body") or "").strip()
            if ca in DEAD or cb in DEAD:
                st["tlc_dropped_deleted_removed"] += 1
            elif author != "[deleted]" and ca == author:
                st["tlc_dropped_submitter"] += 1
            elif c.get("distinguished") == "moderator":
                st["tlc_dropped_moderator"] += 1
            elif is_bot(ca, cb):
                st["tlc_dropped_bot"] += 1
            else:
                st["tlc_kept"] += 1
                resp.setdefault(ca, []).append({"id": c["id"], "body": cb, "score": c.get("score"),
                                                "created_utc": int(c["created_utc"])})
        title = p.get("title") or ""
        threads.append({
            "sid": p["id"], "month": ym, "created_utc": int(p["created_utc"]), "author": author,
            "title": title, "selftext": body, "score": p.get("score"), "num_comments": p.get("num_comments"),
            "link_flair_text": p.get("link_flair_text"), "over_18": p.get("over_18"),
            "context": f"USER: {title}. {body}\n",
            "responders": [{"author": a, "comments": cs} for a, cs in resp.items()],
        })
        st["threads"] += 1
        st["threads_with_responder"] += bool(resp)
        st["responders"] += len(resp)
    return threads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    for ym in month_range(a.months):
        if not (RAW / "posts" / f"{ym}.done").exists() or not (RAW / "comments" / f"{ym}.done").exists():
            print(f"{ym}: pull not finished, skipped")
            continue
        st: Counter = Counter()
        th = build_month(ym, st)
        with (OUT / f"{ym}.jsonl").open("w") as f:
            for t in th:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        st["comments_months_available"] = sum((RAW / "comments" / f"{m}.done").exists() for m in next_months(ym, 2))
        stats[ym] = dict(st)
        print(ym, json.dumps(dict(st)), flush=True)
    (OUT / f"threads_stats_{a.months.replace(':', '_')}.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
