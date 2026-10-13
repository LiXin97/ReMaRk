"""Pull r/suggestmeabook submissions and comments from the Arctic Shift API (Reddit-Books).

Writes one JSONL file per (kind, month) under remark/outputs/books/raw/{posts,comments}/YYYY-MM.jsonl and marks a
finished month with YYYY-MM.done (finished files were later gzipped to YYYY-MM.jsonl.gz; books_threads.py reads both). Resumable: an unfinished month continues from the last created_utc in its file.
Paging: sort=asc, `after` is exclusive, so the cursor is last_ts - 1 and records are de-duplicated by id.
All comments of the month are kept (top-level filtering happens in books_threads.py).

Network-bound, a CPU host is fine:
  env/bin/python remark/scripts/books_pull.py --months 2014-05:2017-09 [--workers 3]
"""

from __future__ import annotations

import argparse
import calendar
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "remark/outputs/books/raw"
API = "https://arctic-shift.photon-reddit.com/api/{kind}/search"
SUB = "suggestmeabook"
UA = "research-books/0.1 (academic dataset build)"
FIELDS = {
    "posts": "id,created_utc,author,title,selftext,num_comments,score,link_flair_text,over_18,distinguished",
    "comments": "id,link_id,parent_id,author,body,created_utc,score,distinguished",
}


def month_range(spec: str) -> list[str]:
    a, b = spec.split(":") if ":" in spec else (spec, spec)
    y, m = map(int, a.split("-"))
    y2, m2 = map(int, b.split("-"))
    out = []
    while (y, m) <= (y2, m2):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_bounds(ym: str) -> tuple[int, int]:
    y, m = map(int, ym.split("-"))
    start = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp())
    end = start + calendar.monthrange(y, m)[1] * 86400
    return start, end


def get(kind: str, after: int, before: int) -> list[dict]:
    q = urllib.parse.urlencode({"subreddit": SUB, "after": after, "before": before, "limit": 100, "sort": "asc",
                                "fields": FIELDS[kind]})
    delay = 2.0
    req = urllib.request.Request(API.format(kind=kind) + "?" + q, headers={"User-Agent": UA})  # default UA gets 403
    for attempt in range(30):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                j = json.loads(r.read())
            if j.get("data") is not None:
                return j["data"]
            err = j.get("error")
        except Exception as exc:  # noqa: BLE001 - network errors: back off and retry
            err = repr(exc)[:200]
        time.sleep(delay)
        delay = min(delay * 1.7, 60)
    raise RuntimeError(f"{kind} after={after}: {err}")


def pull_month(kind: str, ym: str) -> dict:
    d = OUT / kind
    d.mkdir(parents=True, exist_ok=True)
    path, done = d / f"{ym}.jsonl", d / f"{ym}.done"
    if done.exists():
        return json.loads(done.read_text())
    start, end = month_bounds(ym)
    seen: set[str] = set()
    last = start - 1
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seen.add(rec["id"])
                last = max(last, int(rec["created_utc"]))
    cursor = max(start - 1, last - 1)
    t0, pages = time.time(), 0
    with path.open("a") as f:
        while True:
            data = get(kind, cursor, end)
            pages += 1
            new = [x for x in data if x["id"] not in seen]
            for x in new:
                seen.add(x["id"])
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
            f.flush()
            if len(data) < 100:
                break
            top = max(int(x["created_utc"]) for x in data)
            cursor = top - 1 if new else top
    info = {"kind": kind, "month": ym, "records": len(seen), "pages": pages, "seconds": round(time.time() - t0, 1),
            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    done.write_text(json.dumps(info))
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", required=True, help="YYYY-MM or YYYY-MM:YYYY-MM (inclusive)")
    ap.add_argument("--kinds", nargs="+", default=["posts", "comments"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    jobs = [(k, m) for m in month_range(a.months) for k in a.kinds]
    with ThreadPoolExecutor(a.workers) as pool:
        for info in pool.map(lambda km: pull_month(*km), jobs):
            print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
