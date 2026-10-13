"""GPT-5.6 mention extraction for Reddit-Books threads.

Unit = a thread, or a chunk of MAX_RESP responders of a larger thread (request books come from the first chunk).
The API limit is queries per minute, so consecutive units are packed into one call (<= --pack threads, <= MAX_RESP
responders, <= PACK_CHARS characters); the rules are identical to the single-thread prompt (PROMPT) plus a note to
answer per thread id. Strict JSON output: request_books (with a `read` flag) and, per responder, the books that
responder recommends. Calls with unparsable output are re-asked twice, then their units go one per call in the
single-thread format. Resumable through lib.llm.batch_chat's JSONL cache.

Usage (API only):
  env/bin/python remark/scripts/books_extract.py --months 2017-03 --sample 200 --tag pilot2017
  env/bin/python remark/scripts/books_extract.py --months 2014-05:2017-09 --tag full
Outputs: remark/outputs/books/extract/api_cache.jsonl (all calls), mentions_<tag>.jsonl (one row per thread).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter

from books_common import OUTB, REPO, log_usage, parse_json_text
from books_pull import month_range

sys.path.insert(0, str(REPO / "remark"))
from lib.llm import batch_chat  # noqa: E402

VERSION = "v2p"  # packed calls; single-unit calls keep the v1 id / format
MAX_RESP = 40
PACK_CHARS = 24000
MAX_REQ_CHARS, MAX_RESP_CHARS = 6000, 2500
EXT = OUTB / "extract"

PROMPT = """You extract book mentions from a Reddit r/suggestmeabook thread. The thread has a REQUEST (the original \
post) and RESPONSES (top-level replies by other users; each response id is one user, with that user's replies merged).

Return strict JSON of this shape:
{"request_books": [{"title": str, "author": str or null, "year": int or null, "read": true or false}],
 "responses": {"r1": [{"title": str, "author": str or null, "year": int or null, "series": str or null}], "r2": [], ...}}

Rules:
1. request_books: every specific book the requester names (books they read, liked, disliked, want to avoid, or have \
not read yet). read = true if the requester says or clearly implies they have read it (including disliked or \
unfinished books); otherwise false.
2. responses: for EVERY response id, the books that response recommends to the requester, in order of appearance. \
Include a book when the responder suggests it as something to read, including replies that are only a title, \
"+1 for X", or a list. Exclude books mentioned only for comparison or context (e.g. the requester's own examples, \
"it's like X but darker"), books the responder advises against, and books merely mentioned without being suggested. \
Use [] when a response recommends no specific book.
3. Only books: novels, non-fiction, short-story collections, poetry, plays, graphic novels, comics, manga. Not films, \
TV shows, games, podcasts, websites, magazines, genres, or authors without a book.
4. Series: if a response recommends a series without naming a specific book, give the first book of the series as \
"title" and the series name as "series". If it names a specific book, give that book (series may still be filled).
5. A recommendation of an author in general without naming any book ("anything by Terry Pratchett") yields nothing.
6. title: the book's standard published English title, correctly spelled and capitalized, without series numbering, \
quotes or markdown (fix obvious typos and abbreviations, e.g. "HHGTTG" -> "The Hitchhiker's Guide to the Galaxy"). \
author: the author's full name as usually credited, from the text or from your knowledge if you are confident, else \
null. year: the year of first publication (original language) if you know it, else null.
7. Never add books that the thread does not mention. Output the JSON object only.

"""


PACK_NOTE = """This input contains {n} independent threads (t1, t2, ...). Apply the rules to each thread separately \
and return ONE JSON object keyed by thread id, each value being the object described above for that thread: \
{"t1": {"request_books": [...], "responses": {...}}, "t2": {...}, ...}. Response ids restart at r1 in every thread.

"""


def render(th: dict, resp: list[dict], first: bool) -> str:
    body = th["selftext"] if len(th["selftext"]) <= MAX_REQ_CHARS else th["selftext"][:MAX_REQ_CHARS] + " [...]"
    parts = [f"REQUEST TITLE: {th['title']}", f"REQUEST BODY: {body if body.strip() else '(empty)'}", "", "RESPONSES:"]
    for i, r in enumerate(resp, 1):
        txt = "\n\n".join(c["body"] for c in r["comments"])
        txt = txt if len(txt) <= MAX_RESP_CHARS else txt[:MAX_RESP_CHARS] + " [...]"
        parts.append(f"[r{i}] {txt}")
    if not first:
        parts.append("\n(The request_books of this thread were extracted in an earlier call; return [] for them.)")
    return PROMPT + "\n".join(parts)


def load_threads(months: list[str]) -> list[dict]:
    out = []
    for ym in months:
        p = OUTB / "threads" / f"{ym}.jsonl"
        if p.exists():
            out += [json.loads(l) for l in p.open()]
    return out


def units_of(threads: list[dict]) -> list[tuple[dict, int]]:
    """(thread, chunk) units; a chunk holds <= MAX_RESP responders."""
    return [(th, c // MAX_RESP) for th in threads for c in range(0, len(th["responders"]), MAX_RESP)]


def unit_key(u: tuple[dict, int]) -> str:
    return f"{u[0]['sid']}:{u[1]}"


def single_request(u: tuple[dict, int]) -> dict:
    th, c = u
    rs = th["responders"][c * MAX_RESP:(c + 1) * MAX_RESP]
    return {"id": f"xv1:{unit_key(u)}", "max_tokens": 8000, "units": [u],
            "messages": [{"role": "user", "content": render(th, rs, c == 0)}]}


def packed_requests(units: list[tuple[dict, int]], pack: int) -> list[dict]:
    """Pack consecutive units into one call (the API limit is queries per minute): <= pack threads,
    <= MAX_RESP responders and <= PACK_CHARS characters of thread text per call."""
    groups, cur, n_resp, n_chars = [], [], 0, 0
    for u in units:
        th, c = u
        rs = th["responders"][c * MAX_RESP:(c + 1) * MAX_RESP]
        chars = len(render(th, rs, c == 0)) - len(PROMPT)
        if cur and (len(cur) >= pack or n_resp + len(rs) > MAX_RESP or n_chars + chars > PACK_CHARS):
            groups.append(cur)
            cur, n_resp, n_chars = [], 0, 0
        cur.append(u)
        n_resp += len(rs)
        n_chars += chars
    if cur:
        groups.append(cur)
    reqs = []
    for g in groups:
        if len(g) == 1:
            reqs.append(single_request(g[0]))
            continue
        parts = []
        for i, (th, c) in enumerate(g, 1):
            rs = th["responders"][c * MAX_RESP:(c + 1) * MAX_RESP]
            parts.append(f"===== THREAD t{i} =====\n" + render(th, rs, c == 0)[len(PROMPT):])
        key = hashlib.sha1("|".join(unit_key(u) for u in g).encode()).hexdigest()[:20]
        reqs.append({"id": f"x{VERSION}:{key}", "max_tokens": 16000, "units": g,
                     "messages": [{"role": "user", "content": PROMPT + PACK_NOTE.replace("{n}", str(len(g))) +
                                   "\n\n".join(parts)}]})
    return reqs


def unit_results(req: dict, text: str | None) -> dict[str, dict]:
    """unit key -> {"request_books", "responses"} parsed from one call's output (missing units are absent)."""
    j = parse_json_text(text) if text else None
    if not isinstance(j, dict):
        return {}
    if len(req["units"]) == 1 and "responses" in j:
        return {unit_key(req["units"][0]): j}
    out = {}
    for i, u in enumerate(req["units"], 1):
        v = j.get(f"t{i}")
        if isinstance(v, dict) and isinstance(v.get("responses"), dict):
            out[unit_key(u)] = v
    return out


def clean_list(x, keys: tuple[str, ...]) -> list[dict]:
    out = []
    for b in x if isinstance(x, list) else []:
        if isinstance(b, dict) and isinstance(b.get("title"), str) and b["title"].strip():
            out.append({k: b.get(k) for k in keys})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--pack", type=int, default=5, help="threads per call (1 = the single-thread v1 format)")
    a = ap.parse_args()
    threads = [t for t in load_threads(month_range(a.months)) if t["responders"]]
    if a.sample:
        threads = random.Random(42).sample(threads, min(a.sample, len(threads)))
    units = units_of(threads)
    reqs = packed_requests(units, a.pack) if a.pack > 1 else [single_request(u) for u in units]
    print(f"{len(threads)} threads, {len(units)} units, {len(reqs)} calls", flush=True)
    t0 = time.time()
    got: dict[str, dict] = {}
    res = batch_chat([{k: v for k, v in r.items() if k != "units"} for r in reqs], EXT / "api_cache.jsonl",
                     workers=a.workers)
    for r in reqs:
        got.update(unit_results(r, res.get(r["id"], {}).get("text")))
    for attempt in (1, 2):  # re-ask (new cache id, same messages) for calls whose units are missing / not valid JSON
        bad = [r for r in reqs if any(unit_key(u) not in got for u in r["units"])]
        if not bad:
            break
        retry = [{**r, "id": f"{r['id']}:retry{attempt}"} for r in bad]
        rr = batch_chat([{k: v for k, v in r.items() if k != "units"} for r in retry], EXT / "api_cache.jsonl",
                        workers=a.workers)
        for q in retry:
            for k, v in unit_results(q, rr.get(q["id"], {}).get("text")).items():
                got.setdefault(k, v)
        print(f"retry {attempt}: {len(bad)} calls", flush=True)
    missing = [u for u in units if unit_key(u) not in got]
    if missing:  # last resort: one unit per call in the single-thread format
        singles = [single_request(u) for u in missing]
        rr = batch_chat([{k: v for k, v in r.items() if k != "units"} for r in singles], EXT / "api_cache.jsonl",
                        workers=a.workers)
        for q in singles:
            got.update(unit_results(q, rr.get(q["id"], {}).get("text")))
        print(f"single-unit fallback: {len(missing)} units", flush=True)
    print(f"batch done in {time.time() - t0:.0f}s", flush=True)
    st = Counter({"calls": len(reqs), "units": len(units)})
    with (EXT / f"mentions_{a.tag}.jsonl").open("w") as f:
        for th in threads:
            rs = th["responders"]
            row = {"sid": th["sid"], "month": th["month"], "request_books": [], "responders": {}, "ok": True}
            for c in range(0, len(rs), MAX_RESP):
                j = got.get(f"{th['sid']}:{c // MAX_RESP}")
                if not isinstance(j, dict):
                    row["ok"] = False
                    st["units_missing_or_unparsed"] += 1
                    continue
                if c == 0:
                    row["request_books"] = clean_list(j.get("request_books"), ("title", "author", "year", "read"))
                resp = j.get("responses") if isinstance(j.get("responses"), dict) else {}
                for i, r in enumerate(rs[c:c + MAX_RESP], 1):
                    if f"r{i}" not in resp:
                        st["responder_missing_in_output"] += 1
                    row["responders"][r["author"]] = clean_list(resp.get(f"r{i}"),
                                                                ("title", "author", "year", "series"))
            st["threads"] += 1
            st["threads_ok"] += row["ok"]
            st["responders"] += len(rs)
            st["responders_with_book"] += sum(bool(v) for v in row["responders"].values())
            st["mentions"] += sum(len(v) for v in row["responders"].values())
            st["request_books"] += len(row["request_books"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    usage = log_usage("extraction", EXT / "api_cache.jsonl")
    print(json.dumps({"tag": a.tag, **st, "api_usage_cumulative": usage}, indent=1))


if __name__ == "__main__":
    main()
