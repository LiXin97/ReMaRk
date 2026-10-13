"""GPT-5.6 mention extraction for the music dataset, adapted from books_extract.py.

Unit = one r/musicsuggestions thread with its first MAX_RESP responders in time order (the responder cap of the E12
registration; threads have 38 responders on average and the cap bounds both the API cost and the size of a request's
union of positives). Consecutive threads are packed into one call as for Reddit-Books. Strict JSON output:
request_artists (artists the requester names, with a `liked` flag) and, per responder, the artists that responder
recommends, each with the song named for it (or null when the reply names the artist only). The item of the dataset is
the ARTIST (registered rule); the song is kept only to report the song/artist share and to resolve the artist.

Usage (API only):
  env/bin/python remark/scripts/music_extract.py --months 2024-06 --sample 200 --tag pilot
  env/bin/python remark/scripts/music_extract.py --months 2023-01:2024-12 --tag full
Outputs: remark/outputs/music/extract/api_cache.jsonl (all calls), mentions_<tag>.jsonl (one row per thread),
usage in remark/outputs/music/api_usage.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from books_common import parse_json_text
from books_pull import month_range

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "remark"))
from lib.llm import batch_chat  # noqa: E402

OUTM = REPO / "remark/outputs/music"
EXT = OUTM / "extract"
VERSION = "m1"
MAX_RESP = 20
PACK_CHARS = 20000
MAX_REQ_CHARS, MAX_RESP_CHARS = 4000, 1500

PROMPT = """You extract music recommendations from a Reddit r/musicsuggestions thread. The thread has a REQUEST (the \
original post) and RESPONSES (top-level replies by other users; each response id is one user, with that user's \
replies merged).

Return strict JSON of this shape:
{"request_artists": [{"artist": str, "liked": true or false}],
 "responses": {"r1": [{"artist": str, "song": str or null}], "r2": [], ...}}

Rules:
1. request_artists: every specific artist or band the requester names (as examples of their taste, things they like \
or dislike, or artists they already know). liked = false only if the requester says they dislike or want to avoid \
the artist. If the requester names a song, give its artist.
2. responses: for EVERY response id, the artists that response recommends to the requester, in order of appearance. \
Include an artist when the responder suggests them or one of their songs or albums as something to listen to, \
including replies that are only a name, a song ("Artist - Song", "Song by Artist"), a list, or a streaming link \
whose text names the song or artist. Exclude artists mentioned only for comparison or context (e.g. the requester's \
own examples, "it sounds like X"), artists the responder advises against, and artists merely mentioned without \
being suggested. Use [] when a response recommends no specific artist.
3. song: the song title when the response names a specific song (or "album: <title>" when it names an album), else \
null. If a response names several songs by the same artist, list the artist once with the first song.
4. artist: the artist or band's standard name, correctly spelled and capitalized, without quotes or markdown (fix \
obvious typos). For "feat." / "ft." / "featuring" credits give only the main artist. For a group give the group's \
name, not its members. If the response names only a song and you are confident who performs it, give that artist; \
otherwise skip the song.
5. Only music artists: not films, games, books, podcasts, genres, playlists, labels or radio stations.
6. Never add artists that the thread does not mention. Output the JSON object only.

"""

PACK_NOTE = """This input contains {n} independent threads (t1, t2, ...). Apply the rules to each thread separately \
and return ONE JSON object keyed by thread id, each value being the object described above for that thread: \
{"t1": {"request_artists": [...], "responses": {...}}, "t2": {...}, ...}. Response ids restart at r1 in every thread.

"""


def render(th: dict, resp: list[dict]) -> str:
    body = th["selftext"] if len(th["selftext"]) <= MAX_REQ_CHARS else th["selftext"][:MAX_REQ_CHARS] + " [...]"
    parts = [f"REQUEST TITLE: {th['title']}", f"REQUEST BODY: {body if body.strip() else '(empty)'}", "", "RESPONSES:"]
    for i, r in enumerate(resp, 1):
        txt = "\n\n".join(c["body"] for c in r["comments"])
        txt = txt if len(txt) <= MAX_RESP_CHARS else txt[:MAX_RESP_CHARS] + " [...]"
        parts.append(f"[r{i}] {txt}")
    return "\n".join(parts)


def first_responders(th: dict) -> list[dict]:
    """The registered cap: the first MAX_RESP responders by the time of their first top-level comment."""
    rs = sorted(th["responders"], key=lambda r: min(c["created_utc"] for c in r["comments"]))
    return rs[:MAX_RESP]


def load_threads(months: list[str]) -> list[dict]:
    out = []
    for ym in months:
        p = OUTM / "threads" / f"{ym}.jsonl"
        if p.exists():
            out += [json.loads(l) for l in p.open()]
    return out


def requests_for(threads: list[dict], pack: int) -> list[dict]:
    groups, cur, n_chars = [], [], 0
    for th in threads:
        chars = len(render(th, first_responders(th)))
        if cur and (len(cur) >= pack or n_chars + chars > PACK_CHARS):
            groups.append(cur); cur, n_chars = [], 0
        cur.append(th); n_chars += chars
    if cur:
        groups.append(cur)
    reqs = []
    for g in groups:
        if len(g) == 1:
            body = PROMPT + render(g[0], first_responders(g[0]))
        else:
            body = PROMPT + PACK_NOTE.replace("{n}", str(len(g))) + "\n\n".join(
                f"===== THREAD t{i} =====\n" + render(th, first_responders(th)) for i, th in enumerate(g, 1))
        key = hashlib.sha1("|".join(th["sid"] for th in g).encode()).hexdigest()[:20]
        reqs.append({"id": f"{VERSION}:{key}", "max_tokens": 12000, "sids": [th["sid"] for th in g],
                     "messages": [{"role": "user", "content": body}]})
    return reqs


def results_of(req: dict, text: str | None) -> dict[str, dict]:
    j = parse_json_text(text) if text else None
    if not isinstance(j, dict):
        return {}
    if len(req["sids"]) == 1 and "responses" in j:
        return {req["sids"][0]: j}
    out = {}
    for i, sid in enumerate(req["sids"], 1):
        v = j.get(f"t{i}")
        if isinstance(v, dict) and isinstance(v.get("responses"), dict):
            out[sid] = v
    return out


def clean(x) -> list[dict]:
    out = []
    for b in x if isinstance(x, list) else []:
        if isinstance(b, dict) and isinstance(b.get("artist"), str) and b["artist"].strip():
            out.append({"artist": b["artist"].strip(), "song": b.get("song") if isinstance(b.get("song"), str) else None,
                        **({"liked": b.get("liked")} if "liked" in b else {})})
    return out


def log_usage(cache: Path) -> dict:
    n = pt = ct = 0
    if cache.exists():
        for line in cache.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            n += 1; u = r.get("usage") or {}
            pt += u.get("prompt_tokens", 0) or 0; ct += u.get("completion_tokens", 0) or 0
    u = {"calls": n, "prompt_tokens": pt, "completion_tokens": ct}
    path = OUTM / "api_usage.json"
    allu = json.loads(path.read_text()) if path.exists() else {}
    allu["extraction"] = u
    path.write_text(json.dumps(allu, indent=2))
    return u


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--pack", type=int, default=4)
    a = ap.parse_args()
    EXT.mkdir(parents=True, exist_ok=True)
    threads = [t for t in load_threads(month_range(a.months)) if t["responders"]]
    if a.sample:
        threads = random.Random(42).sample(threads, min(a.sample, len(threads)))
    reqs = requests_for(threads, a.pack)
    print(f"{len(threads)} threads, {len(reqs)} calls", flush=True)
    t0 = time.time()
    got: dict[str, dict] = {}
    res = batch_chat([{k: v for k, v in r.items() if k != "sids"} for r in reqs], EXT / "api_cache.jsonl", workers=a.workers)
    for r in reqs:
        got.update(results_of(r, res.get(r["id"], {}).get("text")))
    for attempt in (1, 2):
        bad = [r for r in reqs if any(s not in got for s in r["sids"])]
        if not bad:
            break
        retry = [{**r, "id": f"{r['id']}:retry{attempt}"} for r in bad]
        rr = batch_chat([{k: v for k, v in r.items() if k != "sids"} for r in retry], EXT / "api_cache.jsonl", workers=a.workers)
        for q in retry:
            for k, v in results_of(q, rr.get(q["id"], {}).get("text")).items():
                got.setdefault(k, v)
        print(f"retry {attempt}: {len(bad)} calls", flush=True)
    st = Counter({"calls": len(reqs)})
    with (EXT / f"mentions_{a.tag}.jsonl").open("w") as f:
        for th in threads:
            rs = first_responders(th)
            j = got.get(th["sid"])
            row = {"sid": th["sid"], "month": th["month"], "n_responders_all": len(th["responders"]),
                   "request_artists": [], "responders": {}, "ok": isinstance(j, dict)}
            if isinstance(j, dict):
                row["request_artists"] = clean(j.get("request_artists"))
                resp = j.get("responses") if isinstance(j.get("responses"), dict) else {}
                for i, r in enumerate(rs, 1):
                    if f"r{i}" not in resp:
                        st["responder_missing_in_output"] += 1
                    row["responders"][r["author"]] = clean(resp.get(f"r{i}"))
            st["threads"] += 1; st["threads_ok"] += row["ok"]; st["responders"] += len(rs)
            st["responders_with_artist"] += sum(bool(v) for v in row["responders"].values())
            ms = [m for v in row["responders"].values() for m in v]
            st["mentions"] += len(ms); st["mentions_with_song"] += sum(m["song"] is not None for m in ms)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    usage = log_usage(EXT / "api_cache.jsonl")
    print(json.dumps({"tag": a.tag, "seconds": round(time.time() - t0), **st, "api_usage_cumulative": usage}, indent=1))


if __name__ == "__main__":
    main()
