"""GPT-5.6 teacher lists (20 artists) for the music training requests (the role of Rank-GRPO's teacher
lists, as books_sft_targets.py does for Reddit-Books).

The user message is exactly the dataset prompt (music header + "Here is the conversation: USER: ...\\n"). The system
message asks for exactly 20 lines "Artist Name (start year)" (birth year of a solo artist, formation year of a group),
artists different from those the requester names, active before the request's month. The SFT completion is built by
music_build.py from the grounded lines (MusicBrainz canonical name and begin year), so the teacher's own year is used
only for the registered year-matching check.

Usage (API only): env/bin/python remark/scripts/music_sft_targets.py --tag full [--limit 200]
Input: outputs/music/ground/grounded_<tag>.jsonl, threads/*.jsonl
Output: outputs/music/sft/api_cache.jsonl, teacher_<tag>.jsonl ({sid, lines: [[name, year]]})
"""

from __future__ import annotations

import argparse
import calendar
import json
import random
import re
import sys
import time

from books_common import del_markup, parse_json_text  # noqa: F401
from music_common import OUTM, REPO, prompt_text, split_of

sys.path.insert(0, str(REPO / "remark"))
from lib.llm import batch_chat  # noqa: E402

VERSION = "m1"
SFT = OUTM / "sft"
SYSTEM = ("You are a music recommender. Follow the user's instructions exactly: give exactly 20 lines, one artist or "
          "band per line, in the form \"Artist Name (start_year)\", where start_year is the birth year of a solo artist "
          "or the formation year of a group. Each line must be a real, specific artist or band, different from the "
          "artists the user names. The conversation took place in {date}: recommend artists who had released music "
          "before {date}. No numbering, no extra words.")
LINE = re.compile(r"^(.+?)\s+\((\d{4})\)\s*$")


def month_name(ym: str) -> str:
    y, m = ym.split("-")
    return f"{calendar.month_name[int(m)]} {y}"


def parse_teacher(text: str) -> list[list]:
    out = []
    for line in (text or "").splitlines():
        line = del_markup(line)
        m = LINE.match(line)
        if m:
            out.append([m.group(1).strip().strip('"').strip("*").strip(), int(m.group(2))])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    SFT.mkdir(parents=True, exist_ok=True)
    grounded = {}
    for line in (OUTM / "ground" / f"grounded_{a.tag}.jsonl").open():
        r = json.loads(line)
        if split_of(r["month"]) == "train" and any(v for v in r["responders"].values()):
            grounded[r["sid"]] = r["month"]
    ctx = {}
    for ym in sorted(set(grounded.values())):
        for line in (OUTM / "threads" / f"{ym}.jsonl").open():
            t = json.loads(line)
            if t["sid"] in grounded:
                ctx[t["sid"]] = t["context"]
    sids = sorted(ctx)
    random.Random(42).shuffle(sids)
    if a.limit:
        sids = sids[:a.limit]
    reqs = [{"id": f"t{VERSION}:{s}", "max_tokens": 3000,
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
    pt = ct = 0
    for line in open(SFT / "api_cache.jsonl"):
        try:
            u = json.loads(line).get("usage") or {}
        except json.JSONDecodeError:
            continue
        pt += u.get("prompt_tokens", 0) or 0; ct += u.get("completion_tokens", 0) or 0
    print(json.dumps({"threads": len(sids), "with_20_parsed_lines": n20, "prompt_tokens": pt, "completion_tokens": ct}, indent=1))


if __name__ == "__main__":
    main()
