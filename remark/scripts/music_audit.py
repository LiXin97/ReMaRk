"""Music extraction audit (gate: precision >= 0.95, recall >= 0.90 on 100 pairs), the music version of the Books recall audit.

Population: every (thread, capped responder) pair of a mentions file, INCLUDING responders with no extracted artist.
Sample --n pairs (seed 42). Judge: Gemini (gemini-2.5-pro, temperature 0; a different model family from the GPT-5.6
extractor) lists the artists the reply recommends. Matching: music_mb_index.norm of the two names equal, or equal after
dropping a "feat./ft./featuring/x/with ..." tail on either side; the adjusted variant also accepts a SequenceMatcher
ratio >= 0.9. Metrics: precision = extracted artists matched by a judged artist / extracted artists; recall = judged
artists matched by an extracted artist / judged artists (micro, Wilson 95% CIs); grounded recall additionally requires
the extracted artist to have been grounded to an MBID (music_ground.py output).

Usage: env/bin/python remark/scripts/music_audit.py --tag pilot [--n 100]
Output: remark/outputs/music/audit/audit_<tag>.json, sample_<tag>.json, gemini_cache_<tag>.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from music_mb_index import norm  # noqa: E402
from n917_e9_books_recall import gemini, wilson  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUTM = REPO / "remark/outputs/music"
AUD = OUTM / "audit"
TAIL = re.compile(r"\s+(?:feat\.?|ft\.?|featuring|x|with)\s+.*$", re.I)
PROMPT = """Below is a music request posted on Reddit (r/musicsuggestions) and one user's reply to it.

List every music artist or band that the reply recommends to the requester, including artists recommended through one
of their songs or albums (give the performing artist) and artists only named briefly as a suggestion. Exclude artists
that the requester mentioned as their own examples (unless the reply recommends them again), artists the reply advises
against or only uses for comparison, and anything that is not a music artist. Use only the text of the reply: do
not open, guess or reconstruct the contents of links or playlists, and do not list an artist the text does not name
or identify through a song or album title.

Request:
<<<
{request}
>>>

Reply:
<<<
{reply}
>>>

Return JSON only: {{"artists": ["...", "..."]}}. Return {{"artists": []}} if the reply recommends no artist."""


def keys(name: str) -> set[str]:
    k = {norm(name)}
    k.add(norm(TAIL.sub("", name)))
    return {x for x in k if x}


def match(a: str, b: str, adjusted: bool) -> bool:
    if keys(a) & keys(b):
        return True
    return adjusted and difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--judge_tag", default="v2", help="judge prompt version (v2: text only, no link contents)")
    a = ap.parse_args()
    key = os.environ["GEMINI_API_KEY"]
    AUD.mkdir(parents=True, exist_ok=True)
    threads = {}
    for p in sorted((OUTM / "threads").glob("*.jsonl")):
        for line in open(p):
            t = json.loads(line)
            threads[t["sid"]] = t
    grounded = {}
    gpath = OUTM / "ground" / f"grounded_{a.tag}.jsonl"
    if gpath.exists():
        for line in open(gpath):
            r = json.loads(line)
            grounded[r["sid"]] = r
    pop = []
    for line in open(OUTM / "extract" / f"mentions_{a.tag}.jsonl"):
        m = json.loads(line)
        if not m["ok"]:
            continue
        t = threads[m["sid"]]
        req = (t["title"] + "\n\n" + (t.get("selftext") or "")).strip()
        bodies = {r["author"]: "\n\n".join(c["body"] for c in r["comments"]) for r in t["responders"]}
        for author, arts in m["responders"].items():
            pop.append({"sid": m["sid"], "author": author, "request": req[:4000], "reply": bodies.get(author, "")[:3000],
                        "extracted": [x["artist"] for x in arts],
                        "n_grounded": len((grounded.get(m["sid"]) or {}).get("responders", {}).get(author, []))})
    sample = random.Random(42).sample(pop, min(a.n, len(pop)))
    (AUD / f"sample_{a.tag}.json").write_text(json.dumps(sample, indent=1, ensure_ascii=False))
    cache_p = AUD / f"gemini_cache_{a.tag}_{a.judge_tag}.jsonl"
    cache = {}
    if cache_p.exists():
        for line in open(cache_p):
            r = json.loads(line); cache[r["i"]] = r["text"]

    def ask(i: int) -> tuple[int, str]:
        if i in cache:
            return i, cache[i]
        s = sample[i]
        return i, gemini(a.model, PROMPT.format(request=s["request"], reply=s["reply"]), key)

    with ThreadPoolExecutor(8) as pool, open(cache_p, "a") as f:
        for i, text in pool.map(ask, range(len(sample))):
            if i not in cache:
                f.write(json.dumps({"i": i, "text": text}) + "\n"); f.flush(); cache[i] = text
    res = {}
    for adjusted in (False, True):
        tp_e = n_e = tp_j = n_j = 0
        for i, s in enumerate(sample):
            try:
                judged = json.loads(cache[i]).get("artists") or []
            except (json.JSONDecodeError, AttributeError):
                judged = []
            judged = [x for x in judged if isinstance(x, str) and x.strip()]
            ext = s["extracted"]
            n_e += len(ext); tp_e += sum(any(match(e, j, adjusted) for j in judged) for e in ext)
            n_j += len(judged); tp_j += sum(any(match(j, e, adjusted) for e in ext) for j in judged)
        tag = "adjusted" if adjusted else "strict"
        res[tag] = {"precision": round(tp_e / max(n_e, 1), 4), "precision_ci": [round(x, 4) for x in wilson(tp_e, n_e)],
                    "recall": round(tp_j / max(n_j, 1), 4), "recall_ci": [round(x, 4) for x in wilson(tp_j, n_j)],
                    "extracted": n_e, "judged": n_j}
    out = {"tag": a.tag, "judge_prompt": a.judge_tag, "pairs": len(sample), "judge": a.model, **res,
           "gate": {"precision>=0.95": res["adjusted"]["precision"] >= 0.95, "recall>=0.90": res["adjusted"]["recall"] >= 0.90}}
    (AUD / f"audit_{a.tag}_{a.judge_tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
