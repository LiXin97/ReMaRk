"""Ground extracted artist mentions to MusicBrainz artist MBIDs (match rule, registered before the audit).

Normalization (music_mb_index.norm): casefold, NFKD accents stripped, "&"/"+" -> "and", punctuation removed, spaces
collapsed, a leading "the" dropped. Resolution of a mention, in order:
  1. exactly one MBID carries the normalized name (name, reordered sort name or alias): that MBID ("unique");
  2. several do, and one of them is also matched uniquely by another mention in the same thread: that one
     ("thread");
  3. several do, and the most active candidate (rating votes + tag and genre counts in the dump) has at least
     DOMINANCE times the activity of the runner-up and at least MIN_ACT: that one ("dominant"). Popular band names are
     shared by obscure artists (several MusicBrainz "Nirvana"s), and dropping them would drop the most recommended
     artists; the dominance margin keeps genuinely contested names out;
  4. otherwise the mention is dropped ("ambiguous_dropped"); no candidate at all is "no_match".
If the extracted name does not match and contains " feat"/" ft."/" featuring "/" x "/" with ", the part before it is
tried once. Shares of each outcome are reported.

Usage: env/bin/python remark/scripts/music_ground.py --tag pilot
Output: remark/outputs/music/ground/grounded_<tag>.jsonl (per thread: request MBIDs, per responder MBIDs in order)
        + ground_stats_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

from music_mb_index import norm

REPO = Path(__file__).resolve().parents[2]
OUTM = REPO / "remark/outputs/music"
INDEX = REPO / "remark/data/music/mb_index.pkl"
DOMINANCE, MIN_ACT = 5.0, 3
SPLIT = re.compile(r"\s+(?:feat\.?|ft\.?|featuring|x|with)\s+", re.I)


def candidates(names: dict, artist: str) -> list[str]:
    c = names.get(norm(artist), [])
    if not c:
        parts = SPLIT.split(artist, maxsplit=1)
        if len(parts) > 1:
            c = names.get(norm(parts[0]), [])
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    idx = pickle.load(open(INDEX, "rb"))
    names, artists = idx["names"], idx["artists"]
    st = Counter()
    out_dir = OUTM / "ground"; out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in open(OUTM / "extract" / f"mentions_{a.tag}.jsonl"):
        th = json.loads(line)
        # first pass: unique matches in this thread, to resolve ambiguous names by co-mention
        cand = {}
        thread_unique = Counter()
        for who, ms in [("REQ", th["request_artists"])] + list(th["responders"].items()):
            for m in ms:
                c = candidates(names, m["artist"])
                cand[(who, m["artist"])] = c
                if len(c) == 1:
                    thread_unique[c[0]] += 1

        def resolve(c: list[str]) -> tuple[str | None, str]:
            if len(c) == 1:
                return c[0], "unique"
            if not c:
                return None, "no_match"
            hits = [x for x in c if thread_unique[x] > 0]
            if len(hits) == 1:
                return hits[0], "thread"
            acts = sorted(((artists[x][3] if x in artists else 0), x) for x in c)
            top, second = acts[-1], acts[-2]
            if top[0] >= MIN_ACT and top[0] >= DOMINANCE * max(second[0], 0.5):
                return top[1], "dominant"
            return None, "ambiguous_dropped"

        req = []
        for m in th["request_artists"]:
            mb, how = resolve(cand[("REQ", m["artist"])])
            st[f"request_{how}"] += 1
            if mb:
                req.append(mb)
        resp = {}
        for who, ms in th["responders"].items():
            got = []
            for m in ms:
                mb, how = resolve(cand[(who, m["artist"])])
                st[f"mention_{how}"] += 1
                st["mention_song" if m.get("song") else "mention_bare_artist"] += 1
                if mb and mb not in got:
                    got.append(mb)
            resp[who] = got
        rows.append({"sid": th["sid"], "month": th["month"], "n_responders_all": th.get("n_responders_all"),
                     "request_mbids": sorted(set(req)), "responders": resp})
    with open(out_dir / f"grounded_{a.tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = sum(st[k] for k in st if k.startswith("mention_") and k.split("_", 1)[1] in
            ("unique", "no_match", "thread", "dominant", "ambiguous_dropped"))
    rep = {"threads": len(rows), "mentions": n, **dict(st),
           "share_unique": round(st["mention_unique"] / max(n, 1), 4),
           "share_thread": round(st["mention_thread"] / max(n, 1), 4),
           "share_dominant": round(st["mention_dominant"] / max(n, 1), 4),
           "share_ambiguous_dropped": round(st["mention_ambiguous_dropped"] / max(n, 1), 4),
           "share_no_match": round(st["mention_no_match"] / max(n, 1), 4),
           "share_song_mentions": round(st["mention_song"] / max(n, 1), 4)}
    (out_dir / f"ground_stats_{a.tag}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
