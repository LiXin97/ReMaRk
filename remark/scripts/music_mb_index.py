"""Build the MusicBrainz artist index for the music dataset: normalized name -> artist MBIDs (CC0 data).

Source: data/raw/musicbrainz/artist.tar.xz (JSON dump 20260930-001001, one artist per line inside mbdump/artist).
For every artist: its name, sort name (reordered "Last, First" -> "First Last") and every alias name are normalized
(casefold, NFKD accents stripped, "&"/"+" -> "and", punctuation removed, whitespace collapsed, a leading "the "
dropped) and mapped to the MBID. A name that maps to several MBIDs is ambiguous; the match rule (music_ground.py) drops
ambiguous names unless one candidate clearly dominates by community activity (see there). Kept per MBID: canonical
name, type, disambiguation and an activity score = rating votes + the summed counts of its tags and genres (the dump
has no play counts; tags and votes are what MusicBrainz editors add to artists people listen to).

Run on a node (CPU, streaming xz): python3 remark/scripts/music_mb_index.py
Output: remark/data/music/mb_index.pkl  ({"names": {norm: [mbid, ...]}, "artists": {mbid: [name, type, disamb, activity]}})
"""

from __future__ import annotations

import json
import pickle
import re
import subprocess
import tarfile
import unicodedata
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/raw/musicbrainz/artist.tar.xz"
OUT = REPO / "remark/data/music/mb_index.pkl"
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).casefold()
    s = s.replace("&", " and ").replace("+", " and ")
    s = _WS.sub(" ", _PUNCT.sub(" ", s)).strip()
    if s.startswith("the "):
        s = s[4:]
    return s


def unsort(sort_name: str) -> str:
    """'Beatles, The' -> 'The Beatles'; 'Cave, Nick' -> 'Nick Cave'."""
    if ", " in sort_name and sort_name.count(", ") == 1:
        a, b = sort_name.split(", ")
        return f"{b} {a}"
    return sort_name


def main() -> None:
    names: dict[str, set] = defaultdict(set)
    artists: dict[str, list] = {}
    n = 0
    # stream through xz (multi-threaded) rather than tarfile's slow single-thread lzma
    proc = subprocess.Popen(["xz", "-dc", "-T0", str(SRC)], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for m in tar:
            if not m.isfile() or not m.name.endswith("mbdump/artist"):
                continue
            f = tar.extractfile(m)
            for line in f:
                a = json.loads(line)
                mbid = a["id"]
                act = ((a.get("rating") or {}).get("votes-count") or 0) + sum(t.get("count", 0) for t in a.get("tags") or []) \
                    + sum(g.get("count", 0) for g in a.get("genres") or [])
                artists[mbid] = [a.get("name"), a.get("type"), a.get("disambiguation") or "", act]
                cands = {a.get("name") or "", unsort(a.get("sort-name") or "")}
                cands |= {al.get("name") or "" for al in a.get("aliases") or []}
                for c in cands:
                    k = norm(c)
                    if k:
                        names[k].add(mbid)
                n += 1
                if n % 500000 == 0:
                    print(f"{n} artists, {len(names)} names", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump({"names": {k: sorted(v) for k, v in names.items()}, "artists": artists,
                     "source": str(SRC.relative_to(REPO)), "dump": "json-dumps/20260930-001001"}, f)
    amb = sum(1 for v in names.values() if len(v) > 1)
    print(json.dumps({"artists": n, "names": len(names), "ambiguous_names": amb}), flush=True)


if __name__ == "__main__":
    main()
