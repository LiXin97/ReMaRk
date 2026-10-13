"""MusicBrainz begin years for the music item rendering "Artist (Year)".

The Reddit-v2 / Reddit-Books item format is "Title (Year)", which Rank-GRPO's parser, the rewards and crs_eval all
assume, matching a title and a year within 2. For artists the year is the MusicBrainz life-span begin year: birth year
of a person, formation year of a group. Artists without a begin year cannot be rendered.

Run on a node (CPU, streaming xz): python3 remark/scripts/music_mb_years.py
Output: remark/data/music/mb_years.pkl  {mbid: [canonical name, begin year (int) or None, type]}
"""

from __future__ import annotations

import json
import pickle
import subprocess
import tarfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/raw/musicbrainz/artist.tar.xz"
OUT = REPO / "remark/data/music/mb_years.pkl"


def main() -> None:
    out, st = {}, Counter()
    proc = subprocess.Popen(["xz", "-dc", "-T0", str(SRC)], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for m in tar:
            if not m.isfile() or not m.name.endswith("mbdump/artist"):
                continue
            for line in tar.extractfile(m):
                a = json.loads(line)
                b = ((a.get("life-span") or {}).get("begin") or "")[:4]
                y = int(b) if b.isdigit() and 1000 <= int(b) <= 2026 else None
                out[a["id"]] = [a.get("name"), y, a.get("type")]
                st["artists"] += 1
                st["with_year"] += y is not None
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print(json.dumps(dict(st)), flush=True)


if __name__ == "__main__":
    main()
