"""Shared pieces of the music music dataset: the instruction header, item rendering and parsing.

Items are rendered "Artist Name (Year)" with the MusicBrainz begin year (birth year of a solo artist, formation year
of a group), so Rank-GRPO's "Title (Year)" parser, the rewards and crs_eval work unchanged; a rendered item is kept only
if the parser reads it back exactly (canon_ok).
"""

from __future__ import annotations

import sys
from pathlib import Path

from books_common import REDDIT_HEADER, SEP, parse_lines

REPO = Path(__file__).resolve().parents[2]
OUTM = REPO / "remark/outputs/music"
HEADER = (REDDIT_HEADER.replace("a movie recommender", "a music recommender")
          .replace("the standardized English title of each movie", "the standard name of each artist or band")
          .replace('"movie name" (release_year)', '"artist name" (start_year)'))
assert "movie" not in HEADER and HEADER.count("artist") == 2, HEADER


def prompt_text(context: str) -> str:
    return HEADER + SEP + context


def render(name: str, year: int) -> str:
    return f"{name} ({year})"


def canon_ok(name: str, year: int) -> bool:
    """True when Rank-GRPO's parser reads the rendered line back as exactly (name, year)."""
    got = parse_lines(render(name, year))
    return len(got) == 1 and got[0][0] == name and int(got[0][1]) == year


def split_of(month: str) -> str | None:
    if "2023-07" <= month <= "2024-06":
        return "train"
    return {"2024-07": "validation", "2024-08": "test"}.get(month)
