"""Shared helpers for the Reddit-Books build: title canonicalization, normalization keys, API usage log."""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

# ROOT = the tree this copy of the code lives in; all outputs are written under it. REPO = the project repo, read
# for shared inputs (Reddit-v2 reference data, Rank-GRPO parser, tokenizer, lib/llm.py). They differ only when the
# code runs from a node-local mirror (e.g. while the repo is read-only): WORK_REPO=<repo root> python3 <mirror>/...
ROOT = Path(__file__).resolve().parents[2]
REPO = Path(os.environ.get("WORK_REPO", str(ROOT)))
OUTB = ROOT / "remark/outputs/books"
DATA = ROOT / "remark/data/crs/reddit_books"
sys.path.insert(0, str(REPO / "remark/third_party/Rank-GRPO/libs"))
from utils import del_format, del_space, process_rec_raw, remove_quotes  # noqa: E402  (Rank-GRPO's title parser)

CFG_PATH = Path(os.environ["BOOKS_CFG"]) if os.environ.get("BOOKS_CFG") else ROOT / "remark/configs/crs_reddit_books.yaml"  # C
CFG = yaml.safe_load(open(CFG_PATH))


def split_of(month: str) -> str | None:
    for s, (a, b) in CFG["splits"].items():
        if a <= month <= b:
            return s
    return None


def del_markup(line: str) -> str:
    """Numbering, bullets, markdown and quotes removed exactly as Rank-GRPO's parser does before matching."""
    return remove_quotes(del_format(del_space(line)))

SEP = "Here is the conversation: "
MIN_YEAR, MAX_YEAR = 1000, 2018  # "(\d{4})" parser; UCSD Goodreads was crawled in late 2017
# Reddit-v2 instruction header (verbatim; books_build.py asserts it against the reference prompts) and the minimal
# movie -> book adaptation used for Reddit-Books (same "Title (Year)" output format).
REDDIT_HEADER = ('Pretend you are a movie recommender system.\nI will give you a conversation between a user and you '
                 '(a recommender system). Based on the conversation, you need to reply with 20 recommendations. List '
                 'the standardized English title of each movie in each line in the form of "movie name" '
                 '(release_year) with NO extra words or sentences.\n')
HEADER = (REDDIT_HEADER.replace("a movie recommender", "a book recommender").replace("each movie", "each book")
          .replace('"movie name" (release_year)', '"book title" (first_publication_year)'))
assert HEADER.count("book") == 3 and "movie" not in HEADER


def prompt_text(context: str) -> str:
    return HEADER + SEP + context


def ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def parse_lines(text: str) -> list[tuple[str, int]]:
    err, item = process_rec_raw({"raw": text}, "raw", "rec")
    return [] if err else [(t, int(y)) for t, y in item["rec"]]


@lru_cache(maxsize=None)
def _parse_title_once(t: str) -> str | None:
    rec = parse_lines(f"{t} (2000)")
    return rec[0][0] if len(rec) == 1 and rec[0][1] == 2000 and rec[0][0] else None


@lru_cache(maxsize=None)
def canon(title: str) -> str | None:
    """Title as Rank-GRPO's parser outputs it for the line '<title> (YYYY)'; a fixed point of the parser.
    (Same title canonicalisation as the Rank-GRPO parser, after mapping typographic quotes/dashes to ASCII so that
    catalog titles do not mix "’" and "'".)"""
    t = ws((title or "").translate(_TYPO))
    while True:  # the parser deletes parenthesized groups; delete first so no inner "(dddd)" survives
        t2 = ws(re.sub(r"\([^()]*\)", "", t))
        if t2 == t:
            break
        t = t2
    for _ in range(5):
        if not t:
            return None
        c = _parse_title_once(t)
        if c == t:
            return t
        t = c or ""
    return None


_ARTICLE = re.compile(r"^(the|a|an)\s+")
_TYPO = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "…": "..."})
_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "ł": "l",
                           "Ł": "L", "đ": "d", "Đ": "D", "þ": "th", "Þ": "Th", "ð": "d", "ı": "i"})


@lru_cache(maxsize=1_000_000)
def norm_key(s: str) -> str:
    """Matching key: accents folded, lowercase, '&'->and, bracketed groups and punctuation removed, articles dropped."""
    s = unicodedata.normalize("NFKD", (s or "").translate(_TRANSLIT))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", s)
    s = s.replace("&", " and ").replace("’", "'")
    s = re.sub(r"'s\b", "s", s)
    s = re.sub(r"\bvolume\b", "vol", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return _ARTICLE.sub("", s)


def main_title(s: str) -> str:
    """Title without subtitle (text after the first ': ' or ' - ')."""
    return re.split(r":\s|\s[-–—]\s", s or "", maxsplit=1)[0]


def surname_tokens(name: str | None) -> set[str]:
    """Last name token of each author in 'A and B', 'A, B & C', 'A with B'."""
    out = set()
    for seg in re.split(r",|&|\band\b|\bwith\b|;|/", name or ""):
        toks = [t for t in norm_key(seg).split() if t not in {"jr", "sr", "ii", "iii", "phd", "md", "dr"}]
        if toks and not (len(toks) == 1 and len(toks[0]) <= 2):
            out.add(toks[-1])
    return out


def name_tokens(name: str) -> set[str]:
    return set(norm_key(name).split())


def rel(p: Path) -> str:
    """Path relative to the tree it lives in (ROOT or REPO), for manifests."""
    for base in (ROOT, REPO):
        try:
            return str(Path(p).resolve().relative_to(base))
        except ValueError:
            continue
    return str(p)


def log_usage(stage: str, cache_path: Path) -> dict:
    """Aggregate call counts and token usage of one batch_chat cache into remark/outputs/books/api_usage.json."""
    n = err = pt = ct = rt = 0
    if cache_path.exists():
        for line in cache_path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                err += 1
                continue
            n += 1
            u = r.get("usage") or {}
            pt += u.get("prompt_tokens", 0) or 0
            ct += u.get("completion_tokens", 0) or 0
            rt += ((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
    path = OUTB / "api_usage.json"
    allu = json.loads(path.read_text()) if path.exists() else {}
    allu[stage] = {"cache": rel(cache_path), "calls_ok": n, "error_records": err,
                   "prompt_tokens": pt, "completion_tokens": ct, "reasoning_tokens": rt, "model": "gpt-5.6"}
    allu["_total"] = {k: sum(v[k] for s, v in allu.items() if not s.startswith("_"))
                      for k in ("calls_ok", "error_records", "prompt_tokens", "completion_tokens", "reasoning_tokens")}
    path.write_text(json.dumps(allu, indent=1))
    return allu[stage]


def parse_json_text(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\{.*", t, re.S)
    t = m.group(0).strip() if m else t
    for suffix in ("", "}", "]}", "}}", "]}}", "}]}}"):  # repair a missing closing brace at the end
        try:
            return json.loads(t + suffix)
        except json.JSONDecodeError:
            continue
    return None
