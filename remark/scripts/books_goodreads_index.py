"""Build a title(+author) index of the UCSD Goodreads book graph for grounding Reddit-Books mentions.

Inputs: data/raw/goodreads/goodreads_books.json.gz (2.36M editions), goodreads_book_authors.json.gz, and
remark/outputs/books/goodreads/goodreads_book_works.json.gz (works file of the same UCSD release; it carries
original_publication_year, which the local copy lacked).

Per work: display title = title_without_series of the work's best_book (else of its most-rated edition); titles = distinct
edition titles (title_without_series, most-rated edition first, <= 30) + original_title, for choosing the surface form;
year = works original_publication_year if in [MIN_YEAR, MAX_YEAR], else the earliest valid edition publication_year
(year_source records which); authors = primary authors (role "") over all editions; popularity = works ratings_count.
Keys: norm_key of every edition's title and title_without_series and of the work's original_title, each also without
subtitle ("main"). Also reports the crawl horizon: editions by (publication_year, publication_month) for 2016-2018.

Heavy CPU/RAM: run on a GPU node's CPUs (GPU node), e.g.
  python3 remark/scripts/books_goodreads_index.py
Outputs: remark/outputs/books/goodreads/{works.parquet, keys.parquet, index_stats.json}
"""

from __future__ import annotations

import gzip
import json
import subprocess
import time
from collections import Counter, defaultdict

import orjson
import pandas as pd

from books_common import MAX_YEAR, MIN_YEAR, OUTB, REPO, main_title, norm_key, ws

RAW = REPO / "data/raw/goodreads"
GD = OUTB / "goodreads"


def lines(path):
    p = subprocess.Popen(["zcat", str(path)], stdout=subprocess.PIPE, bufsize=1 << 24)
    for line in p.stdout:
        yield orjson.loads(line)
    p.wait()


def as_int(x) -> int | None:
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def main() -> None:
    t0 = time.time()
    authors = {r["author_id"]: r["name"] for r in lines(RAW / "goodreads_book_authors.json.gz")}
    works = {}
    for r in lines(GD / "goodreads_book_works.json.gz"):
        works[r["work_id"]] = {"orig_year": as_int(r.get("original_publication_year")),
                               "orig_title": ws(r.get("original_title")), "best_book_id": r.get("best_book_id"),
                               "work_ratings": as_int(r.get("ratings_count")) or 0,
                               "books_count": as_int(r.get("books_count")) or 0}
    print(f"authors {len(authors)} works {len(works)} {time.time() - t0:.0f}s", flush=True)

    ed_title: dict[str, tuple[int, str]] = {}          # work -> (ratings, title_without_series) of most-rated edition
    surfaces: dict[str, dict[str, int]] = defaultdict(dict)  # work -> edition title -> max edition ratings
    best_title: dict[str, str] = {}
    min_year: dict[str, int] = {}
    wauth: dict[str, Counter] = defaultdict(Counter)   # work -> primary author name counts
    keys: dict[tuple[str, str], set] = defaultdict(set)  # (key, kind) -> work ids
    horizon = Counter()
    n_ed = 0
    for b in lines(RAW / "goodreads_books.json.gz"):
        n_ed += 1
        w = b.get("work_id") or ""
        if not w:
            continue
        tws, t = ws(b.get("title_without_series")), ws(b.get("title"))
        r = as_int(b.get("ratings_count")) or 0
        if tws and (w not in ed_title or r > ed_title[w][0]):
            ed_title[w] = (r, tws)
        if tws:
            surfaces[w][tws] = max(r, surfaces[w].get(tws, -1))
        if works.get(w, {}).get("best_book_id") == b.get("book_id") and tws:
            best_title[w] = tws
        y, m = as_int(b.get("publication_year")), as_int(b.get("publication_month"))
        if y and MIN_YEAR <= y <= MAX_YEAR:
            min_year[w] = min(min_year.get(w, 9999), y)
        if y and 2016 <= y <= 2019:
            horizon[f"{y}-{m or 0:02d}"] += 1
        for a in b.get("authors") or []:
            if (a.get("role") or "") == "" and a.get("author_id") in authors:
                wauth[w][authors[a["author_id"]]] += 1
        for s in {tws, t}:
            k = norm_key(s)
            if k:
                keys[(k, "full")].add(w)
            km = norm_key(main_title(s))
            if km and km != k:
                keys[(km, "main")].add(w)
        if n_ed % 500_000 == 0:
            print(f"editions {n_ed} {time.time() - t0:.0f}s", flush=True)
    for w, info in works.items():
        if info["orig_title"]:
            k = norm_key(info["orig_title"])
            if k:
                keys[(k, "full")].add(w)
            km = norm_key(main_title(info["orig_title"]))
            if km and km != k:
                keys[(km, "main")].add(w)

    rows, ys = [], Counter()
    for w in set(ed_title) | set(best_title):
        info = works.get(w, {})
        oy = info.get("orig_year")
        if oy is not None and MIN_YEAR <= oy <= MAX_YEAR:
            year, src = oy, "original_publication_year"
        elif w in min_year:
            year, src = min_year[w], "min_edition_year" + ("_orig_out_of_range" if oy is not None else "")
        else:
            year, src = None, "no_valid_year" + ("_orig_out_of_range" if oy is not None else "")
        ys[src] += 1
        names = [n for n, _ in wauth[w].most_common()]
        titles = sorted(surfaces[w], key=lambda t: -surfaces[w][t])[:30]
        if info.get("orig_title") and info["orig_title"] not in titles:
            titles.append(info["orig_title"])
        rows.append({"work_id": w, "title": best_title.get(w) or ed_title[w][1], "orig_title": info.get("orig_title", ""),
                     "year": year, "year_source": src, "orig_year_raw": oy, "min_edition_year": min_year.get(w),
                     "authors": names[:5], "titles": titles, "ratings": info.get("work_ratings", ed_title.get(w, (0,))[0]),
                     "in_works_file": w in works})
    wdf = pd.DataFrame(rows)
    wdf.to_parquet(GD / "works.parquet", index=False)
    valid = set(wdf.loc[wdf.year.notna(), "work_id"])
    krows = [(k, kind, w) for (k, kind), ws_ in keys.items() for w in ws_ if w in valid]
    kdf = pd.DataFrame(krows, columns=["key", "kind", "work_id"])
    kdf.to_parquet(GD / "keys.parquet", index=False)
    stats = {"editions": n_ed, "works_with_editions": len(wdf), "works_with_valid_year": len(valid),
             "year_source": dict(ys), "keys": len(kdf), "unique_full_keys": int((kdf.kind == "full").sum()),
             "edition_pub_year_month_2016_2019": dict(sorted(horizon.items())),
             "valid_year_range": [MIN_YEAR, MAX_YEAR], "seconds": round(time.time() - t0)}
    (GD / "index_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "edition_pub_year_month_2016_2019"}), flush=True)


if __name__ == "__main__":
    main()
