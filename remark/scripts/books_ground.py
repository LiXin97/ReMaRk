"""Ground extracted book mentions to UCSD Goodreads works -> ("Title", first-publication year).

Matching stages (first stage with an acceptable candidate wins): A0 = normalized title equals a work's display or
original title; A1 = equals any edition title; B = subtitle-insensitive (the mention's title equals a Goodreads main
title, or the mention's main title equals a full Goodreads title). With an author, a candidate must share an author
surname (last name token, exact or edit distance <= 2 for >= 5 chars; on A0/A1 any name token of >= 5 chars) with one
of the work's primary authors (on A0/A1 also any shared name token of >= 3 chars); without an author, only A0/A1 and the work needs >= MIN_RATINGS_NOAUTH ratings. Among
candidates the most-rated work wins (ties: closest to the extractor's year). Returned: work id, year
(original_publication_year, else earliest edition year) and `surface` = the work's edition title the mention used
(its main title when the mention dropped the subtitle). books_build.py sets each work's output title to its most
frequent surface over all references and makes it a fixed point of Rank-GRPO's parser. Works whose
original_publication_year is outside [MIN_YEAR, MAX_YEAR] (written before AD 1000: Meditations, The Odyssey, ...)
still match but return status "year_out_of_range": "Title (Year)" needs a four-digit year.

Usage (GPU-node CPUs): python3 remark/scripts/books_ground.py --tag pilot2017
Input:  remark/outputs/books/extract/mentions_<tag>.jsonl
Output: remark/outputs/books/ground/grounded_<tag>.jsonl (+ ground_stats_<tag>.json)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import editdistance
import pandas as pd

from books_common import OUTB, main_title, name_tokens, norm_key, surname_tokens

GD = OUTB / "goodreads"
MIN_RATINGS_NOAUTH = 5
NAME_STOP = {"and", "the", "with", "van", "von", "der", "den", "del", "della", "des", "editor", "editors", "eds",
             "translated", "translator", "illustrated", "various"}


class Resolver:
    def __init__(self) -> None:
        w = pd.read_parquet(GD / "works.parquet",
                            columns=["work_id", "title", "orig_title", "year", "year_source", "authors", "ratings",
                                     "titles"])
        w = w[w.year.notna()]
        # works whose original_publication_year is outside [MIN_YEAR, MAX_YEAR] (e.g. Meditations, 180; The Odyssey,
        # -720) keep only a meaningless edition year: they still match (so a later translation's work cannot take
        # their place) but the mention is dropped as "year_out_of_range"
        self.bad_year = set(w.loc[w.year_source.str.contains("orig_out_of_range"), "work_id"])
        self.works = {r.work_id: (r.title, int(r.year), list(r.authors), int(r.ratings or 0))
                      for r in w.itertuples(index=False)}
        self.titles = {r.work_id: [r.title] + [t for t in r.titles if t != r.title] for r in w.itertuples(index=False)}
        self.atok = {wid: set().union(*[name_tokens(a) for a in v[2]]) if v[2] else set()
                     for wid, v in self.works.items()}
        self.asur = {wid: set().union(*[surname_tokens(a) for a in v[2]]) if v[2] else set()
                     for wid, v in self.works.items()}
        self.disp: dict[str, list[str]] = defaultdict(list)       # display / original title of the work
        self.disp_main: dict[str, list[str]] = defaultdict(list)  # same, subtitle dropped
        for r in w.itertuples(index=False):
            for t in {r.title, r.orig_title or ""}:
                if t:
                    k, km = norm_key(t), norm_key(main_title(t))
                    if k:
                        self.disp[k].append(r.work_id)
                    if km and km != k:
                        self.disp_main[km].append(r.work_id)
        k = pd.read_parquet(GD / "keys.parquet")
        self.full: dict[str, list[str]] = defaultdict(list)       # any edition title
        self.main: dict[str, list[str]] = defaultdict(list)
        for key, kind, wid in k.itertuples(index=False):
            if wid in self.works:
                (self.full if kind == "full" else self.main)[key].append(wid)

    def author_ok(self, sur: set[str], wid: str, exact_title: bool, mention_tokens: set[str] = frozenset()) -> bool:
        """Surname = last name token of a primary author (exact, or edit distance <= 2 if >= 5 chars); on an exact
        title match any shared name token of >= 3 chars also counts (name-order variants: "Cixin Liu" / "Liu Cixin")."""
        at = self.asur[wid]
        if sur & at or any(len(s) >= 5 and abs(len(s) - len(a)) <= 2 and editdistance.eval(s, a) <= 2
                           for s in sur for a in at):
            return True
        return exact_title and bool(mention_tokens & self.atok[wid])

    def surface(self, wid: str, k: str, km: str) -> str:
        """The work's edition title the mention used (its main title if the mention dropped the subtitle)."""
        ts = self.titles[wid]
        for f in (lambda t: t if norm_key(t) == k else None,
                  lambda t: main_title(t) if norm_key(main_title(t)) == k else None,
                  lambda t: t if norm_key(t) == km else None):
            for t in ts:
                r = f(t)
                if r:
                    return r
        return ts[0]

    def pick(self, cands: set[str], gyear) -> str:
        gy = gyear if isinstance(gyear, int) else None
        return min(cands, key=lambda w: (-self.works[w][3], abs(self.works[w][1] - gy) if gy else 0, w))

    def ground(self, title: str, author: str | None, gyear=None) -> dict:
        k, km = norm_key(title or ""), norm_key(main_title(title or ""))
        if not k:
            return {"status": "empty_title"}
        sur = surname_tokens(author) if author else set()
        mtok = {t for t in name_tokens(author or "") if len(t) >= 3} - NAME_STOP
        a0 = set(self.disp.get(k, []))
        a1 = set(self.full.get(k, [])) - a0
        b = set()
        for idx, key in ((self.disp_main, k), (self.main, k), (self.disp, km), (self.full, km)):
            if key:
                b |= set(idx.get(key, []))
        b -= a0 | a1
        had_title = bool(a0 or a1 or b)
        for name, cands in (("A0", a0), ("A1", a1), ("B", b)):
            if not cands:
                continue
            if sur:
                ca = {w for w in cands if self.author_ok(sur, w, name != "B", mtok)}
                how = f"{name}_author"
            elif name != "B":
                ca = {w for w in cands if self.works[w][3] >= MIN_RATINGS_NOAUTH}
                how = f"{name}_noauthor"
            else:
                ca = set()
            if not ca:
                continue
            wid = self.pick(ca, gyear)
            d, y, auths, r = self.works[wid]
            if wid in self.bad_year:
                return {"status": "year_out_of_range", "work_id": wid, "gr_title": d}
            sf = self.surface(wid, k, km)
            form = "display" if sf == d else "display_main" if sf == main_title(d) else "other_edition"
            return {"status": "ok", "how": how, "work_id": wid, "gr_title": d, "surface": sf, "form": form, "year": y,
                    "gr_authors": auths[:2], "ratings": r, "n_title_cands": len(cands)}
        return {"status": "author_mismatch" if had_title and sur else "no_title_match"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    res = Resolver()
    out = OUTB / "ground"
    out.mkdir(parents=True, exist_ok=True)
    st, unres_year, ex = Counter(), Counter(), defaultdict(list)
    with (OUTB / "extract" / f"mentions_{a.tag}.jsonl").open() as f, (out / f"grounded_{a.tag}.jsonl").open("w") as g:
        for line in f:
            row = json.loads(line)
            for b in row["request_books"]:
                b["g"] = res.ground(b["title"], b.get("author"), b.get("year"))
                st["request_" + b["g"]["status"]] += 1
            for au, books in row["responders"].items():
                for b in books:
                    b["g"] = res.ground(b["title"], b.get("author"), b.get("year"))
                    s = b["g"]["status"]
                    st["mention_" + s] += 1
                    st["mention_total"] += 1
                    if s == "ok":
                        st["how_" + b["g"]["how"]] += 1
                        st["form_" + b["g"]["form"]] += 1
                    else:
                        y = b.get("year")
                        unres_year["null" if not isinstance(y, int) else (">2017" if y > 2017 else "<=2017")] += 1
                        if len(ex[s]) < 40:
                            ex[s].append([b["title"], b.get("author"), b.get("year")])
            g.write(json.dumps(row, ensure_ascii=False) + "\n")
    tot = st["mention_total"]
    stats = {"tag": a.tag, "counts": dict(st), "mention_resolved_rate": st["mention_ok"] / max(1, tot),
             "unresolved_by_extractor_year": dict(unres_year),
             "unresolved_share_extractor_year_after_2017": unres_year[">2017"] / max(1, sum(unres_year.values())),
             "examples": dict(ex)}
    (out / f"ground_stats_{a.tag}.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in stats.items() if k != "examples"}, indent=1))


if __name__ == "__main__":
    main()
