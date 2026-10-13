"""Pre-registered go/no-go audit for Reddit-Books.

Steps (run in order; each is resumable and writes under remark/outputs/books/audit/):
  sample   (CPU)  200 random responder comments with >= 1 extracted book; 200 random grounded GT mentions
  judge    (CPU)  GPT-5.6 judge with a prompt different from the extractor's: per extracted book of the 200
                         comments -> correct / not_recommended / not_in_comment / wrong_book / not_a_book, plus
                         books the comment recommends that the extraction missed; and, per sampled mention, whether
                         the grounded Goodreads work is the book the comment means
  openlib  (GPU node)    Open Library search (title + author) -> first_publish_year for the 200 mentions
                         (openlibrary.org rejects the CPU host; the GPU node can reach it)
  wiki     (CPU)  Wikipedia infobox publication year for a 50-mention subsample
  report   (CPU)  go_nogo.json with Wilson 95% intervals; manual spot check read from manual_spotcheck.json

Criteria (pre-registered): >= 15k training threads with >= 1 grounded responder;
extraction precision >= 0.85 on the 200-comment audit; grounded-year agreement >= 0.9 on the 200-mention audit
(agreement = |grounded year - reference year| <= 2, the evaluation's tolerance; exact agreement also reported).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import urllib.parse
import urllib.request

from books_common import DATA, OUTB, REPO, log_usage, main_title, norm_key, parse_json_text, surname_tokens, split_of

AUD = OUTB / "audit"
UA = "research-books/0.1 (academic dataset audit)"


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def load_threads() -> dict:
    th = {}
    for p in sorted((OUTB / "threads").glob("*.jsonl")):
        if split_of(p.stem):
            for line in p.open():
                t = json.loads(line)
                th[t["sid"]] = t
    return th


def step_sample(tag: str) -> None:
    AUD.mkdir(parents=True, exist_ok=True)
    th = load_threads()
    comments, mentions = [], []
    for line in (OUTB / "ground" / f"grounded_{tag}.jsonl").open():
        r = json.loads(line)
        if r["sid"] not in th:
            continue
        t = th[r["sid"]]
        by_author = {x["author"]: x for x in t["responders"]}
        for au, books in r["responders"].items():
            if not books or au not in by_author:
                continue
            text = "\n\n".join(c["body"] for c in by_author[au]["comments"])
            comments.append({"sid": r["sid"], "author": au, "text": text, "books": books})
            for i, b in enumerate(books):
                if b["g"]["status"] == "ok":
                    mentions.append({"sid": r["sid"], "author": au, "i": i, "text": text, "mention": b})
    rng = random.Random(42)
    sc = rng.sample(comments, 200)
    sm = rng.sample(mentions, 200)
    for x in sc + sm:
        t = th[x["sid"]]
        x["request"] = {"title": t["title"], "body": t["selftext"][:3000]}
    json.dump({"population_comments_with_extraction": len(comments), "population_grounded_mentions": len(mentions),
               "comments": sc, "mentions": sm}, open(AUD / "samples.json", "w"), ensure_ascii=False, indent=1)
    print(f"sampled 200/{len(comments)} comments, 200/{len(mentions)} mentions")


JUDGE_EXTRACT = """You are auditing an automatic extractor of book recommendations from Reddit (r/suggestmeabook). \
Below are a book request, ONE reply to it, and the list of books the extractor says this reply recommends.

For each listed book decide:
- "correct": the reply recommends (suggests reading) this book to the requester, and the title/author identify the \
book the reply refers to (a series recommendation mapped to the series' first book counts as correct);
- "not_recommended": the reply mentions the book but does not recommend it (e.g. only as a comparison, the \
requester's own example, or a book it advises against);
- "not_in_reply": the reply does not refer to this book at all;
- "wrong_book": the reply recommends a book, but the given title/author identify a different book;
- "not_a_book": the reply refers to a film, show, game, website, author only, etc.
Then list books the reply recommends that are missing from the list.

Return JSON only: {"verdicts": [{"n": 1, "verdict": "...", "why": "<10 words"}, ...], "missed": [{"title": "..."}]}

REQUEST TITLE: {title}
REQUEST BODY: {body}

REPLY:
{reply}

EXTRACTED BOOKS:
{books}"""

JUDGE_GROUND = """A Reddit reply recommends a book; an automatic system matched the mention to a catalog record. \
Decide whether the catalog record is the same book (same work; a different edition or translation of the same \
work counts as the same) as the one the reply refers to.

REPLY: {reply}
MENTION (as extracted): {title} by {author}
CATALOG RECORD: "{gtitle}" by {gauthors}, first published {gyear}

Return JSON only: {"same_book": true or false, "why": "<12 words"}"""


def step_judge() -> None:
    sys.path.insert(0, str(REPO / "remark"))
    from lib.llm import batch_chat
    s = json.load(open(AUD / "samples.json"))
    reqs = []
    for k, c in enumerate(s["comments"]):
        books = "\n".join(f"{i}. {b['title']}" + (f" by {b['author']}" if b.get("author") else "")
                          for i, b in enumerate(c["books"], 1))
        msg = (JUDGE_EXTRACT.replace("{title}", c["request"]["title"]).replace("{body}", c["request"]["body"] or "(empty)")
               .replace("{reply}", c["text"][:4000]).replace("{books}", books))
        reqs.append({"id": f"je:{k}", "messages": [{"role": "user", "content": msg}], "max_tokens": 4000})
    for k, m in enumerate(s["mentions"]):
        b, g = m["mention"], m["mention"]["g"]
        msg = (JUDGE_GROUND.replace("{reply}", m["text"][:3000]).replace("{title}", b["title"])
               .replace("{author}", b.get("author") or "(no author given)").replace("{gtitle}", g["gr_title"])
               .replace("{gauthors}", ", ".join(g["gr_authors"]) or "unknown").replace("{gyear}", str(g["year"])))
        reqs.append({"id": f"jg:{k}", "messages": [{"role": "user", "content": msg}], "max_tokens": 2000})
    res = batch_chat(reqs, AUD / "judge_cache.jsonl", workers=16)
    out = {"extraction": [], "grounding": []}
    for k, c in enumerate(s["comments"]):
        j = parse_json_text(res.get(f"je:{k}", {}).get("text", ""))
        out["extraction"].append({"k": k, "n_books": len(c["books"]), "judge": j})
    for k in range(len(s["mentions"])):
        out["grounding"].append({"k": k, "judge": parse_json_text(res.get(f"jg:{k}", {}).get("text", ""))})
    json.dump(out, open(AUD / "judge.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(log_usage("audit_judge", AUD / "judge_cache.jsonl")))


def http_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as exc:  # noqa: BLE001
            err = exc
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(repr(err))


def title_match(a: str, b: str) -> bool:
    ka, kb = norm_key(a), norm_key(b)
    return ka == kb or norm_key(main_title(a)) == norm_key(main_title(b)) or ka == norm_key(main_title(b)) \
        or kb == norm_key(main_title(a))


def step_openlib() -> None:
    s = json.load(open(AUD / "samples.json"))
    path = AUD / "openlibrary.json"
    done = json.load(open(path)) if path.exists() else {}
    for k, m in enumerate(s["mentions"]):
        if str(k) in done:
            continue
        b, g = m["mention"], m["mention"]["g"]
        sur = surname_tokens(b.get("author") or ", ".join(g["gr_authors"][:1]))
        rec = {"query_title": b["title"], "query_author": b.get("author"), "year": None, "match": None}
        for title in dict.fromkeys([b["title"], g["surface"], g["gr_title"]]):
            q = {"title": title, "fields": "key,title,author_name,first_publish_year", "limit": 10}
            if sur:
                q["author"] = " ".join(sur)
            try:
                docs = http_json("https://openlibrary.org/search.json?" + urllib.parse.urlencode(q)).get("docs", [])
            except RuntimeError as exc:
                rec["error"] = str(exc)[:200]
                docs = []
            cands = [d for d in docs if d.get("first_publish_year") and title_match(d.get("title", ""), title)
                     and (not sur or sur & set().union(*[surname_tokens(a) for a in d.get("author_name", [])] or [set()]))]
            if cands:
                rec["year"] = min(d["first_publish_year"] for d in cands)
                rec["match"] = [[d.get("title"), (d.get("author_name") or [None])[0], d["first_publish_year"]]
                                for d in cands[:3]]
                break
            time.sleep(0.5)
        done[str(k)] = rec
        json.dump(done, open(path, "w"), ensure_ascii=False, indent=1)
        time.sleep(0.5)
    print(f"openlibrary: {sum(v['year'] is not None for v in done.values())}/{len(done)} with a year")


def step_wiki(n: int = 50) -> None:
    s = json.load(open(AUD / "samples.json"))
    path = AUD / "wikipedia.json"
    done = json.load(open(path)) if path.exists() else {}
    api = "https://en.wikipedia.org/w/api.php?"
    for k, m in enumerate(s["mentions"][:n]):
        if str(k) in done:
            continue
        b, g = m["mention"], m["mention"]["g"]
        author = b.get("author") or ", ".join(g["gr_authors"][:1])
        rec = {"year": None, "page": None}
        try:
            hits = http_json(api + urllib.parse.urlencode({"action": "query", "list": "search", "format": "json",
                                                          "srsearch": f"{b['title']} {author} book", "srlimit": 3}))
            for h in hits.get("query", {}).get("search", []):
                page = http_json(api + urllib.parse.urlencode({"action": "parse", "page": h["title"], "prop": "wikitext",
                                                              "format": "json", "section": 0}))
                wt = page.get("parse", {}).get("wikitext", {}).get("*", "")
                if "{{Infobox book" not in wt and "{{infobox book" not in wt.lower():
                    continue
                if not title_match(re.sub(r"\s*\(.*\)$", "", h["title"]), b["title"]) and \
                        not title_match(re.sub(r"\s*\(.*\)$", "", h["title"]), g["gr_title"]):
                    continue
                f = re.search(r"\|\s*(?:pub_date|published|release_date|first_published)\s*=\s*([^\n]*)", wt, re.I)
                y = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", f.group(1)) if f else None
                if y:
                    rec = {"year": int(y.group(1)), "page": h["title"], "field": f.group(1)[:80]}
                    break
        except RuntimeError as exc:
            rec["error"] = str(exc)[:200]
        done[str(k)] = rec
        json.dump(done, open(path, "w"), ensure_ascii=False, indent=1)
        time.sleep(0.3)
    print(f"wikipedia: {sum(v['year'] is not None for v in done.values())}/{len(done)} with a year")


def current_status(tag: str, mentions: list[dict]) -> list[str]:
    """Grounding status of each sampled mention in the current grounded_<tag>.jsonl (the resolver may have changed
    after sampling, e.g. pre-1000 works now return year_out_of_range; such mentions leave the year audit)."""
    need = {m["sid"] for m in mentions}
    cur = {}
    for line in (OUTB / "ground" / f"grounded_{tag}.jsonl").open():
        r = json.loads(line)
        if r["sid"] in need:
            cur[r["sid"]] = r
    return [cur[m["sid"]]["responders"][m["author"]][m["i"]]["g"]["status"] for m in mentions]


def step_report(tag: str = "full") -> None:
    s = json.load(open(AUD / "samples.json"))
    j = json.load(open(AUD / "judge.json"))
    status = current_status(tag, s["mentions"])
    keep = [st == "ok" for st in status]
    # extraction precision
    verdicts, missed, comments_all_ok, n_ok_json = [], 0, 0, 0
    for e, c in zip(j["extraction"], s["comments"]):
        jj = e["judge"] if isinstance(e["judge"], dict) else {}
        vs = [v.get("verdict") for v in jj.get("verdicts", []) if isinstance(v, dict)]
        if len(vs) != len(c["books"]):
            vs = (vs + ["unjudged"] * len(c["books"]))[:len(c["books"])]
        else:
            n_ok_json += 1
        verdicts += vs
        missed += len(jj.get("missed") or [])
        comments_all_ok += all(v == "correct" for v in vs)
    from collections import Counter
    vc = Counter(verdicts)
    n_items, n_cor = len(verdicts), vc["correct"]
    # grounding correctness (judge)
    same = [bool((g["judge"] or {}).get("same_book")) for g, kp in zip(j["grounding"], keep) if kp]
    # year agreement
    ol = json.load(open(AUD / "openlibrary.json")) if (AUD / "openlibrary.json").exists() else {}
    wk = json.load(open(AUD / "wikipedia.json")) if (AUD / "wikipedia.json").exists() else {}

    def agree(ref: dict, subset=None):
        rows = []
        for k, m in enumerate(s["mentions"]):
            r = ref.get(str(k))
            if r and r.get("year") and keep[k] and (subset is None or subset[k]):
                rows.append(m["mention"]["g"]["year"] - r["year"])
        n = len(rows)
        k2, k0 = sum(abs(d) <= 2 for d in rows), sum(d == 0 for d in rows)
        return {"n_with_reference_year": n, "within_2": k2, "rate_within_2": round(k2 / max(1, n), 4),
                "ci95_within_2": wilson(k2, n), "exact_rate": round(k0 / max(1, n), 4)}

    gpt = {str(k): {"year": m["mention"].get("year")} for k, m in enumerate(s["mentions"])
           if isinstance(m["mention"].get("year"), int)}
    same_all = [bool((g["judge"] or {}).get("same_book")) for g in j["grounding"]]
    year = {"open_library": agree(ol), "open_library_on_judge_same_book": agree(ol, same_all),
            "wikipedia_subsample50": agree(wk), "gpt56_extractor_year": agree(gpt)}
    disagreements = [{"title": m["mention"]["title"], "grounded": [m["mention"]["g"]["surface"], m["mention"]["g"]["year"]],
                      "open_library": ol.get(str(k), {}).get("year"), "wiki": wk.get(str(k), {}).get("year"),
                      "gpt": m["mention"].get("year")}
                     for k, m in enumerate(s["mentions"])
                     if keep[k] and ol.get(str(k), {}).get("year")
                     and abs(m["mention"]["g"]["year"] - ol[str(k)]["year"]) > 2]
    man = json.load(open(AUD / "manual_spotcheck.json")) if (AUD / "manual_spotcheck.json").exists() else None
    stats = json.load(open(DATA / "stats.json")) if (DATA / "stats.json").exists() else {}
    n_train = (stats.get("threads_with_grounded_responder") or {}).get("train")
    prec = n_cor / max(1, n_items)
    yr = year["open_library"]["rate_within_2"]
    go = {
        "criteria": {
            "train_threads_with_grounded_responder>=15000": {"value": n_train, "pass": bool(n_train and n_train >= 15000)},
            "extraction_precision>=0.85": {"value": round(prec, 4), "ci95": wilson(n_cor, n_items),
                                            "n_items": n_items, "n_comments": len(s["comments"]), "pass": prec >= 0.85},
            "grounded_year_agreement>=0.9": {"value": yr, "reference": "Open Library first_publish_year, |dy|<=2",
                                             "detail": year["open_library"], "pass": yr >= 0.9},
        },
        "extraction_verdicts": dict(vc), "comments_all_correct_rate": round(comments_all_ok / len(s["comments"]), 4),
        "judge_items_json_ok_comments": n_ok_json,
        "missed_books_per_comment": round(missed / len(s["comments"]), 3),
        "recall_proxy": round(n_cor / max(1, n_cor + missed), 4),
        "grounding_same_book_rate": {"value": round(sum(same) / len(same), 4), "ci95": wilson(sum(same), len(same)),
                                     "n": len(same)},
        "year_agreement": year, "year_disagreements_open_library": disagreements,
        "sampled_mentions_no_longer_grounded": {str(k): st for k, st in enumerate(status) if st != "ok"},
        "manual_spotcheck": man,
        "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    go["decision"] = "GO" if all(c["pass"] for c in go["criteria"].values()) else "NO-GO"
    (OUTB / "go_nogo.json").write_text(json.dumps(go, indent=1, ensure_ascii=False))
    print(json.dumps({k: go[k] for k in ("criteria", "decision", "grounding_same_book_rate", "extraction_verdicts")},
                     indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["sample", "judge", "openlib", "wiki", "report"])
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    {"sample": lambda: step_sample(a.tag), "judge": step_judge, "openlib": step_openlib, "wiki": step_wiki,
     "report": lambda: step_report(a.tag)}[a.step]()


if __name__ == "__main__":
    main()
