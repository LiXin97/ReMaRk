"""Reddit-Books extraction-recall audit with an independent judge.

Population: every (submission, responder) pair of the Reddit-Books build (threads in the train/validation/test months
whose extraction succeeded), INCLUDING responders with no extracted book (false negatives live there). Sample 100 pairs
(seed 42). Judge: Gemini (gemini-2.5-pro, temperature 0; a different model family from the GPT-5.6 extractor) lists the
books or series the reply recommends to the requester. Matching (book level): normalized main titles equal, or the judged
title equals the extracted book's series, or one normalized title contains the other and an author surname agrees.
Metrics: recall = judged books matched by an extracted book / judged books (micro, Wilson 95% CI; also per-comment
macro); grounded recall = judged books matched by an extracted book that was grounded to Goodreads; precision on the same
sample = extracted books matched by a judged book / extracted books. The first 30 sampled pairs are written for a manual
check (manual labels in outputs/books/audit/e9_manual.json; agreement reported when present).
Usage: env/bin/python remark/scripts/n917_e9_books_recall.py [--model gemini-2.5-pro]
Output: outputs/books/audit/e9_recall.json, e9_sample.json, e9_gemini_cache.jsonl, e9_manual_view.md
"""

from __future__ import annotations

import argparse
import difflib
import re
import json
import math
import os
import random
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from books_common import OUTB, REPO, main_title, norm_key, split_of, surname_tokens  # noqa: E402

AUD = OUTB / "audit"
PROMPT = """Below is a book request posted on Reddit (r/suggestmeabook) and one user's reply to it.

List every book or book series that the reply recommends to the requester. Include a book even if it is only named
briefly as a suggestion. Exclude books that the requester mentioned as their own examples (unless the reply recommends
them again), books the reply advises against, and anything that is not a book (films, games, TV shows).

Request:
<<<
{request}
>>>

Reply:
<<<
{reply}
>>>

Return JSON only: {{"books": [{{"title": "...", "author": "... or null"}}]}}. Return {{"books": []}} if the reply
recommends no book."""


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def gemini(model: str, text: str, key: str) -> str:
    body = {"contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=180))
            return "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        except Exception as e:  # noqa: BLE001
            time.sleep(min(60, 4 * 2 ** attempt))
            last = repr(e)[:200].replace(key, "<key>")
    raise RuntimeError(last)


def key_of(t: str) -> str:
    return norm_key(main_title(t))


def match(j: dict, e: dict) -> bool:
    jt, et = key_of(j.get("title") or ""), key_of(e.get("title") or "")
    if not jt:
        return False
    if jt == et or (e.get("series") and jt == key_of(e["series"])):
        return True
    if et and (jt in et or et in jt) and min(len(jt), len(et)) >= 4:
        return bool(surname_tokens(j.get("author")) & surname_tokens(e.get("author"))) or not j.get("author")
    return False


SERIES_WORDS = re.compile(r"\b(series|trilogy|universe|saga|cycle|books|novels|chronicles)\b")
NONSPECIFIC = re.compile(r"^(books?|anything|other works?|works|stuff|novels?) (by|from|of)\b|\bgenre\b")


def key_adj(t: str) -> str:
    return " ".join(SERIES_WORDS.sub(" ", key_of(t)).split())


def match_adj(j: dict, e: dict) -> bool:
    """Documented normalization for the adjusted recall: series words dropped, fuzzy titles (ratio >= 0.85)."""
    if match(j, e):
        return True
    jt = key_adj(j.get("title") or "")
    cands = [key_adj(e.get("title") or ""), key_adj(e.get("series") or "")]
    return bool(jt) and any(c and (difflib.SequenceMatcher(None, jt, c).ratio() >= 0.85 or
                                   (len(c) >= 5 and (c in jt or jt in c))) for c in cands)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    key = os.environ["GEMINI_API_KEY"]
    AUD.mkdir(parents=True, exist_ok=True)
    samp_path = AUD / "e9_sample.json"
    if samp_path.exists():
        sample = json.load(open(samp_path))
    else:
        extracted = {}
        for line in open(OUTB / "ground/grounded_full.jsonl"):
            r = json.loads(line)
            extracted[r["sid"]] = r
        pop = []
        for p in sorted((OUTB / "threads").glob("*.jsonl")):
            for line in open(p):
                t = json.loads(line)
                if split_of(t["month"]) is None or t["sid"] not in extracted:
                    continue
                ex = extracted[t["sid"]]
                req = (t["title"] + "\n\n" + (t.get("selftext") or "")).strip()
                for resp in t["responders"]:
                    au = resp["author"]
                    if au not in ex["responders"]:
                        continue
                    pop.append({"sid": t["sid"], "month": t["month"], "author": au, "request": req,
                                "reply": "\n\n".join(c["body"] for c in resp["comments"]),
                                "extracted": [{k: b.get(k) for k in ("title", "author", "year", "series")} |
                                              {"grounded": (b.get("g") or {}).get("status") == "ok"}
                                              for b in ex["responders"][au]]})
        sample = random.Random(42).sample(pop, a.n)
        json.dump({"population": len(pop), "seed": 42, "items": sample}, open(samp_path, "w"), ensure_ascii=False, indent=1)
        sample = json.load(open(samp_path))
    items = sample["items"]
    cache_path = AUD / "e9_gemini_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            r = json.loads(line)
            cache[r["k"]] = r
    todo = [(k, it) for k, it in enumerate(items) if str(k) not in cache]

    def run(k_it):
        k, it = k_it
        try:
            txt = gemini(a.model, PROMPT.format(request=it["request"][:6000], reply=it["reply"][:8000]), key)
            rec = {"k": str(k), "text": txt}
        except Exception as e:  # noqa: BLE001
            rec = {"k": str(k), "error": str(e)}
        with open(cache_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(8) as pool:
        for rec in pool.map(run, todo):
            cache[rec["k"]] = rec
    judged_n = matched_n = matched_g = ext_n = ext_matched = adj_n = adj_m = adj_prec = 0
    per_comment, rows = [], []
    for k, it in enumerate(items):
        rec = cache.get(str(k), {})
        try:
            books = json.loads(rec.get("text", ""))["books"]
        except Exception:  # noqa: BLE001
            books = None
        if books is None:
            rows.append({"k": k, "error": rec.get("error", "unparsed")})
            continue
        ext = it["extracted"]
        m = [any(match(j, e) for e in ext) for j in books]
        mg = [any(match(j, e) and e["grounded"] for e in ext) for j in books]
        em = [any(match(j, e) for j in books) for e in ext]
        spec = [j for j in books if not NONSPECIFIC.search((j.get("title") or "").lower())]
        adj_n += len(spec); adj_m += sum(any(match_adj(j, e) for e in ext) for j in spec)
        adj_prec += sum(any(match_adj(j, e) for j in books) for e in ext)
        judged_n += len(books); matched_n += sum(m); matched_g += sum(mg)
        ext_n += len(ext); ext_matched += sum(em)
        if books:
            per_comment.append(sum(m) / len(books))
        rows.append({"k": k, "sid": it["sid"], "author": it["author"], "judged": books, "extracted": ext,
                     "judged_matched": m, "extracted_matched": em})
    res = {"judge": a.model, "population_pairs": sample["population"], "sampled": len(items),
           "judged_ok": sum("error" not in r for r in rows), "pairs_with_judged_books": len(per_comment),
           "judged_books": judged_n, "recall": matched_n / max(1, judged_n), "recall_ci95": wilson(matched_n, judged_n),
           "recall_macro": sum(per_comment) / max(1, len(per_comment)),
           "grounded_recall": matched_g / max(1, judged_n), "grounded_recall_ci95": wilson(matched_g, judged_n),
           "extracted_books": ext_n, "precision_vs_judge": ext_matched / max(1, ext_n),
           "precision_ci95": wilson(ext_matched, ext_n),
           "adjusted": {"judged_specific_books": adj_n, "recall": adj_m / max(1, adj_n), "recall_ci95": wilson(adj_m, adj_n),
                        "precision_vs_judge": adj_prec / max(1, ext_n), "precision_ci95": wilson(adj_prec, ext_n),
                        "rule": "non-specific judged items dropped; series words removed; fuzzy title ratio >= 0.85"},
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()}
    man = AUD / "e9_manual.json"
    if man.exists():
        labels = json.load(open(man))["labels"]
        tot = sum(len(v) for v in labels.values()); cov = sum(1 for v in labels.values() for x in v if x[1])
        res["manual"] = {"pairs": len(labels), "manual_items": tot, "extraction_covered": cov,
                         "extraction_recall_vs_manual": cov / max(1, tot), "extraction_recall_vs_manual_ci95": wilson(cov, tot)}
    (AUD / "e9_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (AUD / "e9_recall.json").write_text(json.dumps(res, indent=2))
    with open(AUD / "e9_manual_view.md", "w") as f:
        for k, it in enumerate(items[:30]):
            f.write(f"## k={k} sid={it['sid']} author={it['author']}\nREQUEST: {it['request'][:1500]}\n\nREPLY: {it['reply'][:3000]}\n\n"
                    f"EXTRACTED: {[e['title'] for e in it['extracted']]}\n\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
