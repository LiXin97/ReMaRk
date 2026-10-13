"""Build Reddit-Books in the Reddit-v2 (Rank-GRPO) on-disk format.

Thread = r/suggestmeabook submission; responder = distinct top-level commenter (books_threads.py); mentions from
GPT-5.6 (books_extract.py) grounded to UCSD Goodreads works (books_ground.py). Item = (title, first-publication year)
where title is the work's most frequent surface form over all references (GT mentions, requester books, teacher
lines), made a fixed point of Rank-GRPO's parser (canon).
  seen_titles  = requester books marked read (grounded title, else canon of the extracted title)
  GT(responder) = the responder's grounded recommendations in order, de-duplicated, minus seen titles
A responder with empty GT gives no row; a thread needs >= 1 such responder and a prompt of <= max_prompt_tokens
(Qwen3.5 chat template). Splits by submission month (configs/crs_reddit_books.yaml).

Outputs under remark/data/crs/reddit_books/:
  sft_dataset/{train,validation,test}  features identical to Reddit-v2 sft_dataset; one row per (thread, responder).
                                       train: completion = the thread's grounded GPT-5.6 teacher list (shared by its
                                       rows; only threads whose list keeps >= min_sft_items); val/test: GT lines
  grpo/train                           per-responder rows of all train threads; features = Reddit-v2 grpo_dataset
  grpo_consensus/train                 one row per train thread: union GT, weight = fraction of responders naming the
                                       item, n_responders; features = remark/data/crs/grpo_consensus
  catalog.pkl                          sorted list[(title, int year)]: GT items of all splits + teacher items used in
                                       completions + top-N most-rated Goodreads works
  meta/*.jsonl                         ids per row, same order (submission id, responder comment ids, timestamps);
                                       meta/catalog_works.jsonl maps each catalog item to its Goodreads work ids
  stats.json, manifest.json

Usage (GPU-node CPUs): /tmp/qenv/bin/python remark/scripts/books_build.py --tag full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from collections import Counter, OrderedDict, defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk

from books_common import CFG, DATA, HEADER, OUTB, REDDIT_HEADER, REPO, SEP, canon, parse_lines, prompt_text, rel, \
    split_of, CFG_PATH
from books_ground import Resolver

sys.path.append(str(REPO / "remark/scripts"))  # crs_eval (evaluation hit logic) lives in the project repo
from crs_eval import hits_for, recall_ndcg  # noqa: E402

P = {k: REPO / v for k, v in CFG["paths"].items()}
SPLITS = ("train", "validation", "test")


def dist(x) -> dict:
    x = np.asarray(list(x), dtype=float)
    if not len(x):
        return {}
    q = np.quantile(x, [0.5, 0.9, 0.99])
    return {"n": int(len(x)), "mean": round(float(x.mean()), 2), "p50": float(q[0]), "p90": float(q[1]),
            "p99": float(q[2]), "max": float(x.max())}


def hist(x, cap: int) -> dict:
    c = Counter(min(int(v), cap) for v in x)
    return {(f"{k}+" if k == cap else str(k)): c[k] for k in sorted(c)}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    ap.add_argument("--out", default=str(DATA))
    ap.add_argument("--teacher", default="", help="teacher lines file (default outputs/books/sft/teacher_<tag>.jsonl)")
    a = ap.parse_args()
    out = __import__("pathlib").Path(a.out)
    t0 = time.time()
    from books_sft_targets import parse_teacher  # noqa: F401  (same line format; teacher file holds parsed lines)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REPO / CFG["tokenizer"]))

    # ---------------------------------------------------------------- inputs
    ref_sft = load_from_disk(str(P["reddit_sft"] / "test"))
    heads = {p[0]["content"].partition(SEP)[0] for p in ref_sft.select(range(2000))["prompt"]}
    assert heads == {REDDIT_HEADER}, heads
    threads: dict[str, dict] = {}
    for sp, (m0, m1) in CFG["splits"].items():
        for path in sorted((OUTB / "threads").glob("*.jsonl")):
            if m0 <= path.stem <= m1:
                for line in path.open():
                    t = json.loads(line)
                    threads[t["sid"]] = t
    grounded = {}
    for line in (OUTB / "ground" / f"grounded_{a.tag}.jsonl").open():
        r = json.loads(line)
        if r["sid"] in threads:
            grounded[r["sid"]] = r
    teacher = {}
    tpath = __import__("pathlib").Path(a.teacher) if a.teacher else OUTB / "sft" / f"teacher_{a.tag}.jsonl"
    if tpath.exists():
        for line in tpath.open():
            r = json.loads(line)
            teacher[r["sid"]] = r["lines"]
    res = Resolver()
    print(f"inputs: {len(threads)} threads, {len(grounded)} extracted, {len(teacher)} teacher lists "
          f"{time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- ground teacher lines; vote titles
    votes: dict[str, Counter] = defaultdict(Counter)
    tgr: dict[str, list[dict]] = {}
    st_t = Counter()
    for sid, lines in teacher.items():
        g = []
        for title, year, author in lines[:20]:
            x = res.ground(title, author, year)
            st_t["lines"] += 1
            st_t["line_" + x["status"]] += 1
            if year > 2017:
                st_t["line_teacher_year_after_2017"] += 1
                st_t["line_teacher_year_after_2017_" + x["status"]] += 1
            if x["status"] == "ok":
                st_t["line_year_within2_of_teacher_year"] += abs(x["year"] - year) <= 2
                votes[x["work_id"]][x["surface"]] += 1
            g.append(x)
        tgr[sid] = g
    for r in grounded.values():
        for b in r["request_books"]:
            if b["g"]["status"] == "ok":
                votes[b["g"]["work_id"]][b["g"]["surface"]] += 1
        for bs in r["responders"].values():
            for b in bs:
                if b["g"]["status"] == "ok":
                    votes[b["g"]["work_id"]][b["g"]["surface"]] += 1
    item: dict[str, tuple[str, int] | None] = {}
    st_title = Counter()

    def item_of(wid: str) -> tuple[str, int] | None:
        if wid not in item:
            cands = [s for s, _ in votes[wid].most_common()] + res.titles.get(wid, [])[:1]
            t = next((c for c in map(canon, cands) if c), None)
            item[wid] = (t, res.works[wid][1]) if t else None
            st_title["works_titled" if t else "works_canon_failed"] += 1
            if t and cands and t != canon(res.works[wid][0]):
                st_title["title_differs_from_goodreads_display"] += 1
        return item[wid]

    # ---------------------------------------------------------------- per-thread rows
    rows: dict[str, list[dict]] = {s: [] for s in SPLITS}
    st = {s: Counter() for s in SPLITS}
    thread_rows: dict[str, list[dict]] = {}
    for sid, th in sorted(threads.items(), key=lambda kv: (kv[1]["created_utc"], kv[0])):
        sp = split_of(th["month"])
        c = st[sp]
        c["threads"] += 1
        if not th["responders"]:
            c["threads_no_responder"] += 1
            continue
        r = grounded.get(sid)
        if r is None:
            c["threads_not_extracted"] += 1
            continue
        c["threads_with_responder"] += 1
        c["threads_extraction_failed"] += not r.get("ok", True)
        seen: list[str] = []
        for b in r["request_books"]:
            if not b.get("read"):
                continue
            it = item_of(b["g"]["work_id"]) if b["g"]["status"] == "ok" else None
            t = it[0] if it else canon(b["title"])
            if t and t not in seen:
                seen.append(t)
        seen_set = set(seen)
        resp_rows = []
        for resp in th["responders"]:
            books = r["responders"].get(resp["author"], [])
            c["responders"] += 1
            c["responders_with_mention"] += bool(books)
            gt, pairs = [], set()
            for b in books:
                c["mentions"] += 1
                if b["g"]["status"] != "ok":
                    c["mentions_unresolved"] += 1
                    continue
                it = item_of(b["g"]["work_id"])
                if it is None:
                    c["mentions_canon_failed"] += 1
                    continue
                if it[0] in seen_set:
                    c["mentions_dropped_seen"] += 1
                    continue
                if it in pairs:
                    c["mentions_duplicate"] += 1
                    continue
                pairs.add(it)
                gt.append(it)
            if gt:
                resp_rows.append({"author": resp["author"], "gt": gt, "comment_ids": [x["id"] for x in resp["comments"]],
                                  "first_comment_utc": resp["comments"][0]["created_utc"]})
        if not resp_rows:
            c["threads_no_grounded_responder"] += 1
            continue
        prompt = prompt_text(th["context"])
        thread_rows[sid] = resp_rows
        rows[sp].append({"sid": sid, "th": th, "prompt": prompt, "seen": seen, "resp": resp_rows})

    # ---------------------------------------------------------------- prompt length filter (Qwen chat template)
    lim = CFG["max_prompt_tokens"]
    ntok = {}
    for sp in SPLITS:
        texts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False,
                                         add_generation_prompt=True) for x in rows[sp]]
        lens = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]] if texts else []
        for x, n in zip(rows[sp], lens):
            x["ntok"] = n
        ntok[sp] = lens
        st[sp]["threads_dropped_prompt_over_limit"] = sum(n > lim for n in lens)
        rows[sp] = [x for x in rows[sp] if x["ntok"] <= lim]

    # ---------------------------------------------------------------- SFT teacher completions (train)
    min_items = CFG["min_sft_items"]
    teach_items: dict[str, list[tuple[str, int]]] = {}
    st_sft = Counter()
    for x in rows["train"]:
        g = tgr.get(x["sid"])
        st_sft["train_threads"] += 1
        if g is None:
            st_sft["no_teacher_output"] += 1
            continue
        seen_set, lst, got = set(x["seen"]), [], set()
        st_sft["teacher_lines_parsed"] += len(g)
        for y in g:
            if y["status"] != "ok":
                continue
            st_sft["teacher_lines_grounded"] += 1
            it = item_of(y["work_id"])
            if it is None:
                continue
            if it[0] in seen_set:
                st_sft["teacher_lines_dropped_seen"] += 1
                continue
            if it[0] in got:
                st_sft["teacher_lines_dropped_duplicate"] += 1
                continue
            got.add(it[0])
            lst.append(it)
        st_sft["teacher_items_kept_total"] += len(lst)
        if len(lst) >= min_items:
            teach_items[x["sid"]] = lst
            st_sft["threads_with_valid_list"] += 1
        else:
            st_sft["threads_list_too_short"] += 1

    # ---------------------------------------------------------------- catalog
    gt_items = {it for sp in SPLITS for x in rows[sp] for r in x["resp"] for it in r["gt"]}
    t_items = {it for lst in teach_items.values() for it in lst}
    wdf = pd.DataFrame([(w, v[3]) for w, v in res.works.items()], columns=["w", "r"]).sort_values(["r", "w"],
                                                                                                    ascending=[False, True])
    pop = []
    for w in wdf.w.values:
        if w in res.bad_year:  # pre-1000 works have no usable year (see books_ground.py)
            continue
        it = item_of(w)
        if it:
            pop.append(it)
        if len(pop) >= CFG["top_n_popular_works"]:
            break
    catalog = sorted(gt_items | t_items | set(pop))
    cat_set = set(catalog)
    title_years = defaultdict(set)
    for t, y in catalog:
        title_years[t].add(y)
    collide = sum(1 for ys in title_years.values() if any(abs(p - q) <= 2 for p in ys for q in ys if p < q))

    # ---------------------------------------------------------------- write datasets
    sft_feats = load_from_disk(str(P["reddit_sft"] / "train")).features
    grpo_feats = load_from_disk(str(P["reddit_grpo"])).features
    cons_feats = load_from_disk(str(P["reddit_grpo_consensus"])).features
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    lines_of = lambda items: "\n".join(f"{t} ({y})" for t, y in items)  # noqa: E731

    def meta_of(x, r=None) -> dict:
        m = {"submission_id": x["sid"], "created_utc": x["th"]["created_utc"], "month": x["th"]["month"],
             "n_responders": len(x["resp"]), "prompt_tokens": x["ntok"]}
        if r is not None:
            m.update({"responder_hash": hashlib.sha1(r["author"].encode()).hexdigest()[:12],
                      "comment_ids": r["comment_ids"], "first_comment_utc": r["first_comment_utc"]})
        return m

    written = {}
    for sp in SPLITS:
        recs, metas = [], []
        for x in rows[sp]:
            if sp == "train" and x["sid"] not in teach_items:
                continue
            for r in x["resp"]:
                comp = lines_of(teach_items[x["sid"]]) if sp == "train" else lines_of(r["gt"])
                recs.append({"prompt": [{"content": x["prompt"], "role": "user"}],
                             "completion": [{"content": comp, "role": "assistant"}],
                             "seen_titles": list(x["seen"]),
                             "groundtruth_with_release_year": [[t, str(y)] for t, y in r["gt"]]})
                metas.append(meta_of(x, r))
        cols = {c: [r[c] for r in recs] for c in sft_feats}
        Dataset.from_dict(cols, features=sft_feats).save_to_disk(str(out / "sft_dataset" / sp))
        with open(out / "meta" / f"sft_{sp}.jsonl", "w") as f:
            for m in metas:
                f.write(json.dumps(m) + "\n")
        written[f"sft_dataset/{sp}"] = len(recs)
    g, gm = {c: [] for c in grpo_feats}, []
    cons, cm = {c: [] for c in cons_feats}, []
    for x in rows["train"]:
        cnt: "OrderedDict[tuple, int]" = OrderedDict()
        for r in x["resp"]:
            g["prompt"].append([{"content": x["prompt"], "role": "user"}])
            g["seen_titles"].append(list(x["seen"]))
            g["groundtruth_with_release_year"].append([[t, str(y)] for t, y in r["gt"]])
            gm.append(meta_of(x, r))
            for it in r["gt"]:
                cnt[it] = cnt.get(it, 0) + 1
        n = len(x["resp"])
        cons["prompt"].append([{"content": x["prompt"], "role": "user"}])
        cons["seen_titles"].append(list(x["seen"]))
        cons["groundtruth_with_release_year"].append([[t, str(y)] for t, y in cnt])
        cons["groundtruth_weights"].append([v / n for v in cnt.values()])
        cons["n_responders"].append(n)
        cm.append(meta_of(x))
    Dataset.from_dict(g, features=grpo_feats).save_to_disk(str(out / "grpo" / "train"))
    Dataset.from_dict(cons, features=cons_feats).save_to_disk(str(out / "grpo_consensus" / "train"))
    for name, ms in (("grpo_train", gm), ("grpo_consensus_train", cm)):
        with open(out / "meta" / f"{name}.jsonl", "w") as f:
            for m in ms:
                f.write(json.dumps(m) + "\n")
    written["grpo/train"], written["grpo_consensus/train"] = len(gm), len(cm)
    ref_cat = pickle.load(open(P["reddit_catalog"], "rb"))
    assert type(ref_cat) is list and type(ref_cat[0]) is tuple and type(ref_cat[0][1]) is int
    with open(out / "catalog.pkl", "wb") as f:
        pickle.dump([(str(t), int(y)) for t, y in catalog], f)
    written["catalog.pkl"] = len(catalog)
    # traceability: catalog item -> Goodreads work(s) (several works can share one (title, year) item)
    by_item = defaultdict(list)
    for w, it in item.items():
        if it and it in cat_set:
            by_item[it].append(w)
    with open(out / "meta" / "catalog_works.jsonl", "w") as f:
        for it in catalog:
            ws_ = sorted(by_item[it], key=lambda w: -res.works[w][3])
            f.write(json.dumps({"title": it[0], "year": it[1], "goodreads_work_ids": ws_,
                                "goodreads_display_title": res.works[ws_[0]][0] if ws_ else None,
                                "authors": res.works[ws_[0]][2][:2] if ws_ else [],
                                "in_gt": it in gt_items, "in_sft_completions": it in t_items,
                                "surface_votes": dict(votes[ws_[0]].most_common(3)) if ws_ else {}},
                               ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- self-checks (reload from disk)
    checks, fails = {}, []
    for sp in SPLITS:
        if not written[f"sft_dataset/{sp}"]:  # empty split (pilot builds): save_to_disk writes 0 shards
            checks[sp] = {"rows": 0}
            continue
        d = load_from_disk(str(out / "sft_dataset" / sp))
        assert d.features == sft_feats, (sp, d.features)
        ok = orc = n_le20 = in_cat = comp_cat = 0
        sids = [json.loads(line)["submission_id"] for line in open(out / "meta" / f"sft_{sp}.jsonl")]
        assert len(sids) == len(d)
        for it, sid in zip(d, sids):
            gts = [(t, int(y)) for t, y in it["groundtruth_with_release_year"]]
            text = it["completion"][0]["content"]
            recs = parse_lines(text)
            seen = set(it["seen_titles"])
            good = len(recs) == text.count("\n") + 1  # every line parses
            if sp == "train":  # the completion parses back to exactly the intended grounded teacher list
                good &= (recs == teach_items[sid] and all(r in cat_set for r in recs)
                         and not any(r[0] in seen for r in recs) and len(recs) >= min_items)
                comp_cat += all(r in cat_set for r in recs)
            else:
                good &= recs == gts
            good &= all(str(int(y)) == y for _, y in it["groundtruth_with_release_year"])
            ok += good
            if not good and len(fails) < 10:
                fails.append([sp, it["completion"][0]["content"][:300], it["groundtruth_with_release_year"]])
            in_cat += all(x in cat_set for x in gts)
            if len(gts) <= 20:  # a perfect answer: the GT list itself
                hits, _ = hits_for(gts[:20], gts, seen, cat_set, "truncate")
                orc += recall_ndcg(hits, len(gts), 20)[0]
                n_le20 += 1
        checks[sp] = {"rows": len(d), "parse_back_pass_rate": ok / max(1, len(d)),
                      "rows_all_gt_in_catalog": in_cat / max(1, len(d)),
                      "oracle_recall@20_truncate": orc / max(1, n_le20), "rows_gt_le_20": n_le20}
        if sp == "train":
            checks[sp]["completion_all_in_catalog_rate"] = comp_cat / max(1, len(d))
    gd = load_from_disk(str(out / "grpo" / "train"))
    cd = load_from_disk(str(out / "grpo_consensus" / "train"))
    checks["grpo_features_match_reddit_v2"] = gd.features == grpo_feats
    checks["grpo_consensus_features_match"] = cd.features == cons_feats
    checks["sft_features_match_reddit_v2"] = all(
        load_from_disk(str(out / "sft_dataset" / s)).features == sft_feats for s in SPLITS if written[f"sft_dataset/{s}"])
    checks["consensus_weights_in_(0,1]"] = all(0 < w <= 1 for ws in cd["groundtruth_weights"] for w in ws)
    checks["consensus_rows_equal_grpo_threads"] = len(cd) == len({m["submission_id"] for m in gm})
    cat_back = pickle.load(open(out / "catalog.pkl", "rb"))
    checks["catalog_type"] = f"{type(cat_back).__name__}[{type(cat_back[0]).__name__}(str, {type(cat_back[0][1]).__name__})]"
    checks["catalog_same_type_as_gt_catalog"] = (type(cat_back) is type(ref_cat) and type(cat_back[0]) is tuple
                                                 and type(cat_back[0][1]) is int and type(cat_back[0][0]) is str)

    # ---------------------------------------------------------------- stats
    def resp_counts(sp):
        return [len(x["resp"]) for x in rows[sp]]

    sft_train_lines = [len(v) for s, v in teach_items.items()]
    stats = {
        "dataset": "reddit_books", "config": CFG, "splits_months": CFG["splits"],
        "header": HEADER,
        "threads_with_grounded_responder": {sp: len(rows[sp]) for sp in SPLITS},
        "rows": written,
        "responders_per_thread": {sp: dist(resp_counts(sp)) for sp in SPLITS},
        "responder_count_hist": {sp: hist(resp_counts(sp), 10) for sp in SPLITS},
        "frac_threads_n1": {sp: float(np.mean([n == 1 for n in resp_counts(sp)])) if rows[sp] else 0 for sp in SPLITS},
        "gt_items_per_responder": {sp: dist(len(r["gt"]) for x in rows[sp] for r in x["resp"]) for sp in SPLITS},
        "union_items_per_thread_train": dist(len(c) for c in cons["groundtruth_with_release_year"]),
        "seen_titles_per_thread": {sp: dist(len(x["seen"]) for x in rows[sp]) for sp in SPLITS},
        "per_split_counts": {sp: dict(c) for sp, c in st.items()},
        "mention_resolution": {sp: {"mentions": st[sp]["mentions"],
                                    "unresolved_rate": st[sp]["mentions_unresolved"] / max(1, st[sp]["mentions"])}
                               for sp in SPLITS},
        "prompt_tokens_chat_template_before_filter": {sp: dist(v) for sp, v in ntok.items()},
        f"threads_dropped_prompt_over_{lim}": {sp: st[sp]["threads_dropped_prompt_over_limit"] for sp in SPLITS},
        "sft_targets": {**dict(st_sft), "teacher_line_grounding": dict(st_t),
                        "teacher_line_grounded_rate": st_t["line_ok"] / max(1, st_t["lines"]),
                        "valid_list_rate": st_sft["threads_with_valid_list"] / max(1, st_sft["train_threads"]),
                        "completion_lines_hist": hist(sft_train_lines, 20)},
        "titles": dict(st_title),
        "catalog": {"size": len(catalog), "gt_items": len(gt_items), "teacher_items": len(t_items),
                    "popular_works_added": len(set(pop) - gt_items - t_items),
                    "titles_with_two_items_within_2_years": collide,
                    "teacher_items_outside_gt": len(t_items - gt_items)},
        "self_check": checks, "self_check_failures": fails,
        "seconds": round(time.time() - t0),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    manifest = {"script": "remark/scripts/books_build.py", "git": git_hash(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": rel(CFG_PATH),
                "inputs": {"threads": "remark/outputs/books/threads/", "grounded": f"remark/outputs/books/ground/grounded_{a.tag}.jsonl",
                           "teacher": rel(tpath), "goodreads_index": "remark/outputs/books/goodreads/"},
                "outputs": written}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: stats[k] for k in ("threads_with_grounded_responder", "rows", "frac_threads_n1", "self_check")},
                     indent=1))
    bad = {s: c["parse_back_pass_rate"] for s, c in checks.items() if isinstance(c, dict) and c.get("parse_back_pass_rate", 1) < 1}
    assert not bad, (bad, fails)


if __name__ == "__main__":
    main()
