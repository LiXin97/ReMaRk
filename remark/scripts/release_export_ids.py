"""Export the id-only data release (Reddit-Books, Reddit-Books-Late, Reddit-Music).

No Reddit text is exported. Per dataset the script writes
  Ids (request_id, comment_ids, mbid) are written in UPPER case (lower-case them to recover the public ids); this keeps
  accidental letter sequences in random ids out of text scans.
  {name}_requests.jsonl  request_id (Reddit submission id), split, month, n_responders, in_sft_train, [n_responders_all]
  {name}_responses.jsonl request_id, responder (ordinal 0,1,2,... within the request, never a name or hash),
                         item_ids (catalog indices, in the order the responder named them), comment_ids
  {name}_catalog.tsv     Books: item_id, goodreads_work_ids   |   Music: item_id, mbid, name, year

Alignment (checked with asserts below):
  Books  meta/*.jsonl has one line per dataset row (same order). Train responses come from grpo_train (all train
         threads), validation/test responses from sft_{validation,test}. A responder is the ordinal of the row within
         its submission. item_ids = index in catalog.pkl of the (title, int(year)) key of each ground-truth item.
  Music  has no meta/. The build is replayed on the grounded mentions (promotion filter, year rendering, seen-name
         filter, per-responder de-duplication) to obtain the expected rows per request, and the dataset rows are
         matched to them block by block (requests dropped by the prompt-length filter simply do not match).

Usage (needs datasets; the local data trees):
  python remark/scripts/release_export_ids.py --repo <repo root> --out <output dir>
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

SPLITS = ("train", "validation", "test")


def read_jsonl(p: Path):
    with p.open() as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(p: Path, rows) -> int:
    n = 0
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n")
            n += 1
    return n


def esc_name(name: str) -> str:
    """Names are plain UTF-8 text; tabs and newlines become spaces."""
    return name.replace("\t", " ").replace("\n", " ")


def gt_keys(ds) -> list[list[tuple[str, int]]]:
    return [[(t, int(y)) for t, y in x] for x in ds["groundtruth_with_release_year"]]


# ------------------------------------------------------------------------------------------------ Books
def export_books(repo: Path, src: str, name: str, out: Path) -> dict:
    root = repo / "remark/data/crs" / src
    cat = pickle.load(open(root / "catalog.pkl", "rb"))
    idx = {(t, int(y)): i for i, (t, y) in enumerate(cat)}
    assert len(idx) == len(cat)
    stats = json.load(open(root / "stats.json"))
    blocks = {}  # split -> list of (meta row, gt keys)
    for sp, ds_path, meta_name in (("train", root / "grpo/train", "grpo_train"),
                                   ("validation", root / "sft_dataset/validation", "sft_validation"),
                                   ("test", root / "sft_dataset/test", "sft_test")):
        meta = list(read_jsonl(root / "meta" / f"{meta_name}.jsonl"))
        ds = load_from_disk(str(ds_path))
        gts = gt_keys(ds)
        assert len(meta) == len(ds) == stats["rows"][f"{'grpo/train' if sp == 'train' else 'sft_dataset/' + sp}"], sp
        blocks[sp] = list(zip(meta, gts))
    # train SFT rows: which train requests have an SFT target; same rows/order as grpo rows of those requests
    sft_meta = list(read_jsonl(root / "meta/sft_train.jsonl"))
    sft_ds = load_from_disk(str(root / "sft_dataset/train"))
    sft_gts = gt_keys(sft_ds)
    assert len(sft_meta) == len(sft_ds) == stats["rows"]["sft_dataset/train"]
    sft_pos = {}
    for m, g in zip(sft_meta, sft_gts):
        sft_pos.setdefault(m["submission_id"], []).append((m["comment_ids"], g))
    reqs, resps = [], []
    seen_req, per_split = set(), Counter()
    for sp in SPLITS:
        cur, rows = None, []
        for m, g in blocks[sp]:
            if m["submission_id"] != cur:
                cur = m["submission_id"]
                assert cur not in seen_req, cur
                seen_req.add(cur)
                rows.append([m, 0, []])
            rows[-1][1] += 1
            rows[-1][2].append((m, g))
        for m, n, rr in rows:
            assert n == m["n_responders"], (m["submission_id"], n, m["n_responders"])
            sid = m["submission_id"]
            in_sft = None
            if sp == "train":
                in_sft = sid in sft_pos
                if in_sft:  # same responders in the same order with the same ground truth
                    assert [(r[0]["comment_ids"], r[1]) for r in rr] == sft_pos[sid], sid
            req = {"request_id": sid, "split": sp, "month": m["month"], "n_responders": n}
            if sp == "train":
                req["in_sft_train"] = in_sft
            reqs.append(req)
            per_split[sp + "_requests"] += 1
            for k, (mm, g) in enumerate(rr):
                assert mm["submission_id"] == sid and mm["month"] == m["month"]
                resps.append({"request_id": sid, "responder": k, "item_ids": [idx[x] for x in g],
                              "comment_ids": list(mm["comment_ids"])})
                per_split[sp + "_responses"] += 1
    sft_req = sum(1 for r in reqs if r.get("in_sft_train"))
    assert len(sft_pos) == sft_req
    for sp in SPLITS:
        assert per_split[sp + "_requests"] == stats["threads_with_grounded_responder"][sp], sp
    write_jsonl(out / f"{name}_requests.jsonl", reqs)
    write_jsonl(out / f"{name}_responses.jsonl", resps)
    works = {}
    for i, w in enumerate(read_jsonl(root / "meta/catalog_works.jsonl")):
        assert (w["title"], int(w["year"])) == tuple(cat[i]), i  # same order as catalog.pkl
        works[i] = w["goodreads_work_ids"]
    with (out / f"{name}_catalog.tsv").open("w") as f:
        f.write("item_id\tgoodreads_work_ids\n")
        for i in range(len(cat)):
            f.write(f"{i}\t{','.join(works[i])}\n")
    return {**dict(per_split), "sft_train_requests": sft_req, "sft_train_rows": len(sft_meta), "catalog": len(cat),
            "catalog_items_without_work": sum(1 for i in works if not works[i])}


# ------------------------------------------------------------------------------------------------ Music
def music_split(month: str) -> str | None:
    if "2023-07" <= month <= "2024-06":
        return "train"
    return {"2024-07": "validation", "2024-08": "test"}.get(month)


def export_music(repo: Path, out: Path, min_promo_threads: int = 20) -> dict:
    root = repo / "remark/data/crs/music"
    om = repo / "remark/outputs/music"
    cat = pickle.load(open(root / "catalog.pkl", "rb"))
    idx = {(n, int(y)): i for i, (n, y) in enumerate(cat)}
    assert len(idx) == len(cat)
    stats = json.load(open(root / "stats.json"))
    yrs = pickle.load(open(repo / "remark/data/music/mb_years.pkl", "rb"))

    def item(mb: str):
        n, y, _ = yrs.get(mb, [None, None, None])
        if not n or y is None or (n, int(y)) not in idx:
            return None
        return (n, int(y))

    grounded = {}
    for g in read_jsonl(om / "ground/grounded_full.jsonl"):
        grounded[g["sid"]] = g
    thread_ids, comments = set(), {}
    for p in sorted((om / "threads").glob("*.jsonl")):
        for t in read_jsonl(p):
            if music_split(t["month"]):
                thread_ids.add(t["sid"])
                if t["sid"] in grounded:
                    comments[t["sid"]] = {r["author"]: [c["id"] for c in r["comments"]] for r in t["responders"]}
    pair_threads = Counter()
    for g in grounded.values():
        for author, mbs in g["responders"].items():
            for mb in set(mbs):
                pair_threads[(author, mb)] += 1
    promo = {k for k, c in pair_threads.items() if c >= min_promo_threads}
    assert len(promo) == stats["promotion_pairs"], (len(promo), stats["promotion_pairs"])

    # expected rows per request, replaying the build
    expected = []  # (sid, month, split, seen, [(author, items)], n_all)
    for sid, g in sorted(grounded.items()):
        if sid not in thread_ids:
            continue
        sp = music_split(g["month"])
        seen = sorted({it[0] for it in (item(m) for m in g["request_mbids"]) if it})
        seen_set = set(seen)
        gts = []
        for author, mbs in g["responders"].items():
            res, used = [], set()
            for mb in mbs:
                if (author, mb) in promo:
                    continue
                it = item(mb)
                if it is None or it[0] in seen_set or it in used:
                    continue
                used.add(it)
                res.append(it)
            if res:
                gts.append((author, res))
        if gts:
            expected.append((sid, g["month"], sp, seen, gts, g["n_responders_all"]))

    # match dataset rows to expected requests, block by block
    def match(ds_path, split: str, cand: list):
        ds = load_from_disk(str(ds_path))
        gts, seen = gt_keys(ds), ds["seen_titles"]
        pos, matched, skipped = 0, [], 0
        for e in cand:
            n = len(e[4])
            blk = gts[pos:pos + n]
            if len(blk) == n and all(blk[k] == e[4][k][1] for k in range(n)) and \
                    all(set(e[3]) <= set(seen[pos + k]) for k in range(n)):  # seen names outside the catalog are not visible here
                matched.append(e)
                pos += n
            else:
                skipped += 1
        assert pos == len(ds), (split, pos, len(ds))
        return matched, skipped, len(ds)

    info, reqs, resps, sft_flag = {}, [], [], {}
    for sp in SPLITS:
        cand = [e for e in expected if e[2] == sp]
        src = root / "grpo/train" if sp == "train" else root / "sft_dataset" / sp
        matched, skipped, nrows = match(src, sp, cand)
        assert len(matched) == stats[f"requests_{sp}"], (sp, len(matched))
        assert nrows == stats[f"responders_{sp}"], (sp, nrows)
        info[sp + "_requests"], info[sp + "_responses"], info[sp + "_skipped_by_prompt_filter"] = len(matched), nrows, skipped
        if sp == "train":
            m2, _, nr2 = match(root / "sft_dataset/train", "sft_train", matched)
            sft_flag = {e[0] for e in m2}
            assert nr2 == stats["sft_rows"]["train"]
            info["sft_train_requests"], info["sft_train_rows"] = len(m2), nr2
        for sid, month, _, _, gts, n_all in matched:
            req = {"request_id": sid, "split": sp, "month": month, "n_responders": len(gts), "n_responders_all": n_all}
            if sp == "train":
                req["in_sft_train"] = sid in sft_flag
            reqs.append(req)
            for k, (author, its) in enumerate(gts):
                resps.append({"request_id": sid, "responder": k, "item_ids": [idx[x] for x in its],
                              "comment_ids": list(comments.get(sid, {}).get(author, []))})
    write_jsonl(out / "music_requests.jsonl", reqs)
    write_jsonl(out / "music_responses.jsonl", resps)
    # catalog: every MusicBrainz id that renders to the catalog item; ids used in the grounded data first
    use = Counter(mb for g in grounded.values() for mbs in g["responders"].values() for mb in mbs)
    use.update(mb for g in grounded.values() for mb in g["request_mbids"])
    by_item: dict[tuple, list[str]] = {}
    for mb in yrs:
        it = item(mb)
        if it:
            by_item.setdefault(it, []).append(mb)
    with (out / "music_catalog.tsv").open("w") as f:
        f.write("item_id\tmbid\tname\tyear\n")
        for i, (n, y) in enumerate(cat):
            mbs = sorted(by_item.get((n, int(y)), []), key=lambda m: (-use[m], m))
            name = esc_name(n)
            f.write(f"{i}\t{','.join(mbs)}\t{name}\t{int(y)}\n")
    info["catalog"] = len(cat)
    info["catalog_items_without_mbid"] = sum(1 for it in cat if (it[0], int(it[1])) not in by_item)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="books,books_late,music")
    a = ap.parse_args()
    repo, out = Path(a.repo), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    todo = a.only.split(",")
    if "books" in todo:
        res["books"] = export_books(repo, "reddit_books", "books", out)
    if "books_late" in todo:
        res["books_late"] = export_books(repo, "reddit_books_late", "books_late", out)
    if "music" in todo:
        res["music"] = export_music(repo, out)
    print(json.dumps(res, indent=1))
    (out / "export_counts.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
