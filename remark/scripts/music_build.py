"""Build the music music dataset in the Reddit-v2 / Reddit-Books on-disk format.

Request = r/musicsuggestions submission; responders = its first 20 top-level commenters (music_extract.py); mentions
grounded to MusicBrainz artists (music_ground.py). Rules registered before the build:
  item          (MusicBrainz canonical name, begin year); rendered "Name (Year)" and kept only if Rank-GRPO's parser
                reads it back exactly; artists without a begin year are dropped
  promotion     an (author, artist) pair in >= PROMO_THREADS distinct threads is request-independent promotion: all of
                its mentions are dropped
  seen_titles   names of the artists the requester names
  GT(responder) the responder's rendered artists in order, de-duplicated, minus seen names; empty -> no row
  SFT (train)   the request's grounded GPT-5.6 teacher list, seen names and duplicates removed, kept if >= 15 items
  catalogue     every rendered GT item and teacher item used + the 20,000 most active MusicBrainz artists with a year
  prompt        <= 2,048 tokens (Qwen3.5 chat template), as for Reddit-Books
Year-matching check (registered): share of teacher lines that ground to an artist whose MusicBrainz begin year is more
than 2 away from the year the teacher wrote.

Usage: /tmp/qenv/bin/python remark/scripts/music_build.py --tag full   (or env/bin/python on the CPU host)
Outputs under remark/data/crs/music/: sft_dataset/{train,validation,test}, grpo/train, grpo_consensus/train,
grpo_union/train, catalog.pkl, stats.json, manifest.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from music_common import OUTM, REPO, canon_ok, prompt_text, render, split_of
from music_ground import DOMINANCE, MIN_ACT, candidates

OUT = REPO / "remark/data/crs/music"
PROMO_THREADS, MIN_SFT, TOP_N, MAX_PROMPT = 20, 15, 20000, 2048


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    idx = pickle.load(open(REPO / "remark/data/music/mb_index.pkl", "rb"))
    names, arts = idx["names"], idx["artists"]
    yrs = pickle.load(open(REPO / "remark/data/music/mb_years.pkl", "rb"))
    st = Counter()

    def item(mb: str) -> tuple[str, int] | None:
        n, y, _ = yrs.get(mb, [None, None, None])
        if not n or y is None:
            return None
        return (n, y) if canon_ok(n, y) else None

    def resolve(name: str) -> str | None:  # unique, else the dominant-activity candidate (no thread context here)
        c = candidates(names, name)
        if len(c) == 1:
            return c[0]
        if len(c) > 1:
            acts = sorted(((arts[x][3] if x in arts else 0), x) for x in c)
            if acts[-1][0] >= MIN_ACT and acts[-1][0] >= DOMINANCE * max(acts[-2][0], 0.5):
                return acts[-1][1]
        return None

    threads = {}
    for p in sorted((OUTM / "threads").glob("*.jsonl")):
        for line in open(p):
            t = json.loads(line)
            if split_of(t["month"]):
                threads[t["sid"]] = t
    grounded = {json.loads(l)["sid"]: json.loads(l) for l in open(OUTM / "ground" / f"grounded_{a.tag}.jsonl")}
    # promotion filter
    pair_threads = Counter()
    for g in grounded.values():
        for author, mbs in g["responders"].items():
            for mb in set(mbs):
                pair_threads[(author, mb)] += 1
    promo = {k for k, c in pair_threads.items() if c >= PROMO_THREADS}

    teacher = {}
    tp = OUTM / "sft" / f"teacher_{a.tag}.jsonl"
    year_lines = year_off = 0
    if tp.exists():
        for line in open(tp):
            r = json.loads(line)
            items = []
            for nm, ty in r["lines"]:
                mb = resolve(nm)
                st["teacher_lines"] += 1
                if not mb:
                    st["teacher_ungrounded"] += 1; continue
                y = yrs.get(mb, [None, None])[1]
                if y is not None:
                    year_lines += 1; year_off += abs(int(ty) - y) > 2
                it = item(mb)
                if it is None:
                    st["teacher_no_year_or_parse"] += 1; continue
                items.append(it)
            teacher[r["sid"]] = items

    tok = AutoTokenizer.from_pretrained(str(REPO / "data/models/Qwen3.5-2B-Base"))
    def prompt_ok(text: str) -> bool:
        s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        return len(tok(s)["input_ids"]) <= MAX_PROMPT

    rows = {"train": [], "validation": [], "test": []}
    grpo, cons = [], []
    catalog = set()
    for sid, g in sorted(grounded.items()):
        t = threads.get(sid)
        if not t:
            continue
        sp = split_of(t["month"])
        seen = sorted({it[0] for it in (item(m) for m in g["request_mbids"]) if it})
        seen_set = set(seen)
        gts = {}
        for author, mbs in g["responders"].items():
            out, used = [], set()
            for mb in mbs:
                st["mentions_grounded"] += 1
                if (author, mb) in promo:
                    st["dropped_promotion"] += 1; continue
                it = item(mb)
                if it is None:
                    st["dropped_no_year_or_parse"] += 1; continue
                if it[0] in seen_set:
                    st["dropped_seen"] += 1; continue
                if it in used:
                    continue
                used.add(it); out.append(it)
            if out:
                gts[author] = out
        if not gts:
            st[f"requests_no_gt_{sp}"] += 1; continue
        prompt = prompt_text(t["context"])
        if not prompt_ok(prompt):
            st[f"requests_prompt_too_long_{sp}"] += 1; continue
        pr = [{"role": "user", "content": prompt}]
        for its in gts.values():
            catalog.update(its)
        if sp == "train":
            tl = [it for it in teacher.get(sid, []) if it[0] not in seen_set]
            tl = list(dict.fromkeys(tl))
            if len(tl) >= MIN_SFT:
                comp = [{"role": "assistant", "content": "\n".join(render(*it) for it in tl[:20])}]
                catalog.update(tl[:20])
                for its in gts.values():
                    rows["train"].append({"prompt": pr, "completion": comp, "seen_titles": seen,
                                          "groundtruth_with_release_year": [[n, str(y)] for n, y in its]})
            else:
                st["train_requests_without_sft_target"] += 1
            for its in gts.values():
                grpo.append({"prompt": pr, "seen_titles": seen, "groundtruth_with_release_year": [[n, str(y)] for n, y in its]})
            cnt = Counter(it for its in gts.values() for it in its)
            n = len(gts)
            u = sorted(cnt, key=lambda x: (-cnt[x], x))
            cons.append({"prompt": pr, "seen_titles": seen, "groundtruth_with_release_year": [[x[0], str(x[1])] for x in u],
                         "w_frac": [cnt[x] / n for x in u], "n_responders": n})
        else:
            for its in gts.values():
                rows[sp].append({"prompt": pr, "completion": [{"role": "assistant", "content": "\n".join(render(*it) for it in its)}],
                                 "seen_titles": seen, "groundtruth_with_release_year": [[n, str(y)] for n, y in its]})
        st[f"requests_{sp}"] += 1
        st[f"responders_{sp}"] += len(gts)

    # catalogue: + the most active MusicBrainz artists with a renderable year
    top = sorted(((v[3], k) for k, v in arts.items() if v[3] > 0), reverse=True)
    added = 0
    for _, mb in top:
        it = item(mb)
        if it:
            catalog.add(it); added += 1
            if added >= TOP_N:
                break
    OUT.mkdir(parents=True, exist_ok=True)
    for sp, rs in rows.items():
        Dataset.from_list(rs).save_to_disk(str(OUT / "sft_dataset" / sp))
    Dataset.from_list(grpo).save_to_disk(str(OUT / "grpo" / "train"))
    Dataset.from_list([{k: v for k, v in c.items() if k != "w_frac"} | {"groundtruth_weights": c["w_frac"]} for c in cons]
                      ).save_to_disk(str(OUT / "grpo_consensus" / "train"))
    Dataset.from_list([{k: v for k, v in c.items() if k != "w_frac"} | {"groundtruth_weights": [1.0] * len(c["w_frac"])} for c in cons]
                      ).save_to_disk(str(OUT / "grpo_union" / "train"))
    pickle.dump(sorted(catalog), open(OUT / "catalog.pkl", "wb"))
    stats = {**dict(st), "promotion_pairs": len(promo), "catalog": len(catalog), "catalog_top_mb_added": added,
             "sft_rows": {k: len(v) for k, v in rows.items()}, "grpo_rows": len(grpo), "consensus_requests": len(cons),
             "mean_responders_train": round(st["responders_train"] / max(st["requests_train"], 1), 3),
             "teacher_year_check": {"lines_with_mb_year": year_lines, "year_off_by_more_than_2": year_off,
                                    "share": round(year_off / max(year_lines, 1), 4)}}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    (OUT / "manifest.json").write_text(json.dumps({
        "inputs": {"grounded": f"outputs/music/ground/grounded_{a.tag}.jsonl", "teacher": str(tp.relative_to(REPO)),
                   "mb_index": "remark/data/music/mb_index.pkl", "mb_years": "remark/data/music/mb_years.pkl"},
        "rules": {"promo_threads": PROMO_THREADS, "min_sft_items": MIN_SFT, "top_n_catalog": TOP_N, "max_prompt_tokens": MAX_PROMPT},
        "git": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()},
        indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
