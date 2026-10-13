"""Export the SFT targets of Reddit-Books, Reddit-Books-Late and Reddit-Music as catalog item ids (no text).

The SFT completion of a train request is its teacher list: a GPT-5.6 recommendation list grounded to the catalog,
seen items and duplicates removed, one "name (year)" per line. Every responder row of a request has the same completion.
Per dataset the script writes
  {name}_sft_targets.jsonl  request_id, item_ids (catalog indices, in list order)
for the train requests with in_sft_train = true, in the order of {name}_requests.jsonl.

Alignment (checked with asserts): the SFT train rows are the responders of the in_sft_train requests, request by request
in the order of the requests file, n_responders rows each; the ground truth of each row equals item_ids of the
corresponding line of {name}_responses.jsonl, and all rows of a request carry the same completion. Every completion line
must parse and map to a catalog item.

Usage (needs datasets; the local data trees and the files written by release_export_ids.py):
  python remark/scripts/release_export_sft_targets.py --data <dir with reddit_books/, reddit_books_late/, music/> --release <data_release dir>
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

from datasets import load_from_disk

LINE = re.compile(r"^(.*) \((\d{1,4})\)$")
DATASETS = {"books": "reddit_books", "books_late": "reddit_books_late", "music": "music"}


def read_jsonl(p: Path):
    with p.open() as f:
        for line in f:
            yield json.loads(line)


def export(data: Path, rel: Path, name: str) -> dict:
    root = data / DATASETS[name]
    cat = pickle.load(open(root / "catalog.pkl", "rb"))
    idx = {(t, int(y)): i for i, (t, y) in enumerate(cat)}
    assert len(idx) == len(cat)
    ds = load_from_disk(str(root / "sft_dataset/train"))
    comp = [c[-1]["content"] for c in ds["completion"]]
    gts = [[idx[(t, int(y))] for t, y in g] for g in ds["groundtruth_with_release_year"]]
    reqs = [r for r in read_jsonl(rel / f"{name}_requests.jsonl") if r["split"] == "train"]
    resp: dict[str, list[list[int]]] = {}
    for r in read_jsonl(rel / f"{name}_responses.jsonl"):
        resp.setdefault(r["request_id"], []).append(r["item_ids"])
    pos, out, lens = 0, [], []
    for q in reqs:
        if not q["in_sft_train"]:
            continue
        n = q["n_responders"]
        assert gts[pos:pos + n] == resp[q["request_id"]], q["request_id"]
        assert len(set(comp[pos:pos + n])) == 1, q["request_id"]
        items = []
        for line in comp[pos].strip().split("\n"):
            m = LINE.match(line.strip())
            assert m, (q["request_id"], line)
            items.append(idx[(m.group(1), int(m.group(2)))])
        assert len(items) == len(set(items)), q["request_id"]
        out.append({"request_id": q["request_id"], "item_ids": items})
        lens.append(len(items))
        pos += n
    assert pos == len(ds), (name, pos, len(ds))
    with (rel / f"{name}_sft_targets.jsonl").open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n")
    return {"sft_target_requests": len(out), "sft_rows_covered": pos, "items_min": min(lens), "items_max": max(lens),
            "items_mean": round(sum(lens) / len(lens), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data/crs"))
    ap.add_argument("--release", default=str(Path(__file__).resolve().parents[2] / "data_release"))
    ap.add_argument("--only", default="books,books_late,music")
    a = ap.parse_args()
    data, rel = Path(a.data), Path(a.release)
    counts_p = rel / "export_counts.json"
    counts = json.loads(counts_p.read_text())
    for name in a.only.split(","):
        counts[name].update(export(data, rel, name))
        print(name, json.dumps(counts[name]))
    counts_p.write_text(json.dumps(counts, indent=1) + "\n")


if __name__ == "__main__":
    main()
