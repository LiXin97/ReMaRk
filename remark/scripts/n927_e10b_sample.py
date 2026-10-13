"""Sample G lists per training request from the SFT policy and score them with the union reward.

One process per GPU (--shard i --nshards k). Requests: remark/data/crs/grpo_union/train (72,233 requests, the union
positives of all responders). Generation: vLLM chat with the evaluation formatting, temperature 1, n=G, max 320 tokens
(the P0c cap). Score of a list: the number of its first 20 parsed items that match a positive (crs_eval.hits_for,
truncate mode: catalogue items, seen titles skipped, title + year within 2), i.e. the union reward summed over positions.
Output: <out>/shard<i>.jsonl, one line per request {"idx", "texts", "scores"}; resumable (finished idx are skipped).
A shard is claimed by <out>/shard<i>.claim (atomic mkdir, touched every 5 min); a second process on another node waits
until the claim is released or goes stale (20 min without a touch), then resumes, so two nodes never append to the
same shard.

Usage (node, system python3 with vLLM): CUDA_VISIBLE_DEVICES=4 python3 remark/scripts/n927_e10b_sample.py \
    --model <sft ckpt> --out <dir> --shard 0 --nshards 4
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import threading
import time
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import RG, hits_for, prepare_qwen35, process_rec_raw  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
UNION = REPO / "remark/data/crs/grpo_union/train"


def parse(text: str) -> list[tuple[str, int]]:
    item = {"raw": text}
    err = process_rec_raw(item, "raw", "rec")
    recs = err[1]["rec"] if isinstance(err, tuple) else item["rec"]
    out = []
    for t, y in recs:
        try:
            out.append((t, int(y)))
        except (TypeError, ValueError):
            continue
    return out[:20]


def claim(cl: Path) -> threading.Event:
    """Take the shard's claim directory, waiting while another live process holds it; returns the heartbeat stop."""
    while True:
        if cl.exists() and time.time() - cl.stat().st_mtime > 1200:
            shutil.rmtree(cl, ignore_errors=True)
        try:
            cl.mkdir()
            break
        except FileExistsError:
            print(f"{cl.name} held by another process; waiting", flush=True)
            time.sleep(60)
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(300):
            cl.touch()
    threading.Thread(target=beat, daemon=True).start()
    return stop


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=320)
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard{a.shard}.jsonl"
    cl = out / f"shard{a.shard}.claim"
    stop = claim(cl)
    try:
        sample(a, path)
    finally:
        stop.set()
        shutil.rmtree(cl, ignore_errors=True)


def sample(a: argparse.Namespace, path: Path) -> None:
    done = set()
    if path.exists():
        for line in open(path):
            try:
                done.add(json.loads(line)["idx"])
            except Exception:  # noqa: BLE001 - a line cut by a preemption
                pass
    ds = load_from_disk(str(UNION))
    todo = [i for i in range(a.shard, len(ds), a.nshards) if i not in done]
    print(f"shard {a.shard}/{a.nshards}: {len(todo)} requests to sample ({len(done)} done)", flush=True)
    if not todo:
        return
    catalog = set(tuple(x) for x in pickle.load(open(RG / "reddit_catalog/gt_catalog.pkl", "rb")))
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, seed=a.seed,
              **prepare_qwen35(Path(a.model)))
    sp = SamplingParams(temperature=1.0, n=a.G, max_tokens=a.max_tokens, seed=a.seed)
    with open(path, "a") as f:
        for s in range(0, len(todo), a.chunk):
            idxs = todo[s:s + a.chunk]
            rows = ds.select(idxs)
            outs = llm.chat([[{"role": "user", "content": r["prompt"][0]["content"]}] for r in rows], sp)
            for i, r, o in zip(idxs, rows, outs):
                gts = [(g[0], int(g[1])) for g in r["groundtruth_with_release_year"]]
                seen = set(r["seen_titles"])
                texts = [c.text for c in o.outputs]
                scores = [int(hits_for(parse(t), gts, seen, catalog, "truncate")[0].sum()) for t in texts]
                f.write(json.dumps({"idx": i, "texts": texts, "scores": scores}) + "\n")
            f.flush()
            print(f"shard {a.shard}: {s + len(idxs)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
