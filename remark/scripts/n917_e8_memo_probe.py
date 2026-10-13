"""Memorization probe. A base model (Qwen3.5-2B/4B/9B-Base, no SFT, no chat template) completes a fixed
3-shot prompt: three TRAINING requests (seed 42; 3-10 recommended movies, conversation <= 700 characters) each followed by
the movies Reddit users recommended, then the test request. Greedy decoding until a blank line or the next "Request:".
The completion is parsed with crs_eval's parser and scored with its matching (n917_common.score_and_write), so the output
is in crs_eval format; "hits from the request alone" = responders' items the base model lists.
Usage (node, system python3 with vLLM 0.17.1):
  VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python3 remark/scripts/n917_e8_memo_probe.py \
      --model /tmp/models/Qwen3.5-9B-Base --name memo_probe_9B --out /tmp/work_local/test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import prepare_qwen35  # noqa: E402
from n917_common import DATA, REPO, score_and_write  # noqa: E402

RG = REPO / "remark/third_party/Rank-GRPO"
HEAD = "Movies recommended by Reddit users:"


def conversation(p: str) -> str:
    return p.split("Here is the conversation:", 1)[-1].strip()


def build_shots(seed: int = 42) -> tuple[str, list[dict]]:
    tr = load_from_disk(str(REPO / "remark/data/crs/grpo_consensus/train"))
    idx = [i for i in range(len(tr)) if 3 <= len(tr[i]["groundtruth_with_release_year"]) <= 10
           and len(conversation(tr[i]["prompt"][0]["content"])) <= 700]
    pick = random.Random(seed).sample(idx, 3)
    shots, text = [], ""
    for i in pick:
        ex = tr[i]
        items = sorted(zip(ex["groundtruth_weights"], ex["groundtruth_with_release_year"]), key=lambda x: (-x[0], x[1][0]))
        lines = [f"{t} ({y})" for _, (t, y) in items]
        shots.append({"train_index": i, "items": lines})
        text += f"Request: {conversation(ex['prompt'][0]['content'])}\n{HEAD}\n" + "\n".join(lines) + "\n\n"
    return text, shots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=str(REPO / "remark/outputs/crs/test"))
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--dry_run", action="store_true", help="print the prompt for the first test request and exit")
    a = ap.parse_args()
    ds = load_from_disk(str(DATA / "test"))
    ctxs = list(OrderedDict.fromkeys(p[0]["content"] for p in ds["prompt"]))
    prefix, shots = build_shots()
    prompts = [prefix + f"Request: {conversation(c)}\n{HEAD}\n" for c in ctxs]
    if a.dry_run:
        print(prompts[0]); print(json.dumps(shots)); return
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, **prepare_qwen35(Path(a.model)))
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=400, stop=["\n\n", "Request:"]))
    raw = {c: o.outputs[0].text for c, o in zip(ctxs, outs)}
    s = score_and_write(ds, raw, RG / "reddit_catalog/gt_catalog.pkl", DATA, Path(a.out) / a.name,
                        extra={"model": a.model, "probe": "3-shot base-model completion, greedy, max 400 tokens",
                               "shots": shots})
    print({k: v for k, v in s.items() if not isinstance(v, (dict, list))})


if __name__ == "__main__":
    main()
