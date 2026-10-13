"""Zero-shot chat model on test with vLLM, the dataset's own instruction (the test "prompt" message,
as crs_eval.py sends it) and crs_eval's parser/matching (via n917_common.score_and_write).

For Qwen3.5-27B (post-trained; thinking model) thinking is switched off through the chat template
(enable_thinking=False) so the answer is the list itself; greedy, max 1,024 new tokens (same as crs_eval).
Usage (node, system python3 with vLLM 0.17.1):
  VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0 python3 remark/scripts/n917_zeroshot_vllm.py \
      --model /tmp/models/27B --name test_zeroshot_qwen35_27B --dataset reddit --out /tmp/work_local/test
  (then --dataset books --name test_zeroshot_qwen35_27B_books)
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crs_eval import prepare_qwen35  # noqa: E402
from n917_common import DATA, REPO, score_and_write  # noqa: E402

RG = REPO / "remark/third_party/Rank-GRPO"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dataset", choices=["reddit", "books"], required=True)
    ap.add_argument("--out", default=str(REPO / "remark/outputs/crs/test"))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--thinking", action="store_true", help="leave the model's thinking mode on (default: off)")
    ap.add_argument("--max_tokens", type=int, default=1024)
    a = ap.parse_args()
    if a.dataset == "reddit":
        data, catalog = DATA, RG / "reddit_catalog/gt_catalog.pkl"
    else:
        B = REPO / "remark/data/crs/reddit_books"
        data, catalog = B / "sft_dataset", B / "catalog.pkl"
    ds = load_from_disk(str(Path(data) / "test"))
    ctxs = list(OrderedDict.fromkeys(p[0]["content"] for p in ds["prompt"]))
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util, max_model_len=8192,
              **prepare_qwen35(Path(a.model)))
    outs = llm.chat([[{"role": "user", "content": c}] for c in ctxs],
                    SamplingParams(temperature=0, max_tokens=a.max_tokens),
                    chat_template_kwargs={"enable_thinking": bool(a.thinking)})
    raw = {}
    for c, o in zip(ctxs, outs):
        t = o.outputs[0].text
        raw[c] = t.split("</think>", 1)[-1] if "</think>" in t else t
    s = score_and_write(ds, raw, catalog, data, Path(a.out) / a.name,
                        extra={"model": a.model, "zero_shot": True, "thinking": bool(a.thinking),
                               "sampling": f"greedy, max_tokens {a.max_tokens}"})
    print({k: v for k, v in s.items() if not isinstance(v, dict)})


if __name__ == "__main__":
    main()
