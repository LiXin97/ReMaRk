"""Rank-GRPO training with a selectable reward: single-responder (Rank-GRPO exp_inf) or consensus.

Adapted from third_party/Rank-GRPO/train_rank_grpo.py (Zhu et al., ICLR 2026); hyperparameters follow the
Llama-3.2-3B setup there (lr 1e-6 -> 1e-7 after step 8000, KL 1e-3, clip 0.06/0.08, item-level IS, mu=1,
8 generations). Differences: explicit paths, --reward {single,consensus}, --max_steps for equal budgets.

consensus reward: per rank position, the weight (fraction of responders) of the ground-truth item it matches;
this is the expectation of the single-responder reward over the context's responders.

Launch (inside the Rank-GRPO env, from the Rank-GRPO repo dir so `libs` is importable):
  accelerate launch --config_file <cfg> remark/scripts/crs_train_grpo.py --reward consensus ...
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from functools import wraps
from pathlib import Path

import numpy as np
from datasets import load_from_disk

REPO = Path(__file__).resolve().parents[2]
RG = REPO / "remark/third_party/Rank-GRPO"
sys.path.insert(0, str(RG))
from trl import GRPOConfig, RankGRPOTrainer  # noqa: E402  (installed by Rank-GRPO INSTALL.SH)

from libs.data import load_catalog  # noqa: E402
from libs.metrics_align import _default_title_normalizer  # noqa: E402
from libs.reward_funcs import make_reward_func_individual  # noqa: E402
from libs.utils import StepLRSchedulerCallback, process_rec_raw  # noqa: E402


def weighted_hits(recs, gts, weights, seen, catalog, rec_num=20, tol=2):
    """Per-position reward: weight of the first unmatched GT with the same normalized title and |dy| <= tol."""
    out = np.zeros(rec_num)
    remaining = {}
    for (t, y), w in zip(gts, weights):
        try:
            remaining.setdefault(_default_title_normalizer(t), []).append([int(str(y)), float(w)])
        except ValueError:
            continue
    for pos, (rt, ry) in enumerate(recs[:rec_num]):
        if rt in seen or (rt, ry) not in catalog:
            continue
        cands = remaining.get(_default_title_normalizer(rt))
        if not cands:
            continue
        best = min(range(len(cands)), key=lambda k: abs(cands[k][0] - int(ry)))
        if abs(cands[best][0] - int(ry)) <= tol:
            out[pos] = cands[best][1]
            cands.pop(best)
    return out


def make_consensus_reward(catalog, rec_num=20):
    def reward_consensus(completions, groundtruth_with_release_year, groundtruth_weights, seen_titles, **kwargs):
        rewards = []
        for comp, gts, ws, seen in zip(completions, groundtruth_with_release_year, groundtruth_weights, seen_titles):
            item = {"raw": comp[0]["content"]}
            err, item = process_rec_raw(item, "raw", "recs")
            if err:
                rewards.append([0.0] * rec_num)
                continue
            recs = [(t, int(y)) for t, y in item["recs"]]
            rewards.append(weighted_hits(recs, gts, ws, set(seen), catalog, rec_num).tolist())
        return rewards
    return reward_consensus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["single", "consensus"], required=True)
    ap.add_argument("--train_path", required=True, help="load_from_disk dir of the train split")
    ap.add_argument("--sft_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--catalog_path", default=str(RG / "reddit_catalog/gt_catalog.pkl"))
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lr_drop_step", type=int, default=8000)
    ap.add_argument("--kl_beta", type=float, default=1e-3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=6)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=4)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.5)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--wandb_project", default="work_crs")
    ap.add_argument("--init_weights", default=None, help=" resume: policy weights from this model-only checkpoint, "
                    "loaded after the trainer is built from --sft_path (KL reference stays the SFT policy); fresh optimizer")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="disable gradient checkpointing (systems-only change)")
    a = ap.parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")
    from accelerate import Accelerator
    Accelerator()  # same as Rank-GRPO libs.logs.setup_environment (that module has a circular self-import)
    os.environ["WANDB_PROJECT"] = a.wandb_project
    catalog = load_catalog(a.catalog_path)
    train = load_from_disk(a.train_path)
    reward = (make_reward_func_individual(rec_num=20, gt_catalog=catalog) if a.reward == "single"
              else make_consensus_reward(catalog))
    callback = StepLRSchedulerCallback(schedule=[(a.lr_drop_step, 1e-7)], verbose=False)
    cfg = GRPOConfig(
        importance_sampling_level="item", adam_beta1=0.9, adam_beta2=0.99, beta=a.kl_beta, epsilon=0.06,
        epsilon_high=0.08, output_dir=a.output_dir, per_device_train_batch_size=a.per_device_train_batch_size,
        max_steps=a.max_steps, num_iterations=1, save_strategy="steps", save_steps=a.save_steps, logging_steps=10,
        bf16=True, gradient_accumulation_steps=a.gradient_accumulation_steps, learning_rate=a.lr,
        optim="paged_adamw_8bit", lr_scheduler_type="constant", use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.vllm_gpu_memory_utilization, vllm_tensor_parallel_size=a.vllm_tensor_parallel_size,
        seed=a.seed, max_prompt_length=2048, max_completion_length=1024, num_generations=a.num_generations,
        gradient_checkpointing=not a.no_grad_ckpt, run_name=Path(a.output_dir).name, report_to="none", save_only_model=True)
    trainer = RankGRPOTrainer(model=a.sft_path, reward_funcs=reward, args=cfg, train_dataset=train, callbacks=[callback])
    if a.init_weights:
        from safetensors.torch import load_file
        state = {}
        for f in sorted(Path(a.init_weights).glob("*.safetensors")):
            state.update(load_file(str(f)))
        params = list(trainer.model.parameters())
        if any(hasattr(q, "ds_id") for q in params):  # ZeRO-3: parameters are already sharded
            import deepspeed
            import torch.distributed as dist
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                missing, unexpected = trainer.model.load_state_dict(state, strict=False) if dist.get_rank() == 0 else ([], [])
        else:
            missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        assert not unexpected and all("lm_head" in k for k in missing), (missing[:5], unexpected[:5])
        print(f"[resume] policy initialised from {a.init_weights} ({len(state)} tensors; missing {missing})", flush=True)
    trainer.train()


if __name__ == "__main__":
    main()
