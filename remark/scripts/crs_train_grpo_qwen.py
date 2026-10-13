"""Rank-GRPO training of a Qwen3.5 (or Llama-3.x) SFT checkpoint with a single-responder or consensus reward (TRL 1.14, qenv).

Hyperparameters follow crs_train_grpo.py (the Llama-3.2-3B Rank-GRPO setup): lr 1e-6 constant, KL 1e-3, clip
0.06/0.08, bnpo loss, 8 generations, per-device batch 16 x GA 6 x 4 GPUs, no vLLM importance-sampling correction
(TRL 0.21 had none). TRL 1.14 no longer truncates prompts, so contexts whose prompt exceeds --max_prompt_tokens
(Rank-GRPO's max_prompt_length) are dropped, identically for both rewards.

Launch (qenv, node): accelerate launch --config_file remark/configs/accel_zero2_4gpu_ga6.yaml \
    remark/scripts/crs_train_grpo_qwen.py --reward consensus --adv_norm batch --sft_path ... --output_dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import GRPOConfig

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "remark"))
from lib.crs_rewards import RG, make_consensus_reward, make_reward_func_individual  # noqa: E402
from lib.rank_grpo_trl import RankGRPOTrainer  # noqa: E402

sys.path.insert(0, str(RG))
from libs.data import load_catalog  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["single", "consensus"], required=True)
    ap.add_argument("--adv_norm", choices=["group", "batch", "fixed"], required=True)
    ap.add_argument("--adv_scale", type=float, default=None, help="E1 isolation (10-07): constant advantage denominator "
                    "for --adv_norm fixed (required with it)")
    ap.add_argument("--shaping_scale_const", type=float, default=None, help="E1 isolation (10-07): constant P0 shaping "
                    "scale instead of 1 / mean per-position std")
    ap.add_argument("--train_path", required=True, help="load_from_disk dir of the train split")
    ap.add_argument("--sft_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--catalog_path", default=str(RG / "reddit_catalog/gt_catalog.pkl"))
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--lr_scheduler", choices=["constant", "cosine"], default="constant",
                    help="cosine = transformers cosine from --lr to 0 over max_steps, no warmup")
    ap.add_argument("--lr_sched_offset", type=int, default=0, help="resume after a reclaim: steps already trained; "
                    "the cosine continues at offset + step over --lr_sched_total steps instead of restarting")
    ap.add_argument("--lr_sched_total", type=int, default=None, help="total steps of the cosine (default offset + max_steps)")
    ap.add_argument("--kl_beta", type=float, default=1e-3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=6)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=2048)
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--init_weights", default=None, help="resume after a reclaim: policy weights from this "
                    "model-only checkpoint, loaded after the trainer is built from --sft_path, so the KL reference stays the SFT "
                    "policy; the optimizer state starts fresh")
    ap.add_argument("--router_aux_loss_coef", type=float, default=None, help="MoE (35B-A3B): 0.0 disables the "
                    "load-balancing loss so the GRPO objective matches the dense models")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--shaping_mult", type=float, default=1.0,
                    help="multiply the three P0 length-shaping terms (end +0.1, overflow -0.1, early -0.1)")
    ap.add_argument("--p0", action="store_true", help="length stabilization: shaping in reward units, completions "
                    "capped at --p0_max_tokens with truncated completions masked out of the loss")
    ap.add_argument("--p0_max_tokens", type=int, default=320)
    ap.add_argument("--exact_items", action="store_true", help="the end bonus needs rec_num parser-valid items "
                    "(a malformed line no longer counts toward an exact-length list)")
    ap.add_argument("--p0c", action="store_true", help="P0c: every non-blank line takes a rank position; "
                    "malformed lines are zero-reward misses (released-trainer semantics); use together with --p0")
    ap.add_argument("--rb", action="store_true", help="exact Rao-Blackwellization of Rank-GRPO's group-scaled "
                    "estimator (needs --reward single on the per-row split and --adv_norm group)")
    ap.add_argument("--rb_table_path", default=None, help="E3 (10-07): build the --rb table (every responder's ground truth "
                    "per request) from this per-row split instead of the training split, so --rb works with request-uniform "
                    "sampling (--train_path = one random responder per request)")
    ap.add_argument("--seq_level", action="store_true", help="E4 (10-07): ordinary sequence-level GRPO, one recall@20 "
                    "reward per list, group-scaled advantage on every completion token (needs --adv_norm group)")
    ap.add_argument("--dyn_sample", type=int, default=1, help="DAPO-style dynamic sampling, generate for this many "
                    "times the prompts and keep groups with reward signal first (1 = off)")
    a = ap.parse_args()
    out = Path(a.output_dir)
    tok = AutoTokenizer.from_pretrained(a.sft_path)
    # Qwen3.5 (our SFT sets <|im_end|>) or Llama-3.x instruct (native <|eot_id|>). Llama ships no pad token: use its
    # reserved pad token, never the EOS, so the end-of-list shaping sees the real end of a completion.
    assert tok.eos_token in ("<|im_end|>", "<|eot_id|>"), tok.eos_token
    if tok.pad_token is None:
        assert "<|finetune_right_pad_id|>" in tok.get_vocab(), "no pad token"
        tok.pad_token = "<|finetune_right_pad_id|>"
    catalog = load_catalog(a.catalog_path)
    train = load_from_disk(a.train_path)
    n_before = len(train)
    def short(ex):
        text = tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        return len(tok(text)["input_ids"]) <= a.max_prompt_tokens
    train = train.filter(short, num_proc=16)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_filter.json").write_text(json.dumps({"rows_before": n_before, "rows_after": len(train)}))
    reward = (make_reward_func_individual(rec_num=20, gt_catalog=catalog) if a.reward == "single"
              else make_consensus_reward(catalog))
    cfg = GRPOConfig(
        output_dir=str(out), max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type=a.lr_scheduler,
        adam_beta1=0.9, adam_beta2=0.99, beta=a.kl_beta, epsilon=0.06, epsilon_high=0.08, loss_type="bnpo",
        importance_sampling_level="token", num_iterations=1, scale_rewards="group",
        vllm_importance_sampling_correction=False, per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps, num_generations=a.num_generations,
        max_completion_length=a.p0_max_tokens if a.p0 else 1024,
        # P0 masked truncated completions; that made an all-truncated batch an absorbing state (consensusrw_p0 collapsed
        # at step ~1,790). P0c keeps the cap but trains on truncated completions (no EOS bonus, overflow penalties apply)
        mask_truncated_completions=a.p0 and not a.p0c, temperature=1.0, bf16=True, optim="adamw_torch_fused",
        gradient_checkpointing=not a.no_grad_ckpt, use_vllm=True, vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.vllm_gpu_memory_utilization, vllm_tensor_parallel_size=a.vllm_tensor_parallel_size,
        vllm_max_model_length=a.max_prompt_tokens + (a.p0_max_tokens if a.p0 else 1024), save_strategy="steps", save_steps=a.save_steps,
        save_only_model=True, logging_steps=10, seed=a.seed, report_to="none", run_name=out.name,
        model_init_kwargs={"dtype": "bfloat16"},
        **({} if a.router_aux_loss_coef is None else {"router_aux_loss_coef": a.router_aux_loss_coef}))
    rb_table = None
    assert a.rb_table_path is None or a.rb, "--rb_table_path needs --rb"
    if a.rb:  # every responder's ground truth per request (training labels), keyed by the prompt text
        assert a.reward == "single" and a.adv_norm == "group", "--rb is defined for Rank-GRPO (single reward, group scale)"
        rb_table = {}
        src = load_from_disk(a.rb_table_path) if a.rb_table_path else train
        for pr, gt in zip(src["prompt"], src["groundtruth_with_release_year"]):
            rb_table.setdefault(pr[0]["content"], []).append(gt)
        (out / "rb_table.json").write_text(json.dumps({"requests": len(rb_table), "rows": len(src), "table_path": a.rb_table_path,
                                                       "max_responders": max(len(v) for v in rb_table.values())}))
    trainer = RankGRPOTrainer(model=a.sft_path, reward_funcs=reward, args=cfg, train_dataset=train,
                              rb_table=rb_table, rb_catalog=catalog if a.rb else None, dyn_factor=a.dyn_sample, seq_level=a.seq_level,
                              adv_scale=a.adv_scale, shaping_scale_const=a.shaping_scale_const,
                              processing_class=tok, adv_norm=a.adv_norm, shaping_units="reward" if a.p0 else "fixed",
                              line_positions="lines" if a.p0c else "items", exact_requires_items=a.exact_items,
                              end_of_list_reward=0.1 * a.shaping_mult, extra_token_penalty=-0.1 * a.shaping_mult,
                              early_stop_penalty=-0.1 * a.shaping_mult)
    if a.lr_scheduler == "cosine" and a.lr_sched_offset:  # same curve as an uninterrupted cosine run
        import math
        from torch.optim.lr_scheduler import LambdaLR
        off, tot = a.lr_sched_offset, a.lr_sched_total or a.lr_sched_offset + a.max_steps
        def create_scheduler(num_training_steps, optimizer=None):
            if trainer.lr_scheduler is None:
                trainer.lr_scheduler = LambdaLR(optimizer if optimizer is not None else trainer.optimizer,
                                                lambda s: max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(off + s, tot) / tot))))
                trainer._created_lr_scheduler = True
            return trainer.lr_scheduler
        trainer.create_scheduler = create_scheduler
    if a.init_weights:
        from safetensors.torch import load_file
        files = sorted(Path(a.init_weights).glob("*.safetensors"))
        state = {}
        for f in files:
            state.update(load_file(str(f)))
        state = fuse_moe_experts(state)
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
    trainer.train()  # checkpoints are model-only (for per-checkpoint validation), so a killed run restarts


def fuse_moe_experts(state: dict) -> dict:
    """35B-A3B resume (2026-10-02): checkpoints store MoE experts per expert (experts.<e>.{gate,up,down}_proj.weight)
    while transformers 5.3 holds them fused in memory (experts.gate_up_proj [E, 2I, H], experts.down_proj [E, H, I]),
    so a raw load_state_dict missed every expert and the resume assertion fired. Fuse them the way transformers' own
    loader does (experts stacked, gate before up along dim 1; verified bit-identical on three layers of the 35B-A3B SFT
    checkpoint). Dense checkpoints have no such keys and pass through unchanged."""
    import re
    import torch
    groups, out = {}, {}
    for k, v in state.items():
        m = re.match(r"(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", k)
        if m:
            groups.setdefault(m.group(1), {}).setdefault(m.group(3), {})[int(m.group(2))] = v
        else:
            out[k] = v
    for prefix, parts in groups.items():
        n = len(parts["gate_proj"])
        gate = torch.stack([parts["gate_proj"][e] for e in range(n)])
        up = torch.stack([parts["up_proj"][e] for e in range(n)])
        out[prefix + ".gate_up_proj"] = torch.cat([gate, up], dim=1)
        out[prefix + ".down_proj"] = torch.stack([parts["down_proj"][e] for e in range(n)])
    return out


if __name__ == "__main__":
    main()
