"""GRPO on ProtoQA (lists of 10 answers) with the four registered arms (TRL 1.14, qenv).

Arms (amendment, 2026-10-02 09:57 UTC):
  --reward single  (a) per-respondent 0/1 reward, group scaling, rows (question, respondent) sampled uniformly
  --reward rb      (b) exact Rao-Blackwellization of (a) (rb estimator; lib/r9_trainer._rb_hits), rows as (a)
  --reward mean    (c) marginalized mean reward n_c / n at a cluster's first hit, batch scaling, rows as (a)
  --reward union   (d) union (bounded sum) 0/1 reward, group scaling, one row per question (questions uniform)
Everything else is crs_train_grpo_qwen.py's configuration (lr 1e-6 constant, KL 1e-3, clip 0.06/0.08, bnpo, G = 8,
per-device 16 x GA 6 x 4 GPUs = 384 lists = 48 prompts per step, seed 3407) with P0c hygiene always on (shaping in
reward units, completions capped at --p0_max_tokens and trained on, every non-blank line a rank position) and
rec_num = 10. Budget: 250 steps, checkpoints every 25 (model only).

Launch (node, qenv; nltk installed and data staged by n928_r9_node_setup.sh):
  accelerate launch --config_file remark/configs/accel_zero2_4gpu_ga6.yaml remark/scripts/n928_r9_train_grpo.py \
      --reward rb --sft_path <sft ckpt> --output_dir remark/outputs/protoqa/grpo/2B_rb_s3407
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import GRPOConfig

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "remark"))
from lib.r9_common import CRS_DATA, REC_NUM, load_split, prompt_text, setup_nltk  # noqa: E402

setup_nltk()
from lib.r9_rewards import make_r9_reward  # noqa: E402
from lib.r9_trainer import R9RankGRPOTrainer  # noqa: E402

ARMS = {  # reward -> (registered advantage scaling, default training split)
    "single": ("group", CRS_DATA / "grpo/train"),
    "rb": ("group", CRS_DATA / "grpo/train"),
    "mean": ("batch", CRS_DATA / "grpo/train"),
    "union": ("group", CRS_DATA / "grpo_union/train"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=list(ARMS), required=True)
    ap.add_argument("--adv_norm", choices=["group", "batch"], default=None, help="default: the arm's registered scaling")
    ap.add_argument("--train_path", default=None, help="load_from_disk dir (default: the arm's split; stage it to /tmp)")
    ap.add_argument("--sft_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=250)
    ap.add_argument("--save_steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl_beta", type=float, default=1e-3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=6)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_prompt_tokens", type=int, default=512)
    ap.add_argument("--p0_max_tokens", type=int, default=320, help="completion cap (as the Reddit-v2 P0c runs)")
    ap.add_argument("--exact_items", action="store_true", help=": the end bonus needs 10 parser-valid answers")
    ap.add_argument("--rb_weights", choices=["respondents", "clusters"], default="respondents",
                    help="arm (b): one RB row per respondent (the per-responder shaping scale) or per cluster")
    ap.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    ap.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    ap.add_argument("--init_weights", default=None, help="resume after a reclaim: policy weights from this "
                    "model-only checkpoint, loaded after the trainer is built from --sft_path (KL reference = SFT); "
                    "fresh optimizer; pass the remaining --max_steps and seed + k")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no_grad_ckpt", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="check data, prompt lengths and the reward on a few rows, then "
                    "exit before the trainer (no GPU)")
    a = ap.parse_args()
    reg_norm, default_path = ARMS[a.reward]
    adv_norm = a.adv_norm or reg_norm
    if adv_norm != reg_norm:
        print(f"[r9] WARNING: --adv_norm {adv_norm} differs from the registered {reg_norm} for {a.reward}", flush=True)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.sft_path)
    assert tok.eos_token == "<|im_end|>", tok.eos_token  # our Qwen3.5 SFT
    train = load_from_disk(a.train_path or str(default_path))
    if a.reward in ("single", "rb", "mean"):
        assert min(train.data.column("resp_cluster").to_pylist()) >= 0, "arms a/b/c need the per-row split (grpo/train)"
    # prompts are short: check every training question's prompt length once instead of filtering 765k rows
    qs = load_split("train")
    plen = {q["qid"]: len(tok(tok.apply_chat_template([{"role": "user", "content": prompt_text(q["question"])}],
                                                      tokenize=False, add_generation_prompt=True))["input_ids"])
            for q in qs}
    assert max(plen.values()) <= a.max_prompt_tokens, max(plen.values())
    (out / "data_filter.json").write_text(json.dumps({"rows_before": len(train), "rows_after": len(train),
                                                      "max_prompt_tokens_seen": max(plen.values())}))
    rb_table = None
    if a.reward == "rb":  # {prompt text: [cluster counts of each training board]} (training labels only; the RB hits use
        rb_table = {}       # the row's own all_clusters / cluster_counts, the table only checks them)
        for q in qs:
            rb_table.setdefault(prompt_text(q["question"]), []).append(q["counts"])
    reward = make_r9_reward(a.reward, rec_num=REC_NUM)
    if a.dry_run:  # the reward on 8 random rows, each completed by its own question's cluster strings (first string each)
        import random
        ex = train.select(random.Random(a.seed).sample(range(len(train)), 8))
        comps = [[{"role": "assistant", "content": "\n".join(c[0] for c in e["all_clusters"])}] for e in ex]
        cols = {k: list(ex[k]) for k in ex.column_names if k != "prompt"}
        r = reward(prompts=list(ex["prompt"]), completions=comps, completion_ids=[[]] * len(comps), **cols)
        print(json.dumps({"rows": len(train), "max_prompt_tokens_seen": max(plen.values()), "adv_norm": adv_norm,
                          "rb_boards": sum(len(v) for v in rb_table.values()) if rb_table else None,
                          "sample_rewards": [[round(x, 3) for x in v] for v in r], "reward_stats": reward.last_stats},
                         indent=1))
        return
    cfg = GRPOConfig(
        output_dir=str(out), max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="constant",
        adam_beta1=0.9, adam_beta2=0.99, beta=a.kl_beta, epsilon=0.06, epsilon_high=0.08, loss_type="bnpo",
        importance_sampling_level="token", num_iterations=1, scale_rewards="group",
        vllm_importance_sampling_correction=False, per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps, num_generations=a.num_generations,
        max_completion_length=a.p0_max_tokens, mask_truncated_completions=False,  # P0c: truncated lists are trained on
        temperature=1.0, bf16=True, optim="adamw_torch_fused", gradient_checkpointing=not a.no_grad_ckpt, use_vllm=True,
        vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=a.vllm_tensor_parallel_size, vllm_max_model_length=a.max_prompt_tokens + a.p0_max_tokens,
        save_strategy="steps", save_steps=a.save_steps, save_only_model=True, logging_steps=5, seed=a.seed,
        report_to="none", run_name=out.name, model_init_kwargs={"dtype": "bfloat16"})
    trainer = R9RankGRPOTrainer(model=a.sft_path, reward_funcs=reward, args=cfg, train_dataset=train,
                                processing_class=tok, rec_num=REC_NUM, adv_norm=adv_norm, shaping_units="reward",
                                line_positions="lines", exact_requires_items=a.exact_items, rb_table=rb_table,
                                rb_weights=a.rb_weights)
    if trainer.accelerator.is_main_process:
        git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        (out / "r9_run.json").write_text(json.dumps({"args": vars(a), "adv_norm": adv_norm, "rec_num": REC_NUM,
                                                     "train_path": a.train_path or str(default_path),
                                                     "train_rows": len(train), "git": git,
                                                     "rb_M": getattr(trainer, "r9_M", None)}, indent=1))
    if a.init_weights:
        from safetensors.torch import load_file
        state = {}
        for f in sorted(Path(a.init_weights).glob("*.safetensors")):
            state.update(load_file(str(f)))
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        assert not unexpected and all("lm_head" in k for k in missing), (missing[:5], unexpected[:5])
        print(f"[resume] policy initialised from {a.init_weights} ({len(state)} tensors; missing {missing})", flush=True)
    trainer.train()


if __name__ == "__main__":
    main()
