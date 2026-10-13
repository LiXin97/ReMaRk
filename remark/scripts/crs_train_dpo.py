"""DPO from our SFT checkpoint on the pooled-positive pairs (n927_e10b_pairs.py).

TRL DPOTrainer, sigmoid loss, beta 0.1, the SFT model as the frozen reference, learning rate 1e-6 (the RL arms'), cosine
with 5% warm-up, effective batch 64 pairs, one epoch, checkpoints every --save_steps (model only). Qwen3.5 quirks as in
crs_train_sft.py: the text tokenizer is the processing class and <|im_end|> is the EOS; checkpoints keep the VLM layout,
which vLLM (crs_eval.py) loads directly.
Resume after a preemption (rule): --init_weights <checkpoint> --skip <pairs already seen>, fresh optimizer, the
schedule continued with --lr <lr at that step> --warmup_ratio 0; the data order is fixed (shuffled once with seed 3407
when the pairs were built), so skipping the first pairs resumes the same epoch.

Launch (qenv, node): accelerate launch --config_file remark/configs/accel_zero2_4gpu_ga<GA>.yaml remark/scripts/crs_train_dpo.py \
    --model <sft ckpt> --pairs <dir>/train --output_dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None, help="text tokenizer dir (default: --model)")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--init_weights", default=None)
    ap.add_argument("--skip", type=int, default=0, help="pairs already trained on (resume)")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--epochs", type=float, default=1.0, help=" amendment: the 2B sweep uses 3")
    a = ap.parse_args()

    train = load_from_disk(a.pairs)
    if a.skip:
        train = train.select(range(a.skip, len(train)))
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    tok.eos_token = "<|im_end|>"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = DPOConfig(
        output_dir=a.output_dir, beta=a.beta, loss_type="sigmoid", learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_ratio=a.warmup_ratio, num_train_epochs=a.epochs, per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps, max_length=a.max_length, bf16=True,
        gradient_checkpointing=a.grad_ckpt, save_strategy="steps", save_steps=a.save_steps, save_only_model=True,
        logging_steps=10, report_to="none", seed=a.seed, run_name=Path(a.output_dir).name, dataset_num_proc=16,
        model_init_kwargs={"dtype": "bfloat16"})
    trainer = DPOTrainer(model=a.model, ref_model=None, args=cfg, train_dataset=train, processing_class=tok)
    if a.init_weights:
        from safetensors.torch import load_file
        state = {}
        for f in sorted(Path(a.init_weights).glob("*.safetensors")):
            state.update(load_file(str(f)))
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        assert not unexpected and all("lm_head" in k for k in missing), (missing[:5], unexpected[:5])
        print(f"[resume] policy initialised from {a.init_weights}; reference stays the SFT model", flush=True)
    trainer.train()
    trainer.save_model(str(Path(a.output_dir) / "final"))


if __name__ == "__main__":
    main()
