"""SFT of a Qwen3.5 base model on Rank-GRPO's Reddit-v2 SFT split (prompt -> 20 "Title (Year)" lines).

Follows third_party/Rank-GRPO/train_sft.py (lr 5e-5, cosine, warmup 5%, effective batch 384, completion-only
loss). Differences: explicit paths, --max_steps instead of 10 epochs (the released Llama SFT checkpoint is step
1500 = 6 epochs), max_length 2048 for the Qwen tokenizer, <|im_end|> as EOS. TRL keeps the checkpoint in the base model's
Qwen3_5ForConditionalGeneration layout, which vLLM loads directly.

Launch (qenv, node): accelerate launch --config_file remark/configs/<cfg>.yaml remark/scripts/crs_train_sft.py \
    --model /tmp/models/Qwen3.5-4B-Base --output_dir <mnt dir>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dataset_path", default="/tmp/data/sft_dataset")
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--per_device_train_batch_size", type=int, default=12)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--save_steps", type=int, default=250)
    ap.add_argument("--eval_steps", type=int, default=250)
    ap.add_argument("--optim", default="adamw_torch_fused")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--init_weights", default=None, help=" SFT resume: model weights from this model-only checkpoint "
                    "(fresh optimizer); pass the remaining --max_steps, the learning rate at the resume step as --lr and "
                    "--warmup_ratio 0, so the cosine schedule continues approximately")
    ap.add_argument("--save_only_model", action="store_true", help=" (27B/35B-A3B): no optimizer state in checkpoints")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--keep_eos", action="store_true", help=" Llama-3.x instruct: keep the native EOS (<|eot_id|>) and "
                    "generation_config stop ids; pad = <|finetune_right_pad_id|> (never the EOS)")
    a = ap.parse_args()
    train = load_from_disk(os.path.join(a.dataset_path, "train")).select_columns(["prompt", "completion"])
    val = load_from_disk(os.path.join(a.dataset_path, "validation")).select_columns(["prompt", "completion"])
    val = val.shuffle(seed=0).select(range(min(2000, len(val))))
    cfg = SFTConfig(
        output_dir=a.output_dir, max_steps=a.max_steps, learning_rate=a.lr, warmup_ratio=a.warmup_ratio,
        lr_scheduler_type="cosine", optim=a.optim, per_device_train_batch_size=a.per_device_train_batch_size,
        per_device_eval_batch_size=a.per_device_train_batch_size, gradient_accumulation_steps=a.gradient_accumulation_steps,
        max_length=a.max_length, bf16=True, gradient_checkpointing=a.grad_ckpt, eval_strategy="steps",
        eval_steps=a.eval_steps, save_strategy="steps", save_steps=a.save_steps, save_only_model=a.save_only_model, logging_steps=10,
        seed=a.seed, report_to="none", dataset_num_proc=16, run_name=Path(a.output_dir).name,
        model_init_kwargs={"dtype": "bfloat16"})
    # The base checkpoint ships a VLM processor without a chat template; train on the text tokenizer, and make
    # <|im_end|> the EOS so vLLM and TRL stop generation at the end of the assistant turn.
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.keep_eos:  # Llama-3.x instruct
        assert tok.eos_token == "<|eot_id|>", tok.eos_token
        if tok.pad_token is None:
            tok.pad_token = "<|finetune_right_pad_id|>"
        assert tok.pad_token_id != tok.eos_token_id
        trainer = SFTTrainer(model=a.model, args=cfg, train_dataset=train, eval_dataset=val, processing_class=tok)
    else:
        tok.eos_token = "<|im_end|>"
        trainer = SFTTrainer(model=a.model, args=cfg, train_dataset=train, eval_dataset=val, processing_class=tok)
        trainer.model.generation_config.eos_token_id = [tok.eos_token_id, tok.pad_token_id]
    if a.init_weights:
        from safetensors.torch import load_file
        state = {}
        for f in sorted(Path(a.init_weights).glob("*.safetensors")):
            state.update(load_file(str(f)))
        missing, unexpected = trainer.model.load_state_dict(state, strict=False)
        assert not unexpected and all("lm_head" in k for k in missing), (missing[:5], unexpected[:5])
        print(f"[resume] model initialised from {a.init_weights} ({len(state)} tensors)", flush=True)
    last = sorted(Path(a.output_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    trainer.train(resume_from_checkpoint=str(last[-1]) if last else None)
    trainer.save_model(os.path.join(a.output_dir, "final"))


if __name__ == "__main__":
    main()
