#!/bin/bash
# ProtoQA) launches on a GPU node (training env; run n928_r9_node_setup.sh first). Run inside tmux.
# sft  <gpus> <port> <size> [extra crs_train_sft.py args]        -> remark/outputs/protoqa/sft/qwen35_<size>
# grpo <gpus> <port> <arm> <sft_ckpt> [extra launcher args]      -> remark/outputs/protoqa/grpo/2B_<arm>_s3407
# arm: single (a) | rb (b) | mean (c) | union (d); RUN_TAG overrides the run directory name
# eval <gpu> <ckpt> <name> <split> [extra n928_r9_eval.py args]  -> remark/outputs/protoqa/eval/<name>
# SFT: 4 GPUs x 6 x GA 4 = 96 sequences per step (accel_zero2_4gpu_ga4.yaml); lr 5e-5 cosine, warmup 5% (crs_train_sft
# defaults); 540 steps ~ 6 epochs of 8.5k questions (the Reddit-v2 SFT is 6 epochs), model-only checkpoints every 90 steps.
# GRPO: crs_train_grpo_qwen.py's configuration (4 GPUs x 16 x GA 6 = 384 lists, 48 prompts x G 8), 250 steps, save 25.
set -e
MODE=$1; shift
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export LD_PRELOAD=/tmp/libbindlocal.so NCCL_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET NCCL_NET=Socket NCCL_NET_PLUGIN=none
export HF_DATASETS_CACHE=/tmp/hf_datasets TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
export NLTK_DATA=/tmp/nltk_data
test -f /tmp/nltk_data/.r9_ok || { echo "run remark/scripts/n928_r9_node_setup.sh first"; exit 1; }
cd /tmp
case $MODE in
  sft)
    GPUS=$1; PORT=$2; SIZE=$3; shift 3
    export CUDA_VISIBLE_DEVICES=$GPUS
    /tmp/qenv/bin/accelerate launch --config_file $R/remark/configs/accel_zero2_4gpu_ga4.yaml --main_process_ip 127.0.0.1 \
      --main_process_port $PORT $R/remark/scripts/crs_train_sft.py --model /tmp/models/Qwen3.5-$SIZE-Base \
      --dataset_path /tmp/data/protoqa/sft_dataset --output_dir $R/remark/outputs/protoqa/sft/qwen35_$SIZE \
      --per_device_train_batch_size 6 --gradient_accumulation_steps 4 --max_steps 540 --save_steps 90 --eval_steps 90 \
      --save_only_model "$@"
    ;;
  grpo)
    GPUS=$1; PORT=$2; ARM=$3; SFT=$4; shift 4
    export CUDA_VISIBLE_DEVICES=$GPUS
    case $ARM in single|rb|mean) TRAIN=/tmp/data/protoqa/grpo/train;; union) TRAIN=/tmp/data/protoqa/grpo_union/train;;
      *) echo "unknown arm $ARM"; exit 1;; esac
    # vLLM needs the base model's processor configs beside the (VLM-layout) SFT checkpoint
    /tmp/qenv/bin/python -c "import sys; from pathlib import Path; sys.path.insert(0, '$R/remark/scripts'); from crs_eval import prepare_qwen35; prepare_qwen35(Path('$SFT'))"
    /tmp/qenv/bin/accelerate launch --config_file $R/remark/configs/accel_zero2_4gpu_ga6.yaml --main_process_ip 127.0.0.1 \
      --main_process_port $PORT $R/remark/scripts/n928_r9_train_grpo.py --reward $ARM --train_path $TRAIN --sft_path $SFT \
      --output_dir $R/remark/outputs/protoqa/grpo/${RUN_TAG:-2B_${ARM}_s3407} "$@"
    ;;
  eval)
    GPU=$1; CK=$2; NAME=$3; SPLIT=$4; shift 4
    # vLLM must not inherit the training-only bind-local / NCCL settings above (its c10d init then times out; 2026-10-02)
    env -u LD_PRELOAD -u NCCL_SOCKET_IFNAME -u NCCL_SOCKET_FAMILY -u NCCL_NET -u NCCL_NET_PLUGIN \
    VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$GPU /tmp/qenv/bin/python $R/remark/scripts/n928_r9_eval.py \
      --model $CK --name $NAME --split $SPLIT --data /tmp/data/protoqa/sft_dataset "$@"
    ;;
  *) echo "mode: sft | grpo | eval"; exit 1;;
esac
