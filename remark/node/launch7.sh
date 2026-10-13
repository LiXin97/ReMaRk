#!/bin/bash
# launch7 (10-08, req2x): launch6 + accelerate config chosen from --gradient_accumulation_steps N in the args (default 6)
# launch6 = launch5 + P0TAG / RB table (see below)
# NODE-LOCAL variant (2026-09-28, shared-disk quota): outputs under /tmp/work_local. Launch one Qwen3.5 Rank-GRPO run with P0c stabilization (P0 + malformed lines as zero-reward positions) on 4 GPUs of a formal node (qenv).
# Output dirs end in _p0c. Adds reward singlethr (single reward, one random responder per thread).
# Usage: crs_grpo_qwen_launch.sh <gpus> <port> <size> <reward> <adv_norm> <sft_ckpt_dir> [extra args]
set -e
GPUS=$1; PORT=$2; SIZE=$3; REWARD=$4; NORM=$5; SFT=$6; shift 6
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CUDA_VISIBLE_DEVICES=$GPUS LD_PRELOAD=/tmp/libbindlocal.so NCCL_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET NCCL_NET=Socket NCCL_NET_PLUGIN=none
export HF_DATASETS_CACHE=/tmp/hf_datasets TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
# reward: single (Rank-GRPO), consensus (plug-in weights), shrink<alpha> e.g. shrink2 (neighbour-prior shrinkage), union
# (any-responder binary), consensusrw / shrink<alpha>rw (threads sampled ∝ n); all but single use the consensus reward fn
# DATASET=reddit (default) | redial | inspired (; single-recommender datasets, n = 1 so consensus = single)
DATASET=${DATASET:-reddit}
if [ $DATASET = reddit ]; then
  CATALOG=$R/remark/third_party/Rank-GRPO/reddit_catalog/gt_catalog.pkl; TAG=""
  case $REWARD in
    single) SRC=$R/data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train; RFUNC=single;;
    consensus) SRC=$R/remark/data/crs/grpo_consensus/train; RFUNC=consensus;;
    shrink*rw) A=${REWARD#shrink}; SRC=$R/remark/data/crs/grpo_consensus_shrinkx_k200_a${A%rw}_rw/train; RFUNC=consensus;;
    shrink*) SRC=$R/remark/data/crs/grpo_consensus_shrinkx_k200_a${REWARD#shrink}/train; RFUNC=consensus;;
    union) SRC=$R/remark/data/crs/grpo_union/train; RFUNC=consensus;;
    vote025) SRC=$R/remark/data/crs/grpo_vote025/train; RFUNC=consensus;;  # 1 + 0.25*1{c>=2}
    consensusrw) SRC=$R/remark/data/crs/grpo_consensus_rw/train; RFUNC=consensus;;
    singlethr) SRC=$R/remark/data/crs/redditthr/grpo/train; RFUNC=single;;
    meanm*rw) M=${REWARD#meanm}; SRC=$R/remark/data/crs/grpo_mean_m${M%rw}_rw/train; RFUNC=consensus;;
    *) echo "unknown reward $REWARD"; exit 1;;
  esac
else
  D=$R/remark/data/crs/$DATASET; CATALOG=$D/catalog.pkl; TAG=${DATASET}_
  case $REWARD in
    single) SRC=$D/grpo/train; RFUNC=single;;
    consensus) SRC=$D/grpo_consensus/train; RFUNC=consensus;;
    union) SRC=$D/grpo_union/train; RFUNC=consensus;;
    consensusrw) SRC=$D/grpo_consensus_rw/train; RFUNC=consensus;;
    shrink*) SRC=$D/grpo_shrink_k200_a${REWARD#shrink}/train; RFUNC=consensus;;
    *) echo "unknown reward $REWARD for $DATASET"; exit 1;;
  esac
fi
# datasets' multiprocess filter cannot memory-map arrow files on the /mnt fuse mount: stage the split to /tmp
TRAIN=/tmp/data/grpo_${TAG}${REWARD}_$(basename $(dirname $SRC))
if [ ! -f $TRAIN/.staged ]; then mkdir -p /tmp/data; rm -rf $TRAIN.part; cp -r $SRC $TRAIN.part && touch $TRAIN.part/.staged && mv $TRAIN.part $TRAIN || true; fi
# launch6 (10-07): P0TAG (default _p0c; "" = no P0/P0c, E6); singlethr + --rb takes the RB table from the full per-responder split (E3)
P0TAG=${P0TAG-_p0c}; P0ARGS="--p0 --p0c"; [ -z "$P0TAG" ] && P0ARGS=""
RBT=""
if [ $DATASET = reddit ] && [ $REWARD = singlethr ] && echo "$@" | grep -q -- "--rb"; then
  RS=$R/data/raw/reddit_v2/processed_datasets/grpo/grpo_dataset/train; RT=/tmp/data/grpo_single_grpo_dataset
  if [ ! -f $RT/.staged ]; then mkdir -p /tmp/data; rm -rf $RT.part; cp -r $RS $RT.part && touch $RT.part/.staged && mv $RT.part $RT || true; fi
  RBT="--rb_table_path $RT"
fi
# vLLM needs the base model's processor configs beside the (VLM-layout) SFT checkpoint
/tmp/qenv/bin/python -c "import sys; from pathlib import Path; sys.path.insert(0, '$R/remark/scripts'); from crs_eval import prepare_qwen35; prepare_qwen35(Path('$SFT'))"
GA=$(echo "$@" | grep -oE -- "--gradient_accumulation_steps [0-9]+" | awk '{print $2}')  # launch7
cd /tmp
/tmp/qenv/bin/accelerate launch --config_file $R/remark/configs/accel_zero2_4gpu_ga${GA:-6}.yaml --main_process_ip 127.0.0.1 \
  --main_process_port $PORT $R/remark/scripts/crs_train_grpo_qwen.py --reward $RFUNC --adv_norm $NORM \
  --train_path $TRAIN --catalog_path $CATALOG --sft_path $SFT \
  --output_dir ${OUTDIR:-/tmp/work_local/grpo_qwen/${TAG}${SIZE}_${REWARD}_${NORM}_s${SEED:-3407}${P0TAG}${RUNSUF}} $P0ARGS "$@" $RBT ${RESUME_ARGS}
