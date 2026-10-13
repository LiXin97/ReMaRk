#!/bin/bash
# gradient-sketch run, one shard per GPU on a GPU node (shard = GPU index 0-3); resumable (samples appended, vectors per request).
K=$1
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
S=$R/remark/scripts/n931_g2_gradsketch.py
export CUDA_VISIBLE_DEVICES=$K VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_DATASETS_CACHE=/tmp/hf_datasets TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
cd /tmp
/tmp/qenv/bin/python $S sample --shard $K --nshards 4 && /tmp/qenv/bin/python $S grad --shard $K --nshards 4
echo G2_SHARD_EXIT $K $?
