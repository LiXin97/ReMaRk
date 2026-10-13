#!/bin/bash
# Memorization probe (3 base models) on one GPU of a busy node (gpu_util 0.2), outputs synced to outputs/crs/test
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; GPU=$1; T=/tmp/work_local/test; mkdir -p $T
for job in "reddit_books 2B books" "reddit_books 4B books" "music 2B music"; do
  set -- $job; name=memo_probe_$3_$2
  [ -f $R/remark/outputs/crs/test/$name/summary.json ] && continue
  VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$GPU python3 $R/remark/scripts/n935_a_memo_probe.py --dataset $1 \
    --model /tmp/models/Qwen3.5-$2-Base --name $name --out $T --gpu_util 0.2 > /tmp/$name.log 2>&1 \
    && rsync -a $T/$name $R/remark/outputs/crs/test/ && echo "[$(date -u +%H:%M)] done $name" || echo "[$(date -u +%H:%M)] FAILED $name"
done
echo PROBES_DONE
