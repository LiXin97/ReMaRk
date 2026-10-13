#!/bin/bash
# Evaluate every locked/selected Reddit-Books checkpoint once on the frozen Reddit-Books-Late test set
# (unchanged crs_eval decoding). 4 queues on GPUs 0-3; outputs to outputs/crs/books_late/test/<name>.
# Usage on the node: bash c_eval.sh <gpu> <list file>   (list lines: <run> <step>)
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; SEL=$R/remark/outputs/crs/grpo_qwen_selected
D=$R/remark/data/crs/reddit_books_late; OUT=$R/remark/outputs/crs/books_late/test; T=/tmp/books_late_test; mkdir -p $OUT $T
GPU=$1
while read -r run step; do
  [ -z "$run" ] && continue
  name=late_${run}_step$step
  [ -f $OUT/$name/summary.json ] && continue
  util=0.3; case $run in *4B*) util=0.4;; esac
  echo "[$(date -u +%H:%M)] gpu $GPU eval $name"
  VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$GPU python3 $R/remark/scripts/crs_eval.py --split test \
    --data $D/sft_dataset --catalog $D/catalog.pkl --out $T --gpu_util $util --model $SEL/$run/checkpoint-$step \
    --name $name > /tmp/c_eval_$name.log 2>&1 && [ -f $T/$name/summary.json ] && rsync -a $T/$name $OUT/ \
    && echo "[$(date -u +%H:%M)] done $name" || echo "[$(date -u +%H:%M)] FAILED $name (see /tmp/c_eval_$name.log)"
done < $2
echo C_EVAL_QUEUE_DONE $GPU
