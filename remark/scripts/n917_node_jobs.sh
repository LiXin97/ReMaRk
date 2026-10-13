#!/bin/bash
# node jobs (resumable; outputs node-local under /tmp/work_local/test, rsynced to remark/outputs/crs/test after each
# job; a job whose summary.json already exists on /tmp or shared storage is skipped, so a preempted node just reruns this script).
# Usage (on a node, in tmux):  bash remark/scripts/n917_node_jobs.sh <gpu> <job>...
# jobs: zs27B_reddit zs27B_books   (E5: Qwen3.5-27B post-trained zero-shot; needs /tmp/models/27B, 52 GB)
# memo_2B memo_4B memo_9B     (E8: base-model 3-shot memorization probe; needs /tmp/models/Qwen3.5-<size>-Base)
# Models missing from /tmp are staged from the Hub (27B) or from data/models (base models) first.
# Then on the CPU host:  env/bin/python remark/scripts/n917_e8_stratify.py ; env/bin/python remark/scripts/n917_e6_metrics.py
set -u
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; GPU=$1; shift
O=/tmp/work_local/test; mkdir -p $O /tmp/models
sync_out() { rsync -a $O/ $R/remark/outputs/crs/test/; }
done_already() { [ -f $O/$1/summary.json ] || [ -f $R/remark/outputs/crs/test/$1/summary.json ]; }
stage_27B() {
  [ -f /tmp/models/27B/.staged ] && return 0
  /tmp/qenv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-27B', local_dir='/tmp/models/27B.part', max_workers=16)" \
    && rm -rf /tmp/models/27B && mv /tmp/models/27B.part /tmp/models/27B && touch /tmp/models/27B/.staged
}
stage_base() {  # $1 = size
  local d=/tmp/models/Qwen3.5-$1-Base
  [ -f $d/.staged ] && return 0
  rm -rf $d.part && cp -r $R/data/models/Qwen3.5-$1-Base $d.part && rm -rf $d && mv $d.part $d && touch $d/.staged
}
export VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$GPU
cd $R
for job in "$@"; do
  case $job in
    zs27B_reddit|zs27B_books)
      ds=${job#zs27B_}; name=test_zeroshot_qwen35_27B; [ $ds = books ] && name=${name}_books
      done_already $name && { echo "skip $name"; continue; }
      stage_27B || { echo "stage 27B failed"; exit 1; }
      python3 remark/scripts/n917_zeroshot_vllm.py --model /tmp/models/27B --name $name --dataset $ds --out $O \
        --gpu_util 0.85 > /tmp/n917_$name.log 2>&1 && sync_out || echo "FAILED $name (see /tmp/n917_$name.log)";;
    memo_2B|memo_4B|memo_9B)
      size=${job#memo_}; name=memo_probe_$size
      done_already $name && { echo "skip $name"; continue; }
      stage_base $size || { echo "stage $size failed"; exit 1; }
      python3 remark/scripts/n917_e8_memo_probe.py --model /tmp/models/Qwen3.5-$size-Base --name $name --out $O \
        --gpu_util 0.85 > /tmp/n917_$name.log 2>&1 && sync_out || echo "FAILED $name (see /tmp/n917_$name.log)";;
    *) echo "unknown job $job";;
  esac
done
sync_out
echo NODE_JOBS_DONE
