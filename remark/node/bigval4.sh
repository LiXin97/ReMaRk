#!/bin/bash
# Validator: every mirrored grid checkpoint (step a multiple of GRID=250) under outputs/crs/grpo_qwen_mirror/ without a summary
# is validated (crs_eval.py, validation split; Reddit-Books and music runs with their data and catalog) and the summary written to
# outputs/crs/val/grpo_qwen_<run>_step<k>/. Runners and lock watchers reuse existing summaries, so nothing is validated twice.
# Big runs first. Failed evals release their claim and retry up to 3 times. GPUs holding memory are skipped.
# Usage: bigval4.sh <comma-separated GPUs>
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; M=$R/remark/outputs/crs/grpo_qwen_mirror; VAL=$R/remark/outputs/crs/val
L=/tmp/work_local/bigval; mkdir -p $L; IFS=, read -ra GA <<< "$1"; declare -A busy; GRID=${GRID:-250}
while true; do
  for g in "${GA[@]}"; do p=${busy[$g]}; [ -n "$p" ] && ! kill -0 $p 2>/dev/null && busy[$g]=""; done
  for c in $M/27B_*/checkpoint-* $M/35BA3B_*/checkpoint-* $(ls -d $M/*/checkpoint-* 2>/dev/null | grep -vE "/(27B|35BA3B)_"); do
    [ -d "$c" ] && [ -f $c/config.json ] || continue
    run=$(basename $(dirname $c)); k=${c##*-}; name=grpo_qwen_${run}_step$k
    [ $((k % GRID)) -eq 0 ] || continue
    [ -f $VAL/$name/summary.json ] && continue
    [ -f $L/$name.claim ] && [ -z "$(find $L/$name.claim -mmin +120)" ] && continue  # live claim (stale after 2 h)
    [ "$(cat $L/$name.fails 2>/dev/null || echo 0)" -ge 3 ] && continue
    free=""
    for g in "${GA[@]}"; do
      [ -z "${busy[$g]}" ] || continue
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [ "${used:-99999}" -lt 2000 ] && { free=$g; break; }
    done
    [ -z "$free" ] && break
    extra=""; util=0.85
    case $run in reddit_books_*) extra="--data $R/remark/data/crs/reddit_books/sft_dataset --catalog $R/remark/data/crs/reddit_books/catalog.pkl";; music_*) extra="--data $R/remark/data/crs/music/sft_dataset --catalog $R/remark/data/crs/music/catalog.pkl";; esac
    # util 0.85 = crs_eval's default, as in the runners' own validations
    touch $L/$name.claim
    ( if VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$free python3 $R/remark/scripts/crs_eval.py --model $c --name $name \
           --split validation --out $L --gpu_util $util $extra > /tmp/bigval_$name.log 2>&1 && [ -f $L/$name/summary.json ]; then
        rsync -a $L/$name $VAL/ && echo "[$(date -u +%H:%M)] validated $name"
      else
        n=$(( $(cat $L/$name.fails 2>/dev/null || echo 0) + 1 )); echo $n > $L/$name.fails
        echo "[$(date -u +%H:%M)] FAILED $name (attempt $n; /tmp/bigval_$name.log)"
      fi
      rm -f $L/$name.claim ) &
    busy[$free]=$!; sleep 30
  done
  sleep 120
done
