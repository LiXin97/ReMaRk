#!/bin/bash
# ProtoQA work list for one 4-GPU slot; every stage idempotent. GRPO from the selected SFT (@360), checkpoints every 25,
# validation per-respondent recall@10 at every checkpoint, best (ties earlier) locked and tested once on test_scraped and
# test_crowdsourced; then the non-best checkpoints are deleted. Jobs: seeds 42 and 1234 for arms single (a), rb (b), union (d),
# 250 steps; 500-step runs at seed 3407 for (a), (b), (d) (run tag _ms500); then arm mean (c) at seeds 42 and 1234.
# Usage: g1_jobs.sh <gpus e.g. 0,1,2,3> <port>
set -x
GPUS=$1; PORT=$2; IFS=, read -ra GA <<< "$GPUS"
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; O=$R/remark/outputs/protoqa; L=$R/remark/scripts/n928_r9_launch.sh
stamp() { echo "[$(date -u +%H:%M)] $*"; }
evalset() {
  local pre=$1 dir=$2; shift 2; local i=0 pids=()
  for s in "$@"; do
    [ -f $O/eval/${pre}_step$s/summary.json ] && continue
    [ -f $dir/checkpoint-$s/config.json ] || continue
    bash $L eval ${GA[$i]} $dir/checkpoint-$s ${pre}_step$s validation > /tmp/g1_eval_${pre}_$s.log 2>&1 & pids+=($!)
    i=$(( (i + 1) % ${#GA[@]} )); sleep 15
    [ $i -eq 0 ] && { wait "${pids[@]}"; pids=(); }
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  return 0
}
best() {
  local pre=$1; shift
  python3 - "$O/eval" "$pre" "$@" <<'PY'
import json, sys
from pathlib import Path
E, pre, steps = Path(sys.argv[1]), sys.argv[2], [int(s) for s in sys.argv[3:]]
cur = {s: json.load(open(E / f"{pre}_step{s}/summary.json"))["resp_recall@10"] for s in steps if (E / f"{pre}_step{s}/summary.json").exists()}
if len(cur) == len(steps):
    print(max(sorted(cur), key=lambda s: cur[s]))
PY
}
[ -f /tmp/nltk_data/.r9_ok ] || bash $R/remark/scripts/n928_r9_node_setup.sh || { stamp "setup failed"; exit 1; }
SFT=$(python3 -c "import json;print(json.load(open('$O/locks/sft_2B.json'))['path'])"); stamp "SFT $SFT"
run_job() {  # $1 arm, $2 seed, $3 max steps
  local arm=$1 seed=$2 ms=$3 tag=2B_${1}_s${2}; [ $ms != 250 ] && tag=${tag}_ms$ms
  local D=$O/grpo/$tag pre=g1_$tag steps=$(seq -s " " 25 25 $ms)
  if [ ! -f $O/locks/$tag.json ]; then
    [ -f $D/checkpoint-$ms/config.json ] || RUN_TAG=$tag bash $L grpo $GPUS $PORT $arm $SFT --seed $seed --max_steps $ms > /tmp/g1_grpo_$tag.log 2>&1 || { stamp "GRPO $tag failed"; return 1; }
    evalset $pre $D $steps
    local b=$(best $pre $steps); [ -n "$b" ] || { stamp "$tag validation incomplete"; return 1; }
    python3 -c "import json,datetime;json.dump({'arm':'$arm','seed':$seed,'max_steps':$ms,'selected_step':$b,'path':'$D/checkpoint-$b','rule':'best validation resp_recall@10, ties earlier; tested once','at':datetime.datetime.utcnow().isoformat()},open('$O/locks/$tag.json','w'),indent=1)"
  fi
  local b=$(python3 -c "import json;print(json.load(open('$O/locks/$tag.json'))['selected_step'])")
  [ -f $O/eval/g1test_$tag/summary.json ] || bash $L eval ${GA[0]} $D/checkpoint-$b g1test_$tag test > /tmp/g1_test_$tag.log 2>&1 &
  [ -f $O/eval/g1testcs_$tag/summary.json ] || bash $L eval ${GA[1]} $D/checkpoint-$b g1testcs_$tag test_crowdsourced > /tmp/g1_testcs_$tag.log 2>&1 &
  wait
  if [ -f $O/eval/g1test_$tag/summary.json ] && [ -f $O/eval/g1testcs_$tag/summary.json ]; then
    for c in $D/checkpoint-*; do [ "${c##*-}" = "$b" ] || rm -rf -- "${c:?}"; done  # tested; keep the selected only
  fi
  stamp "$tag selected @$b and tested"
}
for seed in 42 1234; do for arm in single rb union; do run_job $arm $seed 250; done; done
for arm in single rb union; do run_job $arm 3407 500; done
for seed in 42 1234; do run_job mean $seed 250; done
stamp "G1_JOBS_DONE"
