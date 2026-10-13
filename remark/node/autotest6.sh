#!/bin/bash
# Lock-and-test watcher. Two passes every 2 minutes:
# A. lock: for each validated local run (grpo_qwen/<run>/DONE) on outputs/crs/locks/test_allowlist.txt without a lock
# record, choose the best step (highest validation per-row R@20, ties to the earlier step; the curve also includes steps validated
# on other nodes under outputs/crs/val), write outputs/crs/locks/<run>.json (UTC time, host, curve, best step) and copy the
# checkpoint to grpo_qwen_selected/.
# B. test: for every lock record whose test output (outputs/crs/test/test_<run>_step<best>/summary.json) is missing, claim it
# (mkdir <run>.testing on shared storage, so one node tests it) and test the locked checkpoint once (reddit, books, music splits).
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; L=/tmp/work_local; G=$L/grpo_qwen; V=$L/val; T=$L/test
LOCK=$R/remark/outputs/crs/locks; SEL=$R/remark/outputs/crs/grpo_qwen_selected; OUT=$R/remark/outputs/crs/test; mkdir -p $T
while true; do
  for d in $G/*/; do
    run=$(basename $d); [ -f $d/DONE ] || continue; [ -f $LOCK/$run.json ] && continue
    grep -qxE -f $LOCK/test_allowlist.txt <<< "$run" || continue
    best=$(python3 - "$V" "$run" "$R" <<'PY'
import json, re, sys
from pathlib import Path
V, run = Path(sys.argv[1]), sys.argv[2]
curve = {}
for p in list((Path(sys.argv[3]) / "remark/outputs/crs/val").glob(f"grpo_qwen_{run}_step*/summary.json")) + list(V.glob(f"grpo_qwen_{run}_step*/summary.json")):
    m = re.search(r"_step(\d+)$", p.parent.name)
    if m: curve[int(m.group(1))] = json.load(open(p))["truncate_R@20"]
if curve:
    b = max(sorted(curve), key=lambda s: curve[s])
    print(b, json.dumps({str(k): curve[k] for k in sorted(curve)}))
PY
)
    [ -n "$best" ] || continue
    step=${best%% *}; curve=${best#* }; [ -d $d/checkpoint-$step ] || continue
    mkdir -p $SEL/$run; [ -f $SEL/$run/checkpoint-$step/config.json ] || { rsync -a $d/checkpoint-$step/ $SEL/$run/.checkpoint-$step.part/ && mv $SEL/$run/.checkpoint-$step.part $SEL/$run/checkpoint-$step; }
    python3 -c "import json,sys,datetime,socket; json.dump({'run':'$run','best_step':$step,'val_R@20':json.loads(sys.argv[1]),'locked_at_utc':datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),'host':socket.gethostname(),'rule':'best validation per-row R@20, ties to the earlier step; tested once'}, open('$LOCK/.$run.json.part','w'), indent=2)" "$curve" \
      && mv $LOCK/.$run.json.part $LOCK/$run.json && echo "[$(date -u +%H:%M)] locked $run @ $step"
  done
  for lk in $LOCK/*.json; do
    [ -f "$lk" ] || continue
    run=$(basename $lk .json); step=$(python3 -c "import json;print(json.load(open('$lk'))['best_step'])")
    name=test_${run}_step$step
    [ -f $OUT/$name/summary.json ] && continue
    [ -f $SEL/$run/checkpoint-$step/config.json ] || continue
    [ -d $LOCK/$run.testing ] && [ -n "$(find $LOCK/$run.testing -maxdepth 0 -mmin +120)" ] && rm -rf $LOCK/$run.testing  # stale claim
    mkdir $LOCK/$run.testing 2>/dev/null || continue
    extra=""; util=0.14
    case $run in reddit_books_*) extra="--data $R/remark/data/crs/reddit_books/sft_dataset --catalog $R/remark/data/crs/reddit_books/catalog.pkl";; music_*) extra="--data $R/remark/data/crs/music/sft_dataset --catalog $R/remark/data/crs/music/catalog.pkl";; esac
    case $run in 4B_*|reddit_books_4B_*) util=0.2;; 9B_*) util=0.3;; esac
    gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
    echo "$(hostname) $(date -u +%H:%M:%S) gpu $gpu" > $LOCK/$run.testing/by
    if VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=$gpu python3 $R/remark/scripts/crs_eval.py --split test --out $T \
         --gpu_util $util $extra --model $SEL/$run/checkpoint-$step --name $name > /tmp/autotest_$run.log 2>&1 && [ -f $T/$name/summary.json ]; then
      rsync -a $T/$name $OUT/ && [ -d $G/$run ] && touch $G/$run/TESTED; echo "[$(date -u +%H:%M)] tested $name"
    else
      echo "[$(date -u +%H:%M)] test failed $name (see /tmp/autotest_$run.log)"
    fi
    rm -rf $LOCK/$run.testing
  done
  sleep 120
done
