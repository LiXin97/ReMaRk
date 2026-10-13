#!/bin/bash
# Slot runner: work-stealing job queue for GRPO runs on 4 GPUs of one node (node-local outputs under /tmp/work_local,
# validated checkpoints and locks mirrored to shared storage).
# Queue file: $QFILE under /tmp/work_local (default queue6.txt); example grammar in configs/grpo_queue_p0c.txt.
# Each job line: "<dataset> <size> <reward> <adv_norm> [extra args for crs_train_grpo_qwen.py]". Run-name suffixes are derived
# from the extra args (_cap<N>, _lr<x>, _rb, _dyn, _G<k>, _cos, ...). The runner-only flag "--plateau <steps>" stops training once
# the best validation R@20 has not been exceeded for <steps> steps (validated grid checkpoints, contiguous from 250).
# Steps: (1) optionally wait for a busy tmux session on this slot and validate its run; (2) loop: claim the first unclaimed job
# (mkdir lock on shared storage), train it (up to 3 attempts, resuming from the mirrored checkpoint), validate every checkpoint,
# write DONE. SFT for reddit_books / music is run first via sft_launch3.sh when the SFT checkpoint is missing.
# Stage launch5.sh and sft_launch3.sh into /tmp/work_local before use and export REPO_ROOT=<package root>.
# Usage: runner7h.sh <gpus e.g. 0,1,2,3> <port> [<busy tmux session> <its run name under grpo_qwen/>]
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}; GPUS=$1; PORT=$2; BUSY=$3; BUSY_RUN=$4
Q=/tmp/work_local/${QFILE:-queue6.txt}; G=/tmp/work_local/grpo_qwen; V=/tmp/work_local/val; mkdir -p $G $V
MIR=$R/remark/outputs/crs/grpo_qwen_mirror
IFS=, read -ra GA <<< "$GPUS"

validate() {  # $1 = run name under grpo_qwen/, $2 = dataset
  local run=$1 ds=$2 extra="" pids=() i=0
  [ "$ds" != reddit ] && extra="--data $R/remark/data/crs/$ds/sft_dataset --catalog $R/remark/data/crs/$ds/catalog.pkl"
  for m in $MIR/$run/checkpoint-*; do [ -d "$m" ] || continue; [ -d $G/$run/$(basename $m) ] || cp -r $m $G/$run/; done 
  for ck in $(ls -d $G/$run/checkpoint-* 2>/dev/null | awk -F- '{print $NF" "$0}' | sort -n | cut -d" " -f2); do
    local name=grpo_qwen_${run}_step${ck##*-}
    [ -f $V/$name/summary.json ] && continue
    if [ -f $R/remark/outputs/crs/val/$name/summary.json ]; then cp -r $R/remark/outputs/crs/val/$name $V/; continue; fi  # reuse
    VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=${GA[$i]} python3 $R/remark/scripts/crs_eval.py --model $ck \
      --name $name --split validation --out $V $extra > /tmp/eval_$name.log 2>&1 &
    pids+=($!); i=$(( (i + 1) % ${#GA[@]} )); sleep 20  # stagger: simultaneous vLLM starts collide on ports
    if [ $i -eq 0 ]; then wait "${pids[@]}"; pids=(); rsync -a $V/ $R/remark/outputs/crs/val/ 2>/dev/null; fi
  done
  wait; rsync -a $V/ $R/remark/outputs/crs/val/ 2>/dev/null
}

plateau_hit() {  # $1 run, $2 patience (steps): exit 0 when the best validated step is >= patience behind the last one
  python3 - "$R/remark/outputs/crs/val" "$1" "$2" <<'PY'
import json, re, sys
from pathlib import Path
V, run, P = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
cur = {}
for p in V.glob(f"grpo_qwen_{run}_step*/summary.json"):
    m = re.search(r"_step(\d+)$", p.parent.name)
    if m and int(m.group(1)) % 250 == 0:
        cur[int(m.group(1))] = json.load(open(p))["truncate_R@20"]
last = 0
while last + 250 in cur:
    last += 250
if not last:
    sys.exit(1)
seen = {k: v for k, v in cur.items() if k <= last}
best = max(sorted(seen), key=lambda k: seen[k])
print(json.dumps({"best_step": best, "best_val": seen[best], "last_validated": last, "patience": P, "curve": seen}))
sys.exit(0 if last - best >= P else 1)
PY
}
plateau_watch() {  # $1 run, $2 patience; stops this run's trainer when the rule fires
  while sleep 120; do
    if plateau_hit $1 $2 > $G/$1/plateau.json; then
      touch $G/$1/PLATEAU; mkdir -p $MIR/$1; cp $G/$1/plateau.json $MIR/$1/PLATEAU
      echo "[$(date -u +%H:%M)] plateau: stopping $1 ($(cat $G/$1/plateau.json | head -c 120))"
      pkill -f "crs_train_grpo_qwen.py.*/$1[/ ]" ; pkill -f "crs_train_grpo_qwen.py.*/$1/_seg"; return 0
    fi
  done
}

if [ -n "$BUSY" ]; then
  while tmux has-session -t "$BUSY" 2>/dev/null && ! tail -c 300 /tmp/$BUSY.log | grep -q EXIT_RC=; do sleep 60; done
  if [ -n "$BUSY_RUN" ]; then
    ds=reddit; case $BUSY_RUN in redial_*) ds=redial;; inspired_*) ds=inspired;; esac
    validate $BUSY_RUN $ds && touch $G/$BUSY_RUN/DONE
  fi
fi

while true; do
  job=""; blocked=0
  while read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    set -- $line
    if [ "$1" = sft ]; then run=sft_$2_$3
    else
      ds=$1; size=$2; reward=$3; norm=$4
      tag=""; [ $ds != reddit ] && tag=${ds}_
      seed=$(echo "$line" | grep -oE -- "--seed [0-9]+" | awk "{print \$2}"); seed=${seed:-3407}; cap=$(echo "$line" | grep -oE -- "--p0_max_tokens [0-9]+" | awk "{print \$2}"); suf=""; [ -n "$cap" ] && suf=_cap$cap; echo "$line" | grep -q -- "--exact_items" && suf=${suf}_ex; ms=$(echo "$line" | grep -oE -- "--max_steps [0-9]+" | awk "{print \$2}"); [ -n "$ms" ] && [ "$ms" != 2000 ] && suf=${suf}_ms$ms; g=$(echo "$line" | grep -oE -- "--num_generations [0-9]+" | awk "{print \$2}"); [ -n "$g" ] && [ "$g" != 8 ] && suf=${suf}_G$g 
      lr=$(echo "$line" | grep -oE -- "--lr [0-9.e-]+" | awk "{print \$2}"); [ -n "$lr" ] && suf=${suf}_lr$lr; echo "$line" | grep -q -- "--rb" && suf=${suf}_rb; echo "$line" | grep -q -- "--dyn_sample" && suf=${suf}_dyn; sm=$(echo "$line" | grep -oE -- "--shaping_mult [0-9.]+" | awk "{print \$2}"); [ -n "$sm" ] && suf=${suf}_shape$(echo "$sm" | awk "{printf \"%d\", \$1 * 100 + 0.5}"); echo "$line" | grep -q -- "--lr_scheduler cosine" && suf=${suf}_cos  # B (_shape);  d (_cos)
      run=${tag}${size}_${reward}_${norm}_s${seed}_p0c${suf}
      # Reddit-Books GRPO starts only after its own SFT checkpoint is selected
      if { [ $ds = reddit_books ] || [ $ds = music ]; } && [ ! -f $R/remark/outputs/crs/sft/qwen35_${ds}_${size}/selected.json ]; then
        [ -d $G/.claim_$run ] || blocked=1; continue
      fi
    fi
    # never retrain a run that is already locked for test; a validation-only run is also skipped once fully validated on shared storage
    [ -f $R/remark/outputs/crs/locks/$run.json ] && continue
    if [ "$1" != sft ] && ! grep -qxE -f $R/remark/outputs/crs/locks/test_allowlist.txt <<< "$run"; then nv=$(ls -d $R/remark/outputs/crs/val/grpo_qwen_${run}_step* 2>/dev/null | wc -l); ms=$(echo "$line" | grep -oE -- "--max_steps [0-9]+" | awk "{print \$2}"); [ "$nv" -ge $(( ${ms:-2000} / 250 )) ] && continue; fi
    mkdir -p $G
    if mkdir $G/.claim_$run 2>/dev/null; then job=$line; break; fi
  done < $Q
  if [ -z "$job" ]; then
    [ $blocked = 1 ] && { sleep 300; continue; }
    echo "[$(date -u +%H:%M)] queue empty"; exit 0
  fi
  set -- $job
  if [ "$1" = sft ]; then
    ds=$2; size=$3; echo "[$(date -u +%H:%M)] slot $GPUS: $run"
    bash /tmp/work_local/sft_launch3.sh $GPUS $PORT $ds $size > /tmp/$run.log 2>&1 || { echo "[$(date -u +%H:%M)] sft failed: $run"; continue; }
    S=/tmp/work_local/sft/${ds}_${size}; i=0
    for step in 500 1000 1500; do
      name=sft_${ds}_${size}_step$step
      [ -f $V/$name/summary.json ] || VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=${GA[$i]} python3 \
        $R/remark/scripts/crs_eval.py --model $S/checkpoint-$step --name $name --split validation --out $V \
        --data $R/remark/data/crs/$ds/sft_dataset --catalog $R/remark/data/crs/$ds/catalog.pkl > /tmp/eval_$name.log 2>&1 &
      i=$((i + 1))
    done
    wait
    python3 - "$ds" "$size" "$S" "$V" "$R" <<'PY'
import json, shutil, sys
from pathlib import Path
ds, size, S, V, R = sys.argv[1:]
res = {s: json.load(open(f"{V}/sft_{ds}_{size}_step{s}/summary.json"))["truncate_R@20"] for s in (500, 1000, 1500)}
best = max(res, key=res.get)
dst = Path(R) / f"remark/outputs/crs/sft/qwen35_{ds}_{size}"
src = Path(S) / f"checkpoint-{best}"
shutil.copytree(src, dst / f"checkpoint-{best}", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("global_step*", "rng_state*", "*.pt", "zero_to_*", "latest"))
json.dump({"val_R@20": res, "selected_step": best, "path": str(dst / f"checkpoint-{best}")},
          open(dst / "selected.json", "w"), indent=2)
print("selected", ds, size, best, res)
PY
    rsync -a $V/ $R/remark/outputs/crs/val/ 2>/dev/null
    continue
  fi
  ds=$1; size=$2; reward=$3; norm=$4; shift 4
  PAT=$(echo "$@" | grep -oE -- "--plateau [0-9]+" | awk '{print $2}')  # (runner-only)
  set -- $(echo "$@" | sed -E 's/--plateau [0-9]+//')
  echo "[$(date -u +%H:%M)] slot $GPUS: $run"
  sft_key=$size; case $ds in reddit|redditthr) ;; reddit_books|music) sft_key=${ds}_$size;; *) sft_key=2B;; esac
  SFT=$(python3 -c "import json;print(json.load(open('$R/remark/outputs/crs/sft/qwen35_$sft_key/selected.json'))['path'])")
  if [ $ds = redial ] || [ $ds = inspired ] && [ ! -f $V/${ds}_sft${size}_step0/summary.json ]; then
    VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=${GA[0]} python3 $R/remark/scripts/crs_eval.py --model $SFT \
      --name ${ds}_sft${sft_size}_step0 --split validation --out $V --data $R/remark/data/crs/$ds/sft_dataset \
      --catalog $R/remark/data/crs/$ds/catalog.pkl > /tmp/eval_${ds}_sft_step0.log 2>&1
  fi
  mkdir -p $G/$run; [ -n "$PAT" ] && [ -f $MIR/$run/PLATEAU ] && { touch $G/$run/PLATEAU; touch $G/$run/DONE_TRAIN; }
  PW=""; [ -n "$PAT" ] && [ ! -f $G/$run/DONE_TRAIN ] && { plateau_watch $run $PAT & PW=$!; }
  for attempt in 1 2 3; do [ -f $G/$run/DONE_TRAIN ] && break; [ $attempt -gt 1 ] && { echo "[$(date -u +%H:%M)] retry $attempt for $run in 300 s"; sleep 300; }
  if [ ! -f $G/$run/DONE_TRAIN ]; then
    want=$(echo "$@" | grep -oE -- "--max_steps [0-9]+" | awk '{print $2}'); want=${want:-2000}
    # resume: the latest checkpoint mirrored to shared storage survives a node reclaim
    k=$(ls -d $MIR/$run/checkpoint-* 2>/dev/null | awk -F- '{print $NF}' | sort -n | tail -1)
    mkdir -p $G/$run
    if [ -n "$k" ] && [ ! -d $G/$run/checkpoint-$k ]; then cp -r $MIR/$run/checkpoint-$k $G/$run/; fi
    if [ -n "$k" ] && [ "$k" -ge "$want" ]; then touch $G/$run/DONE_TRAIN
    elif [ -n "$k" ]; then
      echo "[$(date -u +%H:%M)] resume $run from step $k (fresh optimizer, KL reference = SFT, seed $((seed + k)))" | tee -a $G/$run/RESUMES
      cosarg=""; echo "$@" | grep -q -- "--lr_scheduler cosine" && cosarg="--lr_sched_offset $k --lr_sched_total $want"
      OUTDIR=$G/$run/_seg$k RESUME_ARGS="--init_weights $G/$run/checkpoint-$k --max_steps $((want - k)) --seed $((seed + k)) $cosarg" \
        DATASET=$ds SEED=$seed RUNSUF=$suf bash /tmp/work_local/launch5.sh $GPUS $PORT $size $reward $norm $SFT "$@" > /tmp/grpo_$run.log 2>&1
      rc=$?
      for c in $G/$run/_seg$k/checkpoint-*; do [ -d "$c" ] || continue; j=${c##*-}; mv $c $G/$run/checkpoint-$((k + j)); done
      last=$(ls -d $G/$run/checkpoint-* 2>/dev/null | awk -F- '{print $NF}' | sort -n | tail -1)
      if [ $rc -eq 0 ] || [ "$last" = "$want" ] || [ -f $G/$run/PLATEAU ]; then touch $G/$run/DONE_TRAIN
      else echo "[$(date -u +%H:%M)] train failed (rc $rc, last checkpoint ${last:-none}): $run"; fi
    else
      DATASET=$ds SEED=$seed RUNSUF=$suf bash /tmp/work_local/launch5.sh $GPUS $PORT $size $reward $norm $SFT "$@" > /tmp/grpo_$run.log 2>&1
      rc=$?
      last=$(ls -d $G/$run/checkpoint-* 2>/dev/null | awk -F- '{print $NF}' | sort -n | tail -1)
      if [ $rc -eq 0 ] || [ "$last" = "$want" ] || [ -f $G/$run/PLATEAU ]; then touch $G/$run/DONE_TRAIN
      else echo "[$(date -u +%H:%M)] train failed (rc $rc, last checkpoint ${last:-none}): $run"; fi
    fi
  fi
  done
  [ -n "$PW" ] && kill $PW 2>/dev/null
  [ -f $G/$run/DONE_TRAIN ] || continue
  validate $run $ds && touch $G/$run/DONE
  rsync -a $V/ $R/remark/outputs/crs/val/ 2>/dev/null
done
