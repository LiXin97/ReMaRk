#!/bin/bash
# SFT launcher with resume across interruptions. Every checkpoint (every 250 steps) is mirrored to
# outputs/crs/sft_mirror/<ds>_<size>/checkpoint-<global step>; on a restart the run continues from the latest mirrored
# checkpoint with the cosine learning rate evaluated at that step and no warm-up (weights only, fresh optimizer).
# Output layout is unchanged, so validation and selection over steps 500/1000/1500 work as usual.
# Usage: sft_launch3.sh <gpus> <port> <dataset> <size>
set -e
GPUS=$1; PORT=$2; DS=$3; SIZE=$4
R=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export CUDA_VISIBLE_DEVICES=$GPUS LD_PRELOAD=/tmp/libbindlocal.so NCCL_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET NCCL_NET=Socket NCCL_NET_PLUGIN=none
export HF_DATASETS_CACHE=/tmp/hf_datasets TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
D=/tmp/data/sft_$DS
if [ ! -f $D/.staged ]; then mkdir -p /tmp/data; rm -rf $D.part; cp -r $R/remark/data/crs/$DS/sft_dataset $D.part && touch $D.part/.staged && mv $D.part $D; fi
case $SIZE in 2B) BS=12; GA=8; EXTRA="";; 4B) BS=8; GA=12; EXTRA="--grad_ckpt";; 9B) BS=6; GA=16; EXTRA="--grad_ckpt";; esac
OUT=/tmp/work_local/sft/${DS}_$SIZE; MS=$R/remark/outputs/crs/sft_mirror/${DS}_$SIZE; TOTAL=1500
mkdir -p $OUT $MS

mirror() {  # copy any complete local checkpoint to shared storage under its global step
  for c in $OUT/checkpoint-* $OUT/_seg*/checkpoint-*; do
    [ -d "$c" ] || continue
    ls $c/*.safetensors >/dev/null 2>&1 && [ -f $c/config.json ] || continue
    seg=$(basename $(dirname $c)); off=0; case $seg in _seg*) off=${seg#_seg};; esac
    g=$((off + ${c##*-}))
    [ -d $MS/checkpoint-$g ] || { rsync -a $c/ $MS/.checkpoint-$g.part/ && mv $MS/.checkpoint-$g.part $MS/checkpoint-$g \
      && echo "[$(date -u +%H:%M)] mirrored SFT $DS $SIZE step $g"; }
    [ -d $OUT/checkpoint-$g ] || cp -r $MS/checkpoint-$g $OUT/checkpoint-$g  # the runner validates $OUT/checkpoint-<step>
  done
}

while true; do
  k=$(ls -d $MS/checkpoint-* 2>/dev/null | awk -F- '{print $NF}' | sort -n | tail -1); k=${k:-0}
  [ "$k" -ge $TOTAL ] && break
  RES=()
  if [ "$k" -gt 0 ]; then
    lr=$(python3 -c "import math; w=round(0.05*$TOTAL); print(5e-5*0.5*(1+math.cos(math.pi*($k-w)/($TOTAL-w))))")
    RES=(--init_weights $MS/checkpoint-$k --lr $lr --warmup_ratio 0)
    echo "[$(date -u +%H:%M)] SFT resume $DS $SIZE from $k lr $lr" | tee -a $MS/RESUMES
  fi
  SEG=$OUT/_seg$k; rm -rf $SEG
  (cd /tmp && /tmp/qenv/bin/accelerate launch --config_file $R/remark/configs/accel_zero2_4gpu_ga$GA.yaml \
    --main_process_ip 127.0.0.1 --main_process_port $PORT $R/remark/scripts/crs_train_sft.py \
    --model /tmp/models/Qwen3.5-$SIZE-Base --dataset_path $D --output_dir $SEG \
    --per_device_train_batch_size $BS --gradient_accumulation_steps $GA --max_steps $((TOTAL - k)) \
    --save_steps 250 --eval_steps 500 --save_only_model $EXTRA "${RES[@]}") &
  TP=$!
  while kill -0 $TP 2>/dev/null; do mirror; sleep 120; done
  mirror
  k2=$(ls -d $MS/checkpoint-* 2>/dev/null | awk -F- '{print $NF}' | sort -n | tail -1); k2=${k2:-0}
  [ "$k2" -gt "$k" ] || { echo "[$(date -u +%H:%M)] SFT $DS $SIZE made no progress (at $k2)"; exit 1; }
done
mirror
echo "SFT_DONE $DS $SIZE"
