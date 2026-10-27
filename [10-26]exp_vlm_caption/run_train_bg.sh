#!/bin/bash
# VLM caption 版の本学習。落ちても latest から再開する (最大 3 回)。tmux から起動する
cd ~/Seg2Any
D="[10-26]exp_vlm_caption"
LOG="$D/train_$(date +%Y%m%d_%H%M).log"
export CONFIG="$D/seg2any_ramen_flux3_vlmcap.yaml"
bash train_ramen_flux3.sh ${FIRST_ARGS:-} > "$LOG" 2>&1; rc=$?
for i in 1 2 3; do
  [ $rc -eq 0 ] && break
  echo "rc=$rc -> resume $i" >> "$LOG"
  bash train_ramen_flux3.sh --resume_from_checkpoint=latest >> "$LOG" 2>&1; rc=$?
done
echo "FINISHED rc=$rc" >> "$LOG"
