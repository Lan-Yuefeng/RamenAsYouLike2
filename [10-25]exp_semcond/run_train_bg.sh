#!/bin/bash
# 本学習を落ちても latest から再開しながら回す (最大 3 回)
cd ~/Seg2Any/"[10-25]exp_semcond"
LOG=train_$(date +%Y%m%d_%H%M).log
bash train_semcond.sh ${FIRST_ARGS:-} > $LOG 2>&1; rc=$?
for i in 1 2 3; do
  [ $rc -eq 0 ] && break
  echo "rc=$rc -> resume $i" >> $LOG
  bash train_semcond.sh --resume_from_checkpoint=latest >> $LOG 2>&1; rc=$?
done
echo "FINISHED rc=$rc" >> $LOG
