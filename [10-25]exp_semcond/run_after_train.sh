#!/bin/bash
# 学習ログに FINISHED rc=0 が出たら、run_eval_val10.sh を走らせる (tmux semcond_eval から起動)
cd ~/Seg2Any/"[10-25]exp_semcond"
LOG=train_20261025_2101.log
until grep -q "FINISHED" $LOG; do sleep 60; done
if grep -q "FINISHED rc=0" $LOG; then
  bash run_eval_val10.sh > eval_val10.out 2>&1
else
  echo "training failed: $(grep FINISHED $LOG)" > eval_val10.out
fi
