#!/bin/bash
# 拡張文ができたものから val55 の生成を始める
cd "$(dirname "$0")"
PY=../.venv-flux3/bin/python
wait_for() { until grep -q "parse failures" $1 2>/dev/null; do sleep 30; done; }
( wait_for logs/gen55_zs.log; $PY make_inputs.py --captions expanded/val55_short_zeroshot --split val55.txt --out inputs/val55_short_zs
  GPU=7 ./gen_one.sh vlm_short_zs inputs/val55_short_zs new val55 ) &
( wait_for logs/gen55_ft.log; $PY make_inputs.py --captions expanded/val55_short_ft --split val55.txt --out inputs/val55_short_ft
  GPU=2 ./gen_one.sh vlm_short_ft inputs/val55_short_ft new val55 ) &
( wait_for logs/gen55_ftv2.log; $PY make_inputs.py --captions expanded/val55_short_ftv2 --split val55.txt --out inputs/val55_short_ftv2
  GPU=0 ./gen_one.sh vlm_short_ftv2 inputs/val55_short_ftv2 new val55 ) &
wait
echo CHAIN_DONE >> results_val55/logs/done.txt
