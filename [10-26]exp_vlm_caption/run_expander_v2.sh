#!/bin/bash
# 拡張器 v2 (領域だけ学習、全体 caption は要約) の学習と、val55 の拡張文の生成 (short 段階)
cd "$(dirname "$0")"
export HF_HOME=/export/ssd2/lan-y/huggingface
PY=../.venv-flux3/bin/python
V1=/export/space0/lan-y/seg2any_ckpt/expander_qwen3_4b
V2=/export/space0/lan-y/seg2any_ckpt/expander_qwen3_4b_regonly
mkdir -p logs expanded
CUDA_VISIBLE_DEVICES=1 $PY expander.py gen --split val55.txt --level short --out expanded/val55_short_zeroshot > logs/gen55_zs.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY expander.py gen --split val55.txt --level short --lora $V1/epoch2 --out expanded/val55_short_ft > logs/gen55_ft.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY expander.py train --regions_only --split ../data/ramen/train.txt --out $V2 > logs/expander_v2_train.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=0 $PY expander.py gen --regions_only --split val55.txt --level short --lora $V2/epoch2 \
       --out expanded/val55_short_ftv2 > logs/gen55_ftv2.log 2>&1
wait
echo V2_DONE >> logs/expander_v2_train.log
