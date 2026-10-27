#!/bin/bash
# 拡張器の学習 → val10 での生成 (zero-shot / fine-tune × name, short, detail)。tmux から起動する
# GPU は GPU=<番号> で指定 (Seg2Any 学習との相乗りを想定、約 15GB)
cd "$(dirname "$0")"
export HF_HOME=/export/ssd2/lan-y/huggingface CUDA_VISIBLE_DEVICES=${GPU:-5}
PY=../.venv-flux3/bin/python
OUT=/export/space0/lan-y/seg2any_ckpt/expander_qwen3_4b
mkdir -p logs expanded
$PY expander.py train --split ../data/ramen/train.txt --out $OUT > logs/expander_train.log 2>&1 || exit 1
for lv in name short detail; do
  $PY expander.py gen --split val10.txt --level $lv --out expanded/val10_${lv}_zeroshot > logs/gen_${lv}_zs.log 2>&1
  $PY expander.py gen --split val10.txt --level $lv --lora $OUT/epoch2 --out expanded/val10_${lv}_ft > logs/gen_${lv}_ft.log 2>&1
done
echo EXPANDER_DONE >> logs/expander_train.log
