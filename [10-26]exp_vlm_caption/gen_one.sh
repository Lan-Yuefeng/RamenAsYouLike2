#!/bin/bash
# 1 条件を Base 版 (50 step, CFG 3.5) と蒸留版 (16 step) で続けて生成する
# 使い方: GPU=<n> ./gen_one.sh <name> <inputs> <old|new> <split名 (val55)>
cd "$(dirname "$0")"
export HF_HOME=/export/ssd3/lan-y/huggingface CUDA_VISIBLE_DEVICES=$GPU
G=../"[10-25]exp_ablation"/gen_ablation.py
NEW=/export/space0/lan-y/seg2any_ckpt/ramen_flux3_vlmcap/checkpoint-3000
OLD=/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000
[ "$3" = new ] && L=$NEW || L=$OLD
mkdir -p results_$4/logs results_distill_$4/logs
../.venv-flux3/bin/python "$G" --name $1 --inputs $2 --lora $L --out_root results_$4 > results_$4/logs/$1.log 2>&1
../.venv-flux3/bin/python "$G" --name $1 --inputs $2 --lora $L --out_root results_distill_$4 \
  --model black-forest-labs/flux-3-image-klein --steps 16 > results_distill_$4/logs/$1.log 2>&1
echo "$1 done" >> results_$4/logs/done.txt
