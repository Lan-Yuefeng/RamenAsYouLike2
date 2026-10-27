#!/bin/bash
# val10 を蒸留版 (flux-3-image-klein、16 step、CFG なし) で生成。LoRA は Base 版で学習したものをそのまま載せる
cd "$(dirname "$0")"
export HF_HOME=/export/ssd3/lan-y/huggingface
G=../"[10-25]exp_ablation"/gen_ablation.py
M=black-forest-labs/flux-3-image-klein
NEW=/export/space0/lan-y/seg2any_ckpt/ramen_flux3_vlmcap/checkpoint-3000
OLD=/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000
R=results_distill; mkdir -p $R/logs
run() { CUDA_VISIBLE_DEVICES=$1 ../.venv-flux3/bin/python "$G" --model $M --steps 16 --out_root $R --name $2 --inputs $3 --lora $4 > $R/logs/$2.log 2>&1; }
run 0 hand_full ../"[10-25]exp_ablation"/inputs_val10 $OLD &
run 1 hand_short_raw inputs/val10_short_raw $OLD &
run 2 vlm_oracle inputs/val10_oracle $NEW &
run 3 vlm_short_raw inputs/val10_short_raw $NEW &
run 4 vlm_short_zs inputs/val10_short_zs $NEW &
run 5 vlm_short_ft inputs/val10_short_ft $NEW &
wait
echo GEN_DONE > $R/logs/done.txt
