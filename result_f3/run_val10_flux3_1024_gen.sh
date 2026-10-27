#!/bin/bash
# val の先頭 10 枚を 1024² で生成 (学習は 512² の checkpoint-3000)。mask は 512 の mask を最近傍で 2 倍にしたもの。
# csf2 = cond_scale_factor 2: cond (輪郭) は 512 で描いて VAE に通し、位置 id を 2 倍して 1024 の格子に合わせる (学習時の cond と同じ見え方)
# 評価は生成画像を 512 に縮めて測るので result_f3/val55_flux3 の値とそのまま比べられる
set -u
cd ~/Seg2Any
export HF_HOME=/export/ssd3/lan-y/huggingface
unset HF_HUB_OFFLINE
OUT=result_f3/val10_flux3_1024
IN=${IN:-examples_Ramen_val10_1024}  # val55 の先頭 10 枚
CK3=${CK3:-/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000}
mkdir -p $OUT
for d in real_512 flux1 flux2_base_cfg4_negnocond; do ln -sfn ../val55/$d $OUT/$d; done
for d in flux3_base_cfg35_50step flux3_base_cfg35_20step flux3_distilled; do ln -sfn ../val55_flux3/$d $OUT/${d}_512; done
t() { local gpu=$1 name=$2; shift 2; local s=$(date +%s); CUDA_VISIBLE_DEVICES=$gpu "$@" > $OUT/$name.log 2>&1; echo "$name rc=$? $(( $(date +%s) - s ))s" | tee -a $OUT/gen_times.txt; }
BASE=black-forest-labs/flux-3-image-klein-base
G=".venv-flux3/bin/python infer_flux3.py --seg_mask_path $IN --lora_ckpt_path $CK3"
t 0 flux3_base_cfg35_50step_1024      $G --output_path $OUT/flux3_base_cfg35_50step_1024 --pretrained_model_name_or_path $BASE --num_inference_steps 50 --guidance_scale 3.5 &
t 1 flux3_base_cfg35_50step_1024_csf2 $G --output_path $OUT/flux3_base_cfg35_50step_1024_csf2 --pretrained_model_name_or_path $BASE --num_inference_steps 50 --guidance_scale 3.5 --cond_scale_factor 2 &
t 3 flux3_distilled_1024              $G --output_path $OUT/flux3_distilled_1024 &
t 4 flux3_distilled_1024_csf2         $G --output_path $OUT/flux3_distilled_1024_csf2 --cond_scale_factor 2 &
wait
echo ALLDONE | tee -a $OUT/gen_times.txt
