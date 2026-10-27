#!/bin/bash
# val 55 枚の FLUX.3 評価用生成 (3 条件を GPU 0,1,2 で並列)。FLUX.1 / FLUX.2 の生成物は result_f2/val55 のものをリンクして使う。
# 評価: .venv-flux3/bin/python flux2_eval_val55.py --result_root result_f3/val55_flux3 --conditions ...
set -u
cd ~/Seg2Any
export HF_HOME=/export/ssd3/lan-y/huggingface
unset HF_HUB_OFFLINE
OUT=result_f3/val55_flux3
IN=examples_Ramen_val55
CK3=${CK3:-/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000}
mkdir -p $OUT
for d in flux1 flux2_base_cfg4_negnocond flux2_distilled real_512; do ln -sfn ../val55/$d $OUT/$d; done
t() { local gpu=$1 name=$2; shift 2; local s=$(date +%s); CUDA_VISIBLE_DEVICES=$gpu "$@" > $OUT/$name.log 2>&1; echo "$name rc=$? $(( $(date +%s) - s ))s" | tee -a $OUT/gen_times.txt; }
BASE=black-forest-labs/flux-3-image-klein-base
t 0 flux3_base_cfg35_50step .venv-flux3/bin/python infer_flux3.py --seg_mask_path $IN --lora_ckpt_path $CK3 --output_path $OUT/flux3_base_cfg35_50step \
    --pretrained_model_name_or_path $BASE --num_inference_steps 50 --guidance_scale 3.5 &
t 1 flux3_base_cfg35_20step .venv-flux3/bin/python infer_flux3.py --seg_mask_path $IN --lora_ckpt_path $CK3 --output_path $OUT/flux3_base_cfg35_20step \
    --pretrained_model_name_or_path $BASE --num_inference_steps 20 --guidance_scale 3.5 &
t 2 flux3_distilled .venv-flux3/bin/python infer_flux3.py --seg_mask_path $IN --lora_ckpt_path $CK3 --output_path $OUT/flux3_distilled &
wait
echo ALLDONE | tee -a $OUT/gen_times.txt
