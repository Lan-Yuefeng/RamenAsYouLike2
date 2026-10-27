#!/bin/bash
# val 55 枚の評価用生成 (scripts/flux2_eval_val55.py で評価する)。各条件の所要時間も記録する
set -u
cd ~/Seg2Any
export HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1
OUT=result_f2/val55
IN=examples_Ramen_val55
CK2=ckpt/ramen_flux2/checkpoint-3000
mkdir -p $OUT
t() { local name=$1; shift; local s=$(date +%s); "$@" > $OUT/$name.log 2>&1; echo "$name rc=$? $(( $(date +%s) - s ))s" | tee -a $OUT/gen_times.txt; }
t flux2_base_cfg4      .venv-flux2/bin/python infer_flux2.py --seg_mask_path $IN --lora_ckpt_path $CK2 --output_path $OUT/flux2_base_cfg4 \
    --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B --num_inference_steps 20 --guidance_scale 4.0 --cond_scale_factor 1
t flux2_base_cfg4_negnocond .venv-flux2/bin/python infer_flux2.py --seg_mask_path $IN --lora_ckpt_path $CK2 --output_path $OUT/flux2_base_cfg4_negnocond \
    --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B --num_inference_steps 20 --guidance_scale 4.0 --cond_scale_factor 1 --negative_no_cond
t flux2_distilled      .venv-flux2/bin/python infer_flux2.py --seg_mask_path $IN --lora_ckpt_path $CK2 --output_path $OUT/flux2_distilled \
    --cond_scale_factor 1
t flux1                .venv/bin/python infer.py --pretrained_model_name_or_path /export/ssd3/lan-y/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 --lora_ckpt_path ckpt/ramen/checkpoint-3000 \
    --seg_mask_path $IN --output_path $OUT/flux1 --cond_scale_factor 1
echo ALLDONE | tee -a $OUT/gen_times.txt
