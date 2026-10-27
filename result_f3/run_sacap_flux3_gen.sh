#!/bin/bash
# SACap-1M で学習した FLUX.3 LoRA (checkpoint-2000) で examples_SACap_eval (12 枚, 1024x672) を生成。学習と同じ hard / cond_scale_factor 2
cd ~/Seg2Any
export HF_HOME=/export/ssd3/lan-y/huggingface; unset HF_HUB_OFFLINE
CK=/export/ssd3/lan-y/seg2any_ckpt/sacap_flux3/checkpoint-2000
OUT=result_f3/sacap_flux3/examples_SACap_eval
mkdir -p $OUT
COMMON="--seg_mask_path examples_SACap_eval --cond_scale_factor 2 --attention_mask_method hard"
CUDA_VISIBLE_DEVICES=0 .venv-flux3/bin/python infer_flux3.py $COMMON --pretrained_model_name_or_path black-forest-labs/flux-3-image-klein-base --lora_ckpt_path $CK --output_path $OUT/base_lora > $OUT/log_base_lora.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv-flux3/bin/python infer_flux3.py $COMMON --pretrained_model_name_or_path black-forest-labs/flux-3-image-klein --lora_ckpt_path $CK --output_path $OUT/distill_lora > $OUT/log_distill_lora.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 .venv-flux3/bin/python infer_flux3.py $COMMON --pretrained_model_name_or_path black-forest-labs/flux-3-image-klein-base --output_path $OUT/base_nolora > $OUT/log_base_nolora.txt 2>&1 &
wait
echo ALL_DONE
