#!/bin/bash
# 意味付き cond トークン版の学習 (UEC-Ramen555)。中身は ../train_ramen_flux3.sh と同じで、スクリプトと config だけ違う。
# 使い方: bash train_semcond.sh [--key=value で config を上書き]
#   スモーク: CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 ACCEL_CONFIG=accelerate_2gpu_config.yaml bash train_semcond.sh --trainer.max_train_steps=20 ...
set -eu
cd ~/Seg2Any  # data/ramen/*.txt や config/accelerate_*.yaml は Seg2Any 直下からの相対パス
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/export/ssd3/lan-y/huggingface}
unset HF_HUB_OFFLINE
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E="[10-25]exp_semcond"
.venv-flux3/bin/accelerate launch --config_file config/${ACCEL_CONFIG:-accelerate_8gpu_config.yaml} --num_processes ${NUM_PROCESSES:-8} \
  "$E/train_flux3_semcond.py" ${CONFIG:-$E/seg2any_ramen_flux3_semcond.yaml} "$@"
