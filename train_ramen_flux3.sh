#!/bin/bash
# Seg2Any (FLUX 3 Image Klein base) LoRA training on UEC-Ramen555 (500 train / 55 held out for eval).
# Usage: bash train_ramen_flux3.sh [extra --key=value overrides...]
#
# All settings live in config/seg2any_ramen_flux3.yaml; anything passed here overrides it,
# e.g. `bash train_ramen_flux3.sh --trainer.max_train_steps=20` for a smoke run.
# 中身は train_ramen_flux2.sh と同じ。違いは venv・スクリプト・config と、HF_HUB_OFFLINE を付けないことだけ。
set -eu

cd "$(dirname "$0")"

# GPU を絞るときは CUDA_VISIBLE_DEVICES と NUM_PROCESSES を両方指定する (例: CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NUM_PROCESSES=7)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/export/ssd3/lan-y/huggingface}  # FLUX.3 は ssd3 にキャッシュ済み (space0 は mmap ロードが遅い)
# diffusers dev 版は HF_HUB_OFFLINE=1 だと分割された重みの読み込みに失敗する (FLUX3_PORT.md §2)
unset HF_HUB_OFFLINE

# 8 ranks x default thread pool would each grab all 32 cores and thrash.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Return freed blocks to the driver instead of letting the caching allocator sit on them.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Refuse to start if the node cannot cover the staggered load, rather than OOM-killing it.
avail_gb=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
if [ "$avail_gb" -lt 40 ]; then
  echo "Only ${avail_gb}GB RAM available; need ~21GB for the staggered load. Aborting." >&2
  exit 1
fi
echo "MemAvailable: ${avail_gb}GB -- starting."

# FLUX.3 版は .venv-flux3 (Python 3.11 / torch 2.8.0+cu128 / diffusers main)。FLUX.1 / FLUX.2 用の venv とは別
.venv-flux3/bin/accelerate launch --config_file config/${ACCEL_CONFIG:-accelerate_8gpu_config.yaml} --num_processes ${NUM_PROCESSES:-8} \
  train_flux3.py ${CONFIG:-config/seg2any_ramen_flux3.yaml} "$@"
