#!/bin/bash
# Seg2Any (FLUX 3 Image Klein base) LoRA training on UECFOODPIXCOMPLETE + 皿マスク (UECFOODPIXCOMPLETEwithPLATE_104, 9000 train / 1000 test).
# Usage: bash train_uecfoodpix_plate_flux3.sh [extra --key=value overrides...]
#
# 2026-10-24 は ssd3 が落ちていたので、venv・重み・データはすべて gp41 の ssd2 から読む (config/seg2any_uecfoodpix_plate_flux3.yaml)。
# ssd3 が戻ったら VENV=.venv-flux3 HF_HOME=/export/ssd3/lan-y/huggingface でも動く
set -eu

cd "$(dirname "$0")"

# GPU を絞るときは CUDA_VISIBLE_DEVICES と NUM_PROCESSES を両方指定する (例: CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NUM_PROCESSES=7)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}  # gp39 で PCIe 越し P2P がハングした対策。他のノードでも害はない
export HF_HOME=${HF_HOME:-/export/ssd2/lan-y/huggingface}  # space0 から ssd2 にコピー済み (space0 は mmap ロードが遅い)
# diffusers dev 版は分割された重みの読み込みで Hub に問い合わせるので OFFLINE にしない (FLUX3_PORT.md)
unset HF_HUB_OFFLINE
VENV=${VENV:-/export/ssd2/lan-y/venvs/Seg2Any/.venv-flux3}  # .venv-flux3 (ssd3) の作り直し。diffusers は 2026-10-15 の main (af4271c)

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

avail_gb=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
if [ "$avail_gb" -lt 40 ]; then
  echo "Only ${avail_gb}GB RAM available; need ~21GB for the staggered load. Aborting." >&2
  exit 1
fi
echo "MemAvailable: ${avail_gb}GB -- starting."

$VENV/bin/accelerate launch --config_file config/${ACCEL_CONFIG:-accelerate_8gpu_config.yaml} --num_processes ${NUM_PROCESSES:-8} \
  train_flux3.py ${CONFIG:-config/seg2any_uecfoodpix_plate_flux3.yaml} "$@"
