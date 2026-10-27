#!/bin/bash
# Seg2Any LoRA training on UEC-Ramen555 (500 train / 55 held out for eval).
# Usage: bash train_ramen.sh [extra --key=value overrides...]
#
# All settings live in config/seg2any_ramen.yaml; anything passed here overrides it,
# e.g. `bash train_ramen.sh --trainer.offload=True` if a GPU runs out of memory.
#
# Host RAM, not GPU memory, is what took gp41 down. Measured: loading the transformer
# without torch_dtype pinned 44.9GB per rank (fp32 upcast of a bf16 checkpoint), ~388GB
# across 8 ranks on a 251GB node. With the bf16 load in train.py a rank peaks at 10.4GB
# and settles at 1.1GB (mmap'd weights), i.e. ~83GB across 8 ranks, and
# trainer.load_group_size keeps the transient nearer ~21GB.
set -eu

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

# 8 ranks x default thread pool would each grab all 32 cores and thrash.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Return freed blocks to the driver instead of letting the caching allocator sit on them.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Refuse to start if the node cannot cover the staggered load, rather than OOM-killing it.
# 40GB is ~2x the measured ~21GB transient peak at load_group_size=2.
avail_gb=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
if [ "$avail_gb" -lt 40 ]; then
  echo "Only ${avail_gb}GB RAM available; need ~21GB for the staggered load. Aborting." >&2
  exit 1
fi
echo "MemAvailable: ${avail_gb}GB -- starting."

# Not the conda seg2any env: this repo has its own venv (torch 2.6.0+cu124).
.venv/bin/accelerate launch --config_file config/accelerate_8gpu_config.yaml \
  train.py config/seg2any_ramen.yaml "$@"
