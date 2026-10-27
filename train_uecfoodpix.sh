#!/bin/bash
# Seg2Any LoRA training on UECFOODPIXCOMPLETE (9000 train / 1000 official test held out).
# Usage: bash train_uecfoodpix.sh [extra --key=value overrides...]
#
# Defaults target gp36's two A100 80GB cards (GPU 4,5 -- GPU 0-3 are RTX 4090 24GB, too
# small for this: the FLUX transformer alone is 22.2GiB in bf16). Override for another
# node, e.g. on gp38 (A6000 48GB x8):
#   SEG2ANY_GPUS=0,1,2,3,4,5,6,7 SEG2ANY_ACCEL=config/accelerate_8gpu_config.yaml \
#     bash train_uecfoodpix.sh --trainer.train_batch_size=2
#
# Anything passed here overrides config/seg2any_uecfoodpix.yaml, e.g.
# `--trainer.train_batch_size=4` if a GPU runs out of memory, or
# `--resume_from_checkpoint=latest` to continue a stopped run.
set -eu

cd "$(dirname "$0")"

# $HOME is at 94% on a shared 19T volume, so nothing large may land there. FLUX.1-dev
# (32GB) is already cached under this HF_HOME, and checkpoints go to
# /export/ssd3/lan-y/seg2any_ckpt/uecfoodpix via project.output_dir in the config.
export HF_HOME=/export/ssd3/lan-y/huggingface

export CUDA_VISIBLE_DEVICES=${SEG2ANY_GPUS:-4,5}
ACCEL=${SEG2ANY_ACCEL:-config/accelerate_2gpu_config.yaml}

export TOKENIZERS_PARALLELISM=false

# Keep each rank from grabbing all cores and thrashing (gp36 has 20).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Return freed blocks to the driver instead of letting the caching allocator sit on them.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Host RAM, not GPU memory, is what took gp41 down on the ramen run: loading the
# transformer without torch_dtype pinned 44.9GB per rank (fp32 upcast of a bf16
# checkpoint). With the bf16 load in train.py a rank peaks at a measured 10.4GB and
# settles at 1.1GB (mmap'd weights); trainer.load_group_size keeps the transient down.
avail_gb=$(awk '/MemAvailable/ {print int($2/1024/1024)}' /proc/meminfo)
if [ "$avail_gb" -lt 40 ]; then
  echo "Only ${avail_gb}GB RAM available; need ~21GB for the staggered load. Aborting." >&2
  exit 1
fi

# The ramen run only ever checked host RAM, which is how a start into 8 fully-occupied
# GPUs became possible. Refuse to launch unless every selected GPU can actually hold a
# rank (~50GiB at train_batch_size=8; 40GiB is the floor below which it cannot work).
need_mib=40000
for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
  free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$free" -lt "$need_mib" ]; then
    echo "GPU ${gpu}: only ${free}MiB free, need ${need_mib}MiB. Someone else is using it. Aborting." >&2
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv >&2
    exit 1
  fi
done
echo "MemAvailable: ${avail_gb}GB. GPUs ${CUDA_VISIBLE_DEVICES} free. Starting."

# The bucket cache is built once on the first run (~9000 masks, ~4 min) and reused after.
cache=/export/ssd3/lan-y/seg2any_data/uecfoodpix/cache/512H_512W-group_bucket.parquet
[ -f "$cache" ] || echo "Group-bucket cache missing; rank 0 will build it first (~4 min)."

# Not the conda seg2any env: this repo has its own venv (torch 2.6.0+cu124).
.venv/bin/accelerate launch --config_file "$ACCEL" \
  train.py config/seg2any_uecfoodpix.yaml "$@"
