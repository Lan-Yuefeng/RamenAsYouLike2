#!/bin/bash
# 全 555 枚に VLM caption を付ける (GPU 8 枚で分割)
cd "$(dirname "$0")"
export HF_HOME=/export/ssd2/lan-y/huggingface
mkdir -p logs
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i ../.venv-flux3/bin/python caption_vlm.py --split all555.txt --out captions_all \
    --shard $i --num_shards 8 > logs/shard$i.log 2>&1 &
done
wait
echo ALL_DONE >> logs/done.txt
