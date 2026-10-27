#!/bin/bash
# 教師の軌跡を GPU 4 枚 (gp36 の 4090) で分担して保存する。tmux から起動する
cd "$(dirname "$0")"
export HF_HOME=/export/ssd3/lan-y/huggingface TOKENIZERS_PARALLELISM=false
for s in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$s ~/Seg2Any/.venv-flux3/bin/python precompute_teacher.py $s 4 > precompute_$s.log 2>&1 &
done
wait
echo ALL DONE
