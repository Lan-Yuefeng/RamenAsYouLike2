#!/bin/bash
# Step 2：val55 全体、50 ステップで 今の既定 / s_lay 7 / s_lay 10 を比べる (FLUX.3 Base + ramen LoRA、512²)
# 3 条件 × 55 枚を 24 個のジョブ (各条件 8 分割) にして、GPU 8 枚に 3 つずつ載せる
set -u
cd ~/Seg2Any/"[10-21]exp_layout_cfg"
export HF_HOME=/export/ssd3/lan-y/huggingface
unset HF_HUB_OFFLINE
R=results_val55; mkdir -p $R/logs
G="../.venv-flux3/bin/python gen.py --inputs inputs_val55 --out_root $R --steps 50"
for g in 0 1 2 3 4 5 6 7; do
  ( CUDA_VISIBLE_DEVICES=$g $G --name base_s3.5    --s_txt 3.5 --s_lay 3.5 --shard $g/8 > $R/logs/base_$g.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $G --name txt3.5_lay7  --s_txt 3.5 --s_lay 7   --shard $g/8 > $R/logs/lay7_$g.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $G --name txt3.5_lay10 --s_txt 3.5 --s_lay 10  --shard $g/8 > $R/logs/lay10_$g.log 2>&1
    echo "gpu$g done" ) &
done
wait
echo ALLDONE
