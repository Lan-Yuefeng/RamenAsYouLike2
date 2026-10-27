#!/bin/bash
# val10 で VLM caption 版 LoRA の生成 (4 条件を GPU 0-3 で並列)。設定は exp_ablation の full_hard と同じ (hard, 50 step, CFG 3.5)
cd "$(dirname "$0")"
export HF_HOME=/export/ssd3/lan-y/huggingface
LORA=/export/space0/lan-y/seg2any_ckpt/ramen_flux3_vlmcap/checkpoint-3000
G=../"[10-25]exp_ablation"/gen_ablation.py
mkdir -p results/logs
i=0
for c in oracle short_raw short_zs short_ft; do
  CUDA_VISIBLE_DEVICES=$i ../.venv-flux3/bin/python "$G" --name vlm_$c --inputs inputs/val10_$c --out_root results \
    --lora $LORA > results/logs/vlm_$c.log 2>&1 &
  i=$((i+1))
done
wait
echo GEN_DONE > results/logs/done.txt
