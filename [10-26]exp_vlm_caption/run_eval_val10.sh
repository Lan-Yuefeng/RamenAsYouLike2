#!/bin/bash
# val10 の評価: スープ判定 (Qwen3-VL / InternVL3) と mIoU・境界 (exp_layout_cfg の eval.py / boundary_iou.py)
cd "$(dirname "$0")"
export HF_HOME=/export/ssd2/lan-y/huggingface
PY=../.venv-flux3/bin/python
CONDS=hand_full,hand_short_raw,vlm_oracle,vlm_short_raw,vlm_short_zs,vlm_short_ft
mkdir -p results/judge
i=0
for m in Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3-8B-hf; do
  tag=$(basename $m | cut -d- -f1)
  ( CUDA_VISIBLE_DEVICES=$i $PY judge_soup.py --model $m --split val10.txt \
      --images /export/space0/lan-y/UEC-Ramen555/UEC-Ramen555/Ramen_images --out results/judge/real_$tag.json
    for c in ${CONDS//,/ }; do
      CUDA_VISIBLE_DEVICES=$i $PY judge_soup.py --model $m --split val10.txt --images results/$c \
        --pattern 'ramen_{stem}_0.jpg' --out results/judge/${c}_$tag.json
    done ) > results/logs/judge_$tag.log 2>&1 &
  i=$((i+1))
done
export HF_HOME=/export/ssd3/lan-y/huggingface
( cd ../"[10-21]exp_layout_cfg" && CUDA_VISIBLE_DEVICES=2 $PY eval.py --inputs ../"[10-25]exp_ablation"/inputs_val10 \
    --root ../"[10-26]exp_vlm_caption"/results --conditions $CONDS --seg_model /export/ssd3/lan-y/seg2any_ckpt/ramen_segformer_b4 \
  && CUDA_VISIBLE_DEVICES=2 $PY boundary_iou.py --inputs ../"[10-25]exp_ablation"/inputs_val10 \
    --root ../"[10-26]exp_vlm_caption"/results --conditions $CONDS ) > results/logs/eval.log 2>&1 &
wait
echo EVAL_DONE >> results/logs/eval.log
