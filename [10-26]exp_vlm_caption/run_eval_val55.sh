#!/bin/bash
# val55 の評価 (Base 版と蒸留版の両方): スープ判定 (Qwen3-VL / InternVL3) と mIoU・境界
cd "$(dirname "$0")"
PY=../.venv-flux3/bin/python
CONDS=hand_full,hand_short_raw,vlm_oracle,vlm_short_raw,vlm_short_zs,vlm_short_ft,vlm_short_ftv2
MASKS=../examples/examples_Ramen_val55
gpu=0
for R in results_val55 results_distill_val55; do
  mkdir -p $R/judge
  for m in Qwen/Qwen3-VL-8B-Instruct OpenGVLab/InternVL3-8B-hf; do
    tag=$(basename $m | cut -d- -f1)
    ( export HF_HOME=/export/ssd2/lan-y/huggingface CUDA_VISIBLE_DEVICES=$gpu
      [ -f results_val55/judge/real_$tag.json ] || $PY judge_soup.py --model $m --split val55.txt \
          --images /export/space0/lan-y/UEC-Ramen555/UEC-Ramen555/Ramen_images --out results_val55/judge/real_$tag.json
      for c in ${CONDS//,/ }; do
        $PY judge_soup.py --model $m --split val55.txt --images $R/$c --pattern 'ramen_{stem}_0.jpg' --out $R/judge/${c}_$tag.json
      done ) > $R/logs/judge_$tag.log 2>&1 &
    gpu=$((gpu+1))
  done
  ( export HF_HOME=/export/ssd3/lan-y/huggingface CUDA_VISIBLE_DEVICES=$gpu; A=$(realpath $R)
    cd ../"[10-21]exp_layout_cfg" && $PY eval.py --inputs ../examples/examples_Ramen_val55 --root $A --conditions $CONDS \
      --seg_model /export/ssd3/lan-y/seg2any_ckpt/ramen_segformer_b4 && \
    $PY boundary_iou.py --inputs ../examples/examples_Ramen_val55 --root $A --conditions $CONDS ) > $R/logs/eval.log 2>&1 &
  gpu=$((gpu+1))
done
wait
echo EVAL_DONE >> results_val55/logs/done.txt
