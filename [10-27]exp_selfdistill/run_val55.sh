#!/bin/bash
# val55 に広げた評価。4090 x4 で 4 条件を並列に生成 → mIoU / CLIP (eval.py)、bIoU (boundary_iou.py)、KID / FID (flux2_eval_val55.py)
# 出力は角括弧のないパス ~/Seg2Any/result_f3/val55_selfdistill (cleanfid の glob 対策)。tmux から起動する
set -u
cd "$(dirname "$0")"
export HF_HOME=/export/ssd3/lan-y/huggingface TOKENIZERS_PARALLELISM=false
PY=~/Seg2Any/.venv-flux3/bin
IN=~/Seg2Any/examples/examples_Ramen_val55
OUT=~/Seg2Any/result_f3/val55_selfdistill
OLD=/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000
SD=/export/space0/lan-y/seg2any_ckpt/ramen_flux3_selfdistill/checkpoint-2000
mkdir -p $OUT
CUDA_VISIBLE_DEVICES=0 $PY/python gen_val.py $IN $OUT base $OLD base > gen55_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY/python gen_val.py $IN $OUT ref_fuse $OLD fuse > gen55_fuse.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY/python gen_val.py $IN $OUT sd_0 $OLD fuse_cache > gen55_sd0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY/python gen_val.py $IN $OUT sd_2000 $SD fuse_cache > gen55_sd2000.log 2>&1 &
wait
echo "gen done $(date)"
CONDS=base,ref_fuse,sd_0,sd_2000
cd ~/Seg2Any/"[10-21]exp_layout_cfg"
CUDA_VISIBLE_DEVICES=0 $PY/python eval.py --inputs $IN --root $OUT --conditions $CONDS \
  --seg_model /export/ssd3/lan-y/seg2any_ckpt/ramen_segformer_b4 --panel_n 12 > $OUT/eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY/python boundary_iou.py --inputs $IN --root $OUT --conditions $CONDS > $OUT/biou.log 2>&1 &
cd ~/Seg2Any
CUDA_VISIBLE_DEVICES=2 $PY/python flux2_eval_val55.py --seg_mask_path $IN --result_root $OUT --conditions $CONDS > $OUT/kid.log 2>&1 &
wait
echo "ALL DONE $(date)"
