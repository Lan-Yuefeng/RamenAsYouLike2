#!/bin/bash
# 教師の事前計算の完了を待って → 自己蒸留の学習 (gp36 の GPU 0-5: 4090 x4 + A100 x2) → val10 生成 → 評価。tmux から起動する
set -u
cd "$(dirname "$0")"
export HF_HOME=/export/ssd3/lan-y/huggingface TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/Seg2Any/.venv-flux3/bin
OUT=/export/space0/lan-y/seg2any_ckpt/ramen_flux3_selfdistill
until grep -q "ALL DONE" precompute_all.log 2>/dev/null; do sleep 30; done
echo "precompute done: $(ls /export/ssd2/lan-y/seg2any_data/ramen_selfdistill/teacher_steps4/*.pt | wc -l) files $(date)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $PY/torchrun --nproc_per_node 6 --master_port 29612 train_selfdistill.py --out $OUT \
  --max_steps 2000 --ckpt_every 500 > train.log 2>&1 || { echo "train failed $(date)"; exit 1; }
echo "train done $(date)"

# 比較用: ① だけ (学習前 LoRA、キャッシュなし) と ①+③ の学習前 (exp_speed の結果)
ln -sfn "../../[10-26]exp_speed/result/cache_fuse" result/ref_fuse 2>/dev/null || { mkdir -p result; ln -sfn "../../[10-26]exp_speed/result/cache_fuse" result/ref_fuse; }
ln -sfn "../../[10-26]exp_speed/result/cache_fuse_cache" result/sd_0
CUDA_VISIBLE_DEVICES=4 $PY/python gen_val10.py sd_500 $OUT/checkpoint-500 sd_1000 $OUT/checkpoint-1000 > gen_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $PY/python gen_val10.py sd_1500 $OUT/checkpoint-1500 sd_2000 $OUT/checkpoint-2000 > gen_b.log 2>&1 &
wait
echo "gen done $(date)"

VAL=~/Seg2Any/"[10-26]exp_speed"/val10
CONDS=ref_fuse,sd_0,sd_500,sd_1000,sd_1500,sd_2000
cd ~/Seg2Any/"[10-21]exp_layout_cfg"
CUDA_VISIBLE_DEVICES=4 $PY/python eval.py --inputs "$VAL" --root ~/Seg2Any/"[10-27]exp_selfdistill"/result --conditions $CONDS \
  --seg_model /export/ssd3/lan-y/seg2any_ckpt/ramen_segformer_b4 --panel_n 10 > ~/Seg2Any/"[10-27]exp_selfdistill"/eval.log 2>&1
CUDA_VISIBLE_DEVICES=4 $PY/python boundary_iou.py --inputs "$VAL" --root ~/Seg2Any/"[10-27]exp_selfdistill"/result --conditions $CONDS \
  > ~/Seg2Any/"[10-27]exp_selfdistill"/biou.log 2>&1
echo "ALL DONE $(date)"
