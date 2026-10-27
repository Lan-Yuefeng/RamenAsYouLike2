#!/bin/bash
# 学習後の評価 (val55 の先頭 10 枚)。regional caption は 全文 / 具材名だけ / 空 の 3 通り。
# ベースライン (意味なし) は ../[10-25]exp_ablation の full_hard / cap_classname / cap_empty (同じ gp41・同じ設定) へのリンク
set -u
cd ~/Seg2Any/"[10-25]exp_semcond"
export HF_HOME=/export/ssd3/lan-y/huggingface
unset HF_HUB_OFFLINE
CK=${CK:-/export/space0/lan-y/seg2any_ckpt/ramen_flux3_semcond/checkpoint-3000}
A="../[10-25]exp_ablation"
R=results; mkdir -p $R/logs
ln -sfn "../$A/results/full_hard" $R/base_full
ln -sfn "../$A/results/cap_classname" $R/base_classname
ln -sfn "../$A/results/cap_empty" $R/base_empty
G="../.venv-flux3/bin/python gen_semcond.py --ckpt $CK"
t() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $G --name $name "$@" > $R/logs/$name.log 2>&1; echo "$name rc=$?"; }
t 0 semcond_full      --inputs "$A/inputs_val10" &
t 1 semcond_classname --inputs "$A/inputs_val10_classname" &
t 2 semcond_empty     --inputs "$A/inputs_val10_noregional" &
t 3 semcond_full_nosem --inputs "$A/inputs_val10" --no_sem &   # 同じ LoRA で W だけ外す (W に頼っている度合い)
wait
E="../[10-21]exp_layout_cfg"
C=base_full,semcond_full,semcond_full_nosem,base_classname,semcond_classname,base_empty,semcond_empty
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 ../.venv-flux3/bin/python "$E/eval.py" --inputs "$A/inputs_val10" --root $R --conditions $C \
  --seg_model /export/space0/lan-y/seg2any_ckpt/ramen_segformer_b4 > $R/eval.txt 2>&1
CUDA_VISIBLE_DEVICES=0 ../.venv-flux3/bin/python "$E/boundary_iou.py" --inputs "$A/inputs_val10" --root $R --conditions $C \
  --seg_model /export/space0/lan-y/seg2any_ckpt/ramen_segformer_b4 > $R/boundary_iou.txt 2>&1
echo ALLDONE
