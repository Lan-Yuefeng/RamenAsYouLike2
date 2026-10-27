#!/bin/bash
# Seg2Any の部品を推論時に 1 つずつ外す (FLUX.3 Base + ramen LoRA checkpoint-3000、512²、50 ステップ、CFG 3.5、val55 の先頭 10 枚)
set -u
cd ~/Seg2Any/"[10-25]exp_ablation"
export HF_HOME=/export/ssd3/lan-y/huggingface
unset HF_HUB_OFFLINE
R=results; mkdir -p $R/logs
G="../.venv-flux3/bin/python gen_ablation.py"
t() { local gpu=$1 name=$2; shift 2; local s=$(date +%s); CUDA_VISIBLE_DEVICES=$gpu $G --name $name "$@" > $R/logs/$name.log 2>&1; echo "$name rc=$? $(( $(date +%s) - s ))s"; }
t 0 full_hard        &                                            # 推論の既定 (SAA + AIA [5,18] + cond + LoRA)
t 1 mask_base        --mask_method base &                         # AIA なし (= 学習時と同じマスク)
t 2 hard_all_blocks  --hard_range 0 32 &                          # AIA を全ブロックに
t 3 no_cond          --no_cond &                                  # 輪郭 cond なし (領域アテンションマスクだけで形を伝える)
t 4 no_lora          --no_lora &                                  # 学習なし (素の Base 版 + 領域アテンションマスク)
t 5 cap_classname    --inputs inputs_val10_classname &            # regional caption を具材名だけに
t 6 cap_empty        --inputs inputs_val10_noregional &           # regional caption を空に
t 7 mask_base_no_cond --mask_method base --no_cond &              # AIA も cond もなし (SAA だけ)
wait
echo ALLDONE
