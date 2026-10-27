#!/bin/bash
# Step 1：レイアウトとテキストの強さを分けた CFG を 10 例で試す (FLUX.3 Base + ramen LoRA checkpoint-3000、512²、20 ステップ)
set -u
cd ~/Seg2Any/"[10-21]exp_layout_cfg"
export HF_HOME=/export/ssd3/lan-y/huggingface
unset HF_HUB_OFFLINE
R=results; mkdir -p $R/logs
G="../.venv-flux3/bin/python gen.py"
t() { local gpu=$1 name=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu $G --name $name "$@" > $R/logs/$name.log 2>&1; echo "$name rc=$?"; }
t 0 base_s3.5          --s_txt 3.5 --s_lay 3.5 &              # 今の既定 (テキストもレイアウトも 3.5)
t 1 oldneg_s3.5        --s_txt 3.5 --old_neg &                # 旧既定 (テキストだけ 3.5、レイアウトは 1)
t 2 txt3.5_lay5        --s_txt 3.5 --s_lay 5 &
t 3 txt3.5_lay7        --s_txt 3.5 --s_lay 7 &
t 4 txt3.5_lay10       --s_txt 3.5 --s_lay 10 &
t 5 txt2_lay7          --s_txt 2 --s_lay 7 &
t 6 txt3.5_lay7early   --s_txt 3.5 --s_lay 7 --s_lay_late 3.5 --lay_switch 0.4 &   # 序盤 40% だけレイアウトを強く
t 7 txt3.5_lay7late    --s_txt 3.5 --s_lay 3.5 --s_lay_late 7 --lay_switch 0.4 &   # 終盤 60% だけレイアウトを強く
wait
echo ALLDONE
