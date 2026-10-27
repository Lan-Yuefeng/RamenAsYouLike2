# v2 LoRA (再学習) checkpoint-3000、CFG 3.0、16 ステップ

- モデル：`black-forest-labs/FLUX.2-klein-base-4B` + LoRA `/export/ssd3/lan-y/seg2any_ckpt/ramen_flux2_v2/checkpoint-3000`
  （config/seg2any_ramen_flux2_v2.yaml。gp42 8GPU・全体バッチ 16 で中断なしに 3000 step、1h07m）
- 設定：CFG 3.0、16 ステップ、512×512、hard マスク、cond_scale_factor 1、新既定（ネガティブ側に cond なし）。gp42 GPU 0
- 入力：`input/`（result_f2/cfg3.0_steps16 と同じ 10 例・同じ seed）→ v1 と直接比べられる
- 生成コマンド：
  ```bash
  HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 .venv-flux2/bin/python infer_flux2.py \
      --seg_mask_path result_f2/v2_cfg3.0_steps16/input --lora_ckpt_path /export/ssd3/lan-y/seg2any_ckpt/ramen_flux2_v2/checkpoint-3000 \
      --output_path result_f2/v2_cfg3.0_steps16/generated --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \
      --guidance_scale 3.0 --num_inference_steps 16 --cond_scale_factor 1
  .venv-flux2/bin/python scripts/make_result_panels.py result_f2/v2_cfg3.0_steps16 --title "v2 ckpt-3000, CFG 3.0, 16 steps"
  .venv-flux2/bin/python compare_v1_v2.py
  ```

## ファイル
- `generated/`、`overlay/`、`panel_mask_gt_gen.jpg`、`panel_overlay.jpg`：result_f2/cfg3.0_steps16 と同じ形式
- `compare_v1_v2.jpg`：各行 マスク | GT | v1 | v2

## v1 との比較（境界 recall、utils/boundary_metric.py、tol 3px）
| | 平均 | 各例 |
|---|---|---|
| v1 | 0.772 | 0.52 0.76 0.79 0.83 0.78 0.70 0.83 0.92 0.69 0.91 |
| v2 | **0.925** | 0.72 0.93 0.93 0.95 0.96 0.91 0.94 0.97 0.95 0.99 |

10 例すべてで v2 が上。見た目の配置・色はほぼ同じで、輪郭線を描き込むような破綻は無い。
ただし 10 例・seed 1 つだけなので、確定させるなら val 55 枚で評価する。

> 注: ここに出てくる段階確認スクリプトは整理のため削除済み。`git show 691c47a:<ファイル名>` で取り出せる。
