# FLUX.2 klein-base-4B + LoRA checkpoint-3000、CFG 3.0、16 ステップ

- モデル：`black-forest-labs/FLUX.2-klein-base-4B` + LoRA `ckpt/ramen_flux2/checkpoint-3000`
- 設定：CFG 3.0、16 ステップ、512×512、hard マスク、cond_scale_factor 1、**新しい既定（CFG のネガティブ側に cond を入れない）**
- 入力：`input/`（`result_f2/cfg3.5/input` と同じ 10 例。caption は `examples_Ramen_val55` 由来、seed は各 json = 1000 + 画像番号）
- 生成コマンド：
  ```bash
  HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 .venv-flux2/bin/python infer_flux2.py \
      --seg_mask_path result_f2/cfg3.0_steps16/input --lora_ckpt_path ckpt/ramen_flux2/checkpoint-3000 --output_path result_f2/cfg3.0_steps16/generated \
      --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B --guidance_scale 3.0 --num_inference_steps 16 --cond_scale_factor 1
  ```
- パネル作成：`.venv-flux2/bin/python make_result_panels.py result_f2/cfg3.0_steps16 --title "CFG 3.0, 16 steps"`
- GPU：gp13

## ファイル
- `generated/ramen_XXXX_0.jpg`：生成画像
- `overlay/ramen_XXXX_overlay.jpg`：生成画像にマスク色を 45% で重ね、領域境界を白線で描いたもの
- `panel_mask_gt_gen.jpg`：各行 マスク | GT（元写真。512² に変形）| 生成
- `panel_overlay.jpg`：各行 マスク | 生成 | マスクを重ねた生成
- `generate.log`：生成ログ
