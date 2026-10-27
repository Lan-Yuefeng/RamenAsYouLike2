# FLUX.2 klein-base-4B + LoRA checkpoint-3000、CFG 3.5、16 ステップ

- モデル：`black-forest-labs/FLUX.2-klein-base-4B` + LoRA `ckpt/ramen_flux2/checkpoint-3000`
- 設定：CFG 3.5、16 ステップ、512×512、hard マスク、cond_scale_factor 1、**新しい既定（CFG のネガティブ側に cond を入れない）**
- 入力：`input/`。いつもの 10 例（examples_Ramen と同じマスク）だが、caption は学習時と同じ文言の `examples_Ramen_val55` から取ったもの。seed は各 json（1000 + 画像番号）
- 生成コマンド：
  ```bash
  HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 .venv-flux2/bin/python infer_flux2.py \
      --seg_mask_path result_f2/cfg3.5/input --lora_ckpt_path ckpt/ramen_flux2/checkpoint-3000 --output_path result_f2/cfg3.5/generated \
      --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B --guidance_scale 3.5 --num_inference_steps 16 --cond_scale_factor 1
  ```
- GPU：gp13（RTX 6000 Ada）

## ファイル
- `generated/ramen_XXXX_0.jpg`：生成画像
- `overlay/ramen_XXXX_overlay.jpg`：生成画像の上にマスクの色を 45% の不透明度で重ね、領域の境界を白線で描いたもの
- `panel_mask_gt_gen.jpg`：各行 マスク | GT（元写真。512² に変形）| 生成
- `panel_overlay.jpg`：各行 マスク | 生成 | マスクを重ねた生成
- `generate.log`：生成ログ
