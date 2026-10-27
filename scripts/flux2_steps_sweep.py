"""FLUX.2 klein-base-4B + 学習済み LoRA で、推論ステップ数を変えたときの生成結果と生成時間を比べる。

    HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 \
        .venv-flux2/bin/python scripts/flux2_steps_sweep.py --lora_ckpt_path ckpt/ramen_flux2/checkpoint-3000

出力 (既定 result_f2/flux2_steps_sweep/):
  steps{N}/ramen_XXXX.jpg   各ステップ数の生成画像
  timing.csv / timing.json  画像ごと・ステップ数ごとの生成時間
  panel.jpg                 マスク | 元写真 | 50 | 40 | 30 | 20 | 10 | (参考) 蒸留版 4 step
  README.md                 条件と結果の表
モデルは 1 回だけ読み込み、本番の前に 1 枚ウォームアップしてから測る (初回の CUDA 初期化を含めないため)。
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # リポジトリ直下 (src/ dataset/ utils/ infer.py) を import できるように
import argparse
import csv
import json
import os
import statistics
import time

import torch
from PIL import Image, ImageDraw

from infer import load_seg_map_and_prompt
from src.models.transformer_flux2 import Flux2RegionalTransformer2DModel
from src.pipelines.pipeline_flux2 import Flux2RegionalPipeline

RAMEN_IMAGES = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555/Ramen_images"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-4B")
parser.add_argument("--lora_ckpt_path", required=True)
parser.add_argument("--seg_mask_path", default="examples_Ramen")
parser.add_argument("--steps", default="50,40,30,20,10")
parser.add_argument("--guidance_scale", type=float, default=4.0)
parser.add_argument("--attention_mask_method", default="hard")
parser.add_argument("--distilled_dir", default="result_f2/flux2_checkpoint-3000", help="参考列 (蒸留版 4 step) の既存出力")
parser.add_argument("--output_path", default="result_f2/flux2_steps_sweep")
args = parser.parse_args()

steps_list = [int(s) for s in args.steps.split(",")]
os.makedirs(args.output_path, exist_ok=True)

transformer = Flux2RegionalTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux2RegionalPipeline.from_pretrained(args.model, transformer=transformer, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(args.lora_ckpt_path, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(args.lora_ckpt_path, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

samples = []
for f in sorted(x for x in os.listdir(args.seg_mask_path) if x.endswith(".json")):
    path = os.path.join(args.seg_mask_path, f)
    samples.append((f[:-5], load_seg_map_and_prompt(path.replace(".json", ".png"), path, 1)))


def generate(batch, steps):
    return pipe(
        global_prompt=batch["global_caption"], regional_prompts=batch["regional_captions"],
        regional_labels=batch["label"], cond=(batch["cond_pixel_values"][None] + 1) / 2,
        attention_mask_method=args.attention_mask_method, height=batch["image_height"], width=batch["image_width"],
        num_inference_steps=steps, guidance_scale=args.guidance_scale,
        generator=torch.Generator("cuda").manual_seed(batch["seed"]),
    ).images[0]


generate(samples[0][1], 4)  # ウォームアップ (計測に含めない)

rows = []
for steps in steps_list:
    out_dir = os.path.join(args.output_path, f"steps{steps}")
    os.makedirs(out_dir, exist_ok=True)
    for stem, batch in samples:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image = generate(batch, steps)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        image.save(os.path.join(out_dir, f"{stem}.jpg"), quality=95)
        rows.append({"steps": steps, "image": stem, "sec": round(sec, 3)})
    secs = [r["sec"] for r in rows if r["steps"] == steps]
    print(f"steps={steps}: mean {statistics.mean(secs):.2f}s / image")

gpu = torch.cuda.get_device_name(0)
summary = {}
for steps in steps_list:
    secs = [r["sec"] for r in rows if r["steps"] == steps]
    summary[steps] = {
        "mean_sec": round(statistics.mean(secs), 3), "min_sec": min(secs), "max_sec": max(secs),
        "sec_per_step": round(statistics.mean(secs) / steps, 3),
    }

with open(os.path.join(args.output_path, "timing.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["steps", "image", "sec"])
    w.writeheader()
    w.writerows(rows)
json.dump({"model": args.model, "lora": args.lora_ckpt_path, "gpu": gpu, "resolution": "512x512",
           "guidance_scale": args.guidance_scale, "attention_mask_method": args.attention_mask_method,
           "summary": summary, "per_image": rows},
          open(os.path.join(args.output_path, "timing.json"), "w"), indent=1)

# 一覧画像
cols = ["mask", "real photo"] + [f"base {s} steps ({summary[s]['mean_sec']:.1f}s)" for s in steps_list]
has_distilled = all(os.path.exists(os.path.join(args.distilled_dir, f"{stem}_distilled.jpg")) for stem, _ in samples)
if has_distilled:
    cols.append("(ref) distilled 4 steps")
s = 256
panel = Image.new("RGB", (s * len(cols), 24 + s * len(samples)), "white")
draw = ImageDraw.Draw(panel)
for j, c in enumerate(cols):
    draw.text((j * s + 6, 6), c, fill="black")
for i, (stem, _) in enumerate(samples):
    ims = [Image.open(os.path.join(args.seg_mask_path, stem + ".png")).convert("RGB"),
           Image.open(os.path.join(RAMEN_IMAGES, stem.split("_")[1] + ".png")).convert("RGB")]
    ims += [Image.open(os.path.join(args.output_path, f"steps{st}", stem + ".jpg")) for st in steps_list]
    if has_distilled:
        ims.append(Image.open(os.path.join(args.distilled_dir, f"{stem}_distilled.jpg")))
    for j, im in enumerate(ims):
        panel.paste(im.resize((s, s)), (j * s, 24 + i * s))
panel.save(os.path.join(args.output_path, "panel.jpg"), quality=90)

# README
base = summary[max(steps_list)]["mean_sec"]
lines = [
    "# FLUX.2 klein-base-4B: 推論ステップ数の比較", "",
    f"- モデル: `{args.model}` + LoRA `{args.lora_ckpt_path}`",
    f"- 入力: `{args.seg_mask_path}` の {len(samples)} 例 (seed は各 json)。512×512、CFG {args.guidance_scale}、"
    f"`attention_mask_method={args.attention_mask_method}`",
    f"- GPU: {gpu} 1 枚。時間はテキストエンコード〜VAE デコードまでを含む 1 枚あたりの時間 (ウォームアップ 1 枚は除外)",
    f"- 作成: `scripts/flux2_steps_sweep.py`", "",
    "| ステップ数 | 平均 (秒/枚) | 最小 | 最大 | 1 ステップあたり | 50 ステップ比 |",
    "|---|---|---|---|---|---|",
]
for st in steps_list:
    m = summary[st]
    lines.append(f"| {st} | {m['mean_sec']:.2f} | {m['min_sec']:.2f} | {m['max_sec']:.2f} | {m['sec_per_step']:.3f} | "
                 f"{m['mean_sec'] / base:.2f}× |")
lines += ["", "ファイル: `steps{N}/` 各画像、`timing.csv` 画像ごとの時間、`timing.json` 条件と集計、`panel.jpg` 一覧"]
open(os.path.join(args.output_path, "README.md"), "w").write("\n".join(lines) + "\n")
print("saved", args.output_path)
