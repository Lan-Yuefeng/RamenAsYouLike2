"""FLUX.2 klein-base-4B + LoRA で CFG の強さを変え、マスクの境界への忠実さがどう変わるかを調べる。

    HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 \
        .venv-flux2/bin/python flux2_cfg_sweep.py --lora_ckpt_path ckpt/ramen_flux2/checkpoint-3000

条件 (既定):
  cfg{1.0,1.5,2.0,3.0,4.0}     ネガティブ側にも cond を入れる (これまでの設定)。1.0 は CFG なし
  cfg{2.0,4.0}_negnocond       ネガティブ側から cond を外す (輪郭の情報が CFG で強調される)
境界の指標 (boundary_score):
  マスクの境界画素のうち、生成画像の Canny エッジが tol px 以内にある割合 (recall)。
  逆向きに、生成画像のエッジのうちマスク境界の tol px 以内にある割合 (precision) も出す。
  元写真と FLUX.1 の既存出力 (result/ramen_3000) も同じ指標で測り、目安にする。
出力: result_f2/flux2_cfg_sweep/ (画像、metrics.csv、timing、panel.jpg、README.md)
"""
import argparse
import csv
import json
import os
import statistics
import time

import cv2
import numpy as np
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
parser.add_argument("--cfgs", default="1.0,1.5,2.0,3.0,4.0")
parser.add_argument("--negnocond_cfgs", default="2.0,4.0")
parser.add_argument("--steps", type=int, default=20, help="§11.3 で 20 ステップは 50 とほぼ同じ品質だった")
parser.add_argument("--attention_mask_method", default="hard")
parser.add_argument("--flux1_result", default="result/ramen_3000")
parser.add_argument("--tol", type=int, default=3)
parser.add_argument("--output_path", default="result_f2/flux2_cfg_sweep")
args = parser.parse_args()
os.makedirs(args.output_path, exist_ok=True)


# ---------------- 境界の指標 (utils/boundary_metric.py) ----------------
from utils.boundary_metric import boundary_score, load_rgb  # noqa: E402


# ---------------- 生成 ----------------
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

conditions = [(f"cfg{float(c):.1f}", float(c), True) for c in args.cfgs.split(",") if c]
conditions += [(f"cfg{float(c):.1f}_negnocond", float(c), False) for c in args.negnocond_cfgs.split(",") if c]


def generate(batch, cfg, neg_cond):
    return pipe(
        global_prompt=batch["global_caption"], regional_prompts=batch["regional_captions"],
        regional_labels=batch["label"], cond=(batch["cond_pixel_values"][None] + 1) / 2,
        attention_mask_method=args.attention_mask_method, height=batch["image_height"], width=batch["image_width"],
        num_inference_steps=args.steps, guidance_scale=cfg, negative_use_cond=neg_cond,
        generator=torch.Generator("cuda").manual_seed(batch["seed"]),
    ).images[0]


generate(samples[0][1], 4.0, True)  # ウォームアップ
rows = []
for name, cfg, neg_cond in conditions:
    out_dir = os.path.join(args.output_path, name)
    os.makedirs(out_dir, exist_ok=True)
    for stem, batch in samples:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        image = generate(batch, cfg, neg_cond)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        image.save(os.path.join(out_dir, f"{stem}.jpg"), quality=95)
        rows.append({"condition": name, "image": stem, "sec": round(sec, 3)})
    print(name, "done", flush=True)

# ---------------- 指標 ----------------
metric_rows = []
refs = [("real_photo", lambda stem: os.path.join(RAMEN_IMAGES, stem.split("_")[1] + ".png")),
        ("flux1_ckpt3000", lambda stem: os.path.join(args.flux1_result, stem + "_0.jpg"))]
gens = [(name, (lambda n: (lambda stem: os.path.join(args.output_path, n, stem + ".jpg")))(name)) for name, _, _ in conditions]
for name, path_of in refs + gens:
    for stem, _ in samples:
        # 512 の色分けマスク (NEAREST で作成済み)。補間すると境界に中間色ができるのでリサイズしない
        mask = np.array(Image.open(os.path.join(args.seg_mask_path, stem + ".png")).convert("RGB"))
        r, p = boundary_score(load_rgb(path_of(stem)), mask, args.tol)
        metric_rows.append({"condition": name, "image": stem, "recall": round(r, 4), "precision": round(p, 4)})

summary = {}
for name in [n for n, _ in refs] + [n for n, _, _ in conditions]:
    rs = [m["recall"] for m in metric_rows if m["condition"] == name]
    ps = [m["precision"] for m in metric_rows if m["condition"] == name]
    secs = [r["sec"] for r in rows if r["condition"] == name]
    summary[name] = {"recall": round(statistics.mean(rs), 4), "precision": round(statistics.mean(ps), 4),
                     "sec_per_image": round(statistics.mean(secs), 2) if secs else None}
    print(f"{name:22s} recall {summary[name]['recall']:.3f}  precision {summary[name]['precision']:.3f}  "
          f"{summary[name]['sec_per_image'] or '-'} s")

with open(os.path.join(args.output_path, "metrics.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "image", "recall", "precision"])
    w.writeheader()
    w.writerows(metric_rows)
with open(os.path.join(args.output_path, "timing.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["condition", "image", "sec"])
    w.writeheader()
    w.writerows(rows)
json.dump({"model": args.model, "lora": args.lora_ckpt_path, "steps": args.steps, "tol": args.tol,
           "gpu": torch.cuda.get_device_name(0), "summary": summary},
          open(os.path.join(args.output_path, "summary.json"), "w"), indent=1)

# ---------------- 一覧 ----------------
cols = ["mask", "real photo", "FLUX.1"] + [n for n, _, _ in conditions]
s = 224
panel = Image.new("RGB", (s * len(cols), 24 + s * len(samples)), "white")
draw = ImageDraw.Draw(panel)
for j, c in enumerate(cols):
    draw.text((j * s + 4, 6), c, fill="black")
for i, (stem, _) in enumerate(samples):
    ims = [Image.open(os.path.join(args.seg_mask_path, stem + ".png")).convert("RGB"),
           Image.open(refs[0][1](stem)).convert("RGB"), Image.open(refs[1][1](stem)).convert("RGB")]
    ims += [Image.open(os.path.join(args.output_path, n, stem + ".jpg")) for n, _, _ in conditions]
    for j, im in enumerate(ims):
        panel.paste(im.resize((s, s)), (j * s, 24 + i * s))
panel.save(os.path.join(args.output_path, "panel.jpg"), quality=90)
print("saved", args.output_path)
