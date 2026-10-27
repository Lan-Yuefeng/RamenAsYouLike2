"""Step 3 の目視確認 (LoRA なし = training-free)。examples_Ramen の各例を同じ seed で
  素の klein (global caption のみ) / 領域マスク + L 振り直し / 領域マスク + L 通し番号
の 3 通りで生成し、マスクと横に並べて保存する。cond は学習前なので使わない。

    HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 .venv-flux2/bin/python flux2_step3_compare.py
"""
import argparse
import os

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageDraw

from infer import load_seg_map_and_prompt
from src.models.transformer_flux2 import Flux2RegionalTransformer2DModel
from src.pipelines.pipeline_flux2 import Flux2RegionalPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
parser.add_argument("--seg_mask_path", default="examples_Ramen")
parser.add_argument("--output_path", default="result_f2/flux2_step3_compare")
parser.add_argument("--attention_mask_method", default="hard")
args = parser.parse_args()
os.makedirs(args.output_path, exist_ok=True)

transformer = Flux2RegionalTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux2RegionalPipeline.from_pretrained(args.model, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")
steps = 4 if pipe.config.is_distilled else 50

COLS = ["mask", "klein (global only)", "regional / restart", "regional / continuous"]
rows = []
for f in sorted(x for x in os.listdir(args.seg_mask_path) if x.endswith(".json")):
    path = os.path.join(args.seg_mask_path, f)
    batch = load_seg_map_and_prompt(path.replace(".json", ".png"), path, 1)
    seed = batch["seed"] or 0
    h, w = batch["image_height"], batch["image_width"]

    vanilla = Flux2KleinPipeline.__call__(
        pipe, prompt=batch["global_caption"], height=h, width=w, num_inference_steps=steps,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    regional = {}
    for mode in ["restart", "continuous"]:
        regional[mode] = pipe(
            global_prompt=batch["global_caption"], regional_prompts=batch["regional_captions"],
            regional_labels=batch["label"], cond=None, attention_mask_method=args.attention_mask_method,
            regional_text_position=mode, height=h, width=w, num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        regional[mode].save(os.path.join(args.output_path, f.replace(".json", f"_{mode}.jpg")), quality=95)
    vanilla.save(os.path.join(args.output_path, f.replace(".json", "_klein.jpg")), quality=95)

    mask = Image.open(path.replace(".json", ".png")).convert("RGB").resize((w, h), Image.NEAREST)
    rows.append([mask, vanilla, regional["restart"], regional["continuous"]])
    print("done", f)

# 一覧 (縮小して 1 枚に)
s = 256
panel = Image.new("RGB", (s * len(COLS), 24 + s * len(rows)), "white")
draw = ImageDraw.Draw(panel)
for j, name in enumerate(COLS):
    draw.text((j * s + 6, 6), name, fill="black")
for i, row in enumerate(rows):
    for j, im in enumerate(row):
        panel.paste(im.resize((s, s)), (j * s, 24 + i * s))
panel.save(os.path.join(args.output_path, "panel.jpg"), quality=90)
print("saved", os.path.join(args.output_path, "panel.jpg"))
