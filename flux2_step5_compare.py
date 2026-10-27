"""Step 5: 学習した FLUX.2 LoRA を Base 版 (CFG + 50 step) と蒸留版 (4 step) の両方で推論し、
FLUX.1 版 (result/ramen_3000 = ckpt/ramen/checkpoint-3000 の既存出力) と元写真と並べる。

    HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 .venv-flux2/bin/python flux2_step5_compare.py \
        --lora_ckpt_path ckpt/ramen_flux2/checkpoint-3000

seed は examples_Ramen/*.json のもの (FLUX.1 版の出力と同じ)。
"""
import argparse
import json
import os
import time

import torch
from PIL import Image, ImageDraw

from infer import load_seg_map_and_prompt
from src.models.transformer_flux2 import Flux2RegionalTransformer2DModel
from src.pipelines.pipeline_flux2 import Flux2RegionalPipeline

RAMEN_IMAGES = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555/Ramen_images"

parser = argparse.ArgumentParser()
parser.add_argument("--lora_ckpt_path", required=True)
parser.add_argument("--seg_mask_path", default="examples_Ramen")
parser.add_argument("--flux1_result", default="result/ramen_3000")
parser.add_argument("--output_path", default=None)
parser.add_argument("--attention_mask_method", default="hard")
args = parser.parse_args()
out_dir = args.output_path or os.path.join("result_f2", "flux2_" + os.path.basename(os.path.normpath(args.lora_ckpt_path)))
os.makedirs(out_dir, exist_ok=True)

VARIANTS = {
    # name: (model, steps)
    "base": ("black-forest-labs/FLUX.2-klein-base-4B", 50),
    "distilled": ("black-forest-labs/FLUX.2-klein-4B", 4),
}

samples = []
for f in sorted(x for x in os.listdir(args.seg_mask_path) if x.endswith(".json")):
    path = os.path.join(args.seg_mask_path, f)
    samples.append((f[:-5], load_seg_map_and_prompt(path.replace(".json", ".png"), path, 1)))

timing = {}
for name, (model, steps) in VARIANTS.items():
    transformer = Flux2RegionalTransformer2DModel.from_pretrained(model, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = Flux2RegionalPipeline.from_pretrained(model, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(os.path.join(args.lora_ckpt_path, "default"), adapter_name="default")
    pipe.load_lora_weights(os.path.join(args.lora_ckpt_path, "cond"), adapter_name="cond")
    pipe.set_adapters(["cond", "default"])
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    times = []
    for stem, batch in samples:
        torch.cuda.synchronize()
        t0 = time.time()
        image = pipe(
            global_prompt=batch["global_caption"], regional_prompts=batch["regional_captions"],
            regional_labels=batch["label"], cond=(batch["cond_pixel_values"][None] + 1) / 2,
            attention_mask_method=args.attention_mask_method, height=batch["image_height"], width=batch["image_width"],
            num_inference_steps=steps, guidance_scale=4.0,
            generator=torch.Generator("cuda").manual_seed(batch["seed"]),
        ).images[0]
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        image.save(os.path.join(out_dir, f"{stem}_{name}.jpg"), quality=95)
    timing[name] = {"steps": steps, "sec_per_image_excl_first": sum(times[1:]) / max(1, len(times) - 1)}
    print(name, timing[name])
    del pipe, transformer
    torch.cuda.empty_cache()

# 一覧: [mask | 元写真 | FLUX.1 | FLUX.2 base | FLUX.2 distilled]
cols = ["mask", "real photo", "FLUX.1-dev (ckpt 3000)", "FLUX.2 klein-base 50step", "FLUX.2 klein 4step"]
s = 256
panel = Image.new("RGB", (s * len(cols), 24 + s * len(samples)), "white")
draw = ImageDraw.Draw(panel)
for j, c in enumerate(cols):
    draw.text((j * s + 6, 6), c, fill="black")
for i, (stem, _) in enumerate(samples):
    num = stem.split("_")[1]
    ims = [
        Image.open(os.path.join(args.seg_mask_path, stem + ".png")).convert("RGB"),
        Image.open(os.path.join(RAMEN_IMAGES, num + ".png")).convert("RGB"),
        Image.open(os.path.join(args.flux1_result, stem + "_0.jpg")).convert("RGB"),
        Image.open(os.path.join(out_dir, stem + "_base.jpg")),
        Image.open(os.path.join(out_dir, stem + "_distilled.jpg")),
    ]
    for j, im in enumerate(ims):
        panel.paste(im.resize((s, s)), (j * s, 24 + i * s))
panel.save(os.path.join(out_dir, "panel.jpg"), quality=90)
json.dump(timing, open(os.path.join(out_dir, "timing.json"), "w"), indent=1)
print("saved", os.path.join(out_dir, "panel.jpg"))
