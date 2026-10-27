"""UECFOODPIXCOMPLETE test1000 を FLUX.2 版 LoRA で生成し、マスク | GT | 生成 | オーバーレイ のパネルを作る。

    # GPU ごとに shard を分けて生成 (gp36 の A100 = GPU 4,5)
    HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 \
        .venv-flux2/bin/python scripts/gen_uecfoodpix_flux2_test.py --shard 0 --num_shards 2
    # パネルだけ作り直す
    .venv-flux2/bin/python scripts/gen_uecfoodpix_flux2_test.py --panels_only

--plate を付けると皿マスク版 (UECFoodPixPlateFlux2Dataset、global caption あり) で生成する。
入力は学習と同じ Dataset (val 設定) から取るので、caption・マスク・cond 輪郭は学習時と完全に一致する。
生成は学習と同じ 512x512 正方。パネルでは GT の元アスペクトに戻して並べる。
出力: <out>/gen/<stem>.jpg (512² の生成そのもの), <out>/panel/<stem>.jpg
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # リポジトリ直下 (src/ dataset/ utils/ infer.py) を import できるように
import argparse
import colorsys
import os

import numpy as np
from PIL import Image, ImageDraw

from dataset.uecfoodpix_captions import CLASS_ID_TO_NAME, regional_caption
from dataset.uecfoodpix_flux2_dataset import UECFoodPixFlux2Dataset
from dataset import uecfoodpix_plate_captions as plate_vocab
from dataset.uecfoodpix_plate_flux2_dataset import UECFoodPixPlateFlux2Dataset

ROOT = "/export/ssd3/lan-y/UECFOODPIXCOMPLETE"

parser = argparse.ArgumentParser()
parser.add_argument("--lora_ckpt_path", default="/export/ssd3/lan-y/seg2any_ckpt/uecfoodpix_flux2/checkpoint-5500")
parser.add_argument("--pretrained_model_name_or_path", default="black-forest-labs/FLUX.2-klein-base-4B")
parser.add_argument("--output_path", default="result_f2/uecfoodpix_flux2_test1000")
parser.add_argument("--num_inference_steps", type=int, default=20)  # Base 版は 20 step で 50 とほぼ同品質 (result_f2/flux2_steps_sweep)
parser.add_argument("--guidance_scale", type=float, default=4.0)
parser.add_argument("--attention_mask_method", default="hard")  # infer_flux2.py / val55 評価と同じ
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--stems", default=None, help="カンマ区切りの test 画像 stem。指定時はこれだけ生成")
parser.add_argument("--panels_only", action="store_true")
parser.add_argument("--background_prompt", default=None,
                    help="背景 (id 0) の regional caption を差し替える。学習時は \"background\" の 1 語")
parser.add_argument("--plate", action="store_true", help="皿マスク版の LoRA (config/seg2any_uecfoodpix_plate_flux2.yaml) で生成")
parser.add_argument("--overlay_alpha", type=float, default=0.35)
args = parser.parse_args()


def colour(class_id):
    """examples_UECFoodPix/make_examples.py と同じ配色。0 (背景) は黒"""
    if class_id == 0:
        return (0, 0, 0)
    if class_id == plate_vocab.PLATE_ID:
        return (200, 200, 200)  # 皿・器・お盆は薄い灰色
    h = (class_id * 47) % 180 + 10
    s = 200 + (class_id * 13) % 55
    v = 170 + (class_id * 29) % 85
    r, g, b = colorsys.hsv_to_rgb(h / 255.0, s / 255.0, v / 255.0)
    return (int(r * 255), int(g * 255), int(b * 255))


PALETTE = np.array([colour(i) for i in range(256)], dtype=np.uint8)


def make_panel(stem, gen_path, panel_path):
    gt = Image.open(f"{ROOT}/data/UECFoodPIXCOMPLETE/test/img/{stem}.jpg").convert("RGB")
    segm = dataset.read_segm(os.path.join(dataset.segm_root, f"{stem}.png"))
    mask_rgb = Image.fromarray(PALETTE[segm])
    if mask_rgb.size != gt.size:
        mask_rgb = mask_rgb.resize(gt.size, Image.NEAREST)
    gen = Image.open(gen_path).convert("RGB").resize(gt.size, Image.BICUBIC)
    # 背景 (黒) は塗らず、食べ物 (と皿) の領域だけ薄く色を乗せる
    fg = np.array(mask_rgb.resize(gen.size, Image.NEAREST)).any(-1, keepdims=True)
    g = np.asarray(gen, dtype=np.float32)
    overlay = np.where(fg, (1 - args.overlay_alpha) * g + args.overlay_alpha * np.asarray(mask_rgb, np.float32), g)
    overlay = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))

    # 高さ 384 に揃えて横に 4 枚 + 下にクラス名の凡例
    H = 384
    W = round(gt.width * H / gt.height)
    tiles = [im.resize((W, H), Image.NEAREST if im is mask_rgb else Image.BICUBIC) for im in (mask_rgb, gt, gen, overlay)]
    ids = [int(i) for i in np.unique(segm)]
    head, foot = 22, 22
    panel = Image.new("RGB", (W * 4, head + H + foot), "white")
    d = ImageDraw.Draw(panel)
    for i, (t, name) in enumerate(zip(tiles, ["mask", "GT", "FLUX.2 generated", "overlay"])):
        panel.paste(t, (i * W, head))
        d.text((i * W + 4, 5), name, fill="black")
    x = 4
    for i in ids:
        d.rectangle([x, head + H + 5, x + 12, head + H + 17], fill=colour(i), outline="gray")
        label = plate_vocab.regional_caption(i) if args.plate else regional_caption(CLASS_ID_TO_NAME[i])
        d.text((x + 16, head + H + 5), label, fill="black")
        x += 16 + int(d.textlength(label)) + 12
    panel.save(panel_path, quality=92)


dataset = (UECFoodPixPlateFlux2Dataset if args.plate else UECFoodPixFlux2Dataset)(ROOT, "data/test1000.txt", "test")
indices = list(range(len(dataset)))
if args.stems:
    want = args.stems.split(",")
    pos = {os.path.splitext(d["image"])[0]: i for i, d in enumerate(dataset.data)}
    indices = [pos[s] for s in want]
indices = indices[args.shard :: args.num_shards]
if args.limit:
    indices = indices[: args.limit]
gen_dir = os.path.join(args.output_path, "gen")
panel_dir = os.path.join(args.output_path, "panel")
os.makedirs(gen_dir, exist_ok=True)
os.makedirs(panel_dir, exist_ok=True)

if not args.panels_only:
    import torch

    from src.models.transformer_flux2 import Flux2RegionalTransformer2DModel
    from src.pipelines.pipeline_flux2 import Flux2RegionalPipeline

    transformer = Flux2RegionalTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipeline = Flux2RegionalPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=transformer, torch_dtype=torch.bfloat16
    )
    pipeline.load_lora_weights(os.path.join(args.lora_ckpt_path, "default"), adapter_name="default")
    pipeline.load_lora_weights(os.path.join(args.lora_ckpt_path, "cond"), adapter_name="cond")
    pipeline.set_adapters(["cond", "default"])
    pipeline = pipeline.to("cuda")

for n, idx in enumerate(indices):
    stem = os.path.splitext(dataset.data[idx]["image"])[0]
    gen_path = os.path.join(gen_dir, f"{stem}.jpg")
    if not args.panels_only and not os.path.exists(gen_path):
        item = dataset[idx]
        if args.background_prompt:
            item["regional_captions"] = [
                args.background_prompt if c == regional_caption(CLASS_ID_TO_NAME[0]) else c for c in item["regional_captions"]
            ]
        image = pipeline(
            global_prompt=item["global_caption"],  # 皿マスク版は文、旧版は None
            regional_prompts=item["regional_captions"],
            regional_labels=item["label"],
            cond=(item["cond_pixel_values"][None] + 1) / 2.0,  # [-1,1] -> [0,1]
            attention_mask_method=args.attention_mask_method,
            is_filter_cond_token=True,
            height=512,
            width=512,
            cond_scale_factor=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=[torch.Generator("cuda").manual_seed(int(stem))],
        ).images[0]
        image.save(gen_path, quality=95)
    if os.path.exists(gen_path):
        make_panel(stem, gen_path, os.path.join(panel_dir, f"{stem}.jpg"))
    print(f"[{n + 1}/{len(indices)}] {stem}", flush=True)
