"""UECFOODPIXCOMPLETE test を FLUX.3 皿版 LoRA (config/seg2any_uecfoodpix_plate_flux3.yaml) で生成する。

scripts/gen_uecfoodpix_flux2_test.py --plate の FLUX.3 版。入力は学習と同じ Dataset (UECFoodPixPlateFlux3Dataset, val 設定) から取るので、
caption・マスク・cond 輪郭は学習時と一致する。生成は学習と同じ 512x512 正方、seed は stem (FLUX.2 版と同じ)。
パネルは scripts/make_uecfoodpix_flux3_compare.py で作る。

    HF_HOME=/export/ssd2/lan-y/huggingface CUDA_VISIBLE_DEVICES=0 /export/ssd2/lan-y/venvs/Seg2Any/.venv-flux3/bin/python \
        scripts/gen_uecfoodpix_flux3_test.py --stems 16164,3061 --output_path result_f3/UECFOODPIX_flux3/plate_test10_base20
    # 蒸留版 (16 step, CFG なし)
    ... scripts/gen_uecfoodpix_flux3_test.py --pretrained_model_name_or_path black-forest-labs/flux-3-image-klein --num_inference_steps 16 ...
出力: <out>/gen/<stem>.jpg
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # リポジトリ直下 (src/ dataset/ utils/ infer.py) を import できるように
import argparse
import os

import torch

from dataset.uecfoodpix_plate_flux3_dataset import UECFoodPixPlateFlux3Dataset
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline

ROOT = "/export/ssd2/lan-y/UECFOODPIXCOMPLETE"

parser = argparse.ArgumentParser()
parser.add_argument("--lora_ckpt_path", default="/export/space0/lan-y/seg2any_ckpt/uecfoodpix_plate_flux3/checkpoint-5500")
parser.add_argument("--pretrained_model_name_or_path", default="black-forest-labs/flux-3-image-klein-base")
parser.add_argument("--output_path", required=True)
parser.add_argument("--num_inference_steps", type=int, default=20)  # FLUX.2 皿版の生成と同じ。蒸留版は 16
parser.add_argument("--guidance_scale", type=float, default=3.5)  # Base 版の既定。蒸留版では無視される
parser.add_argument("--attention_mask_method", default="base")  # 学習と同じ (FLUX.2 皿版も base の方が自然だった)
parser.add_argument("--cond_steps", type=int, default=None,
                    help="cond (輪郭) を入れるのは最初の N ステップだけ。既定は全ステップ (pipeline_flux3 の cond_steps)")
parser.add_argument("--stems", default=None, help="カンマ区切りの test 画像 stem。指定時はこれだけ生成")
args = parser.parse_args()

dataset = UECFoodPixPlateFlux3Dataset(ROOT, "data/test1000.txt", "test")
indices = list(range(len(dataset)))
if args.stems:
    pos = {os.path.splitext(d["image"])[0]: i for i, d in enumerate(dataset.data)}
    indices = [pos[s] for s in args.stems.split(",")]
gen_dir = os.path.join(args.output_path, "gen")
os.makedirs(gen_dir, exist_ok=True)

transformer = Flux3RegionalTransformer2DModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipeline = Flux3RegionalPipeline.from_pretrained(
    args.pretrained_model_name_or_path, transformer=transformer, torch_dtype=torch.bfloat16
)
pipeline.load_lora_weights(os.path.join(args.lora_ckpt_path, "default"), adapter_name="default")
pipeline.load_lora_weights(os.path.join(args.lora_ckpt_path, "cond"), adapter_name="cond")
pipeline.set_adapters(["cond", "default"])
pipeline = pipeline.to("cuda")

for n, idx in enumerate(indices):
    stem = os.path.splitext(dataset.data[idx]["image"])[0]
    gen_path = os.path.join(gen_dir, f"{stem}.jpg")
    if not os.path.exists(gen_path):
        item = dataset[idx]
        image = pipeline(
            global_prompt=item["global_caption"],
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
            cond_steps=args.cond_steps,
            generator=[torch.Generator("cuda").manual_seed(int(stem))],
        ).images[0]
        image.save(gen_path, quality=95)
    print(f"[{n + 1}/{len(indices)}] {stem}", flush=True)
