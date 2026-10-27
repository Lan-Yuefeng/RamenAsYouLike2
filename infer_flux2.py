"""Seg2Any (FLUX.2 [klein]) の推論。infer.py の FLUX.2 版。入力形式 (seg_mask_path 内の *.png + *.json) は同じ。

    HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 .venv-flux2/bin/python infer_flux2.py \
        --seg_mask_path examples_Ramen --output_path result_f2/flux2_ramen [--lora_ckpt_path ...] [--no_cond]

LoRA なしでも動く (領域アテンションマスクだけの training-free な領域制御になる)。
学習前の cond 分岐は意味のある特徴を作らないので、LoRA なしのときは --no_cond を付ける。
"""
import argparse
import os

import torch

from infer import load_seg_map_and_prompt
from src.models.transformer_flux2 import Flux2RegionalTransformer2DModel
from src.pipelines.pipeline_flux2 import Flux2RegionalPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_name_or_path", default="black-forest-labs/FLUX.2-klein-4B")
parser.add_argument("--lora_ckpt_path", default=None, help="default/ と cond/ を含むディレクトリ")
parser.add_argument("--seg_mask_path", required=True)
parser.add_argument("--output_path", default="./result_f2/flux2")
parser.add_argument("--weight_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
parser.add_argument("--cond_scale_factor", type=int, default=1)
parser.add_argument("--cond2image_attention_weight", type=float, default=1)
parser.add_argument("--no_cond", action="store_true", help="cond トークンを使わない")
parser.add_argument("--attention_mask_method", choices=["hard", "base", "place"], default="hard")
parser.add_argument("--regional_text_position", choices=["restart", "continuous"], default="restart")
parser.add_argument("--regional_max_sequence_length", type=int, default=50)
parser.add_argument("--max_sequence_length", type=int, default=512)
parser.add_argument("--num_images_per_prompt", type=int, default=1)
parser.add_argument("--num_inference_steps", type=int, default=None, help="既定: 蒸留版 4 / Base 版 50")
parser.add_argument("--guidance_scale", type=float, default=4.0, help="Base 版のみ有効")
parser.add_argument("--negative_with_cond", action="store_true",
                    help="Base 版の CFG でネガティブ側にも cond (輪郭) を入れる (2026-10-07 より前の挙動)。"
                         "既定はネガティブ側から外す方で、境界への忠実さが高い (result_f2/val55)")
parser.add_argument("--negative_no_cond", action="store_true", help="(既定になったので不要。互換のため残している)")
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)
weight_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.weight_dtype]

transformer = Flux2RegionalTransformer2DModel.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
)
pipeline = Flux2RegionalPipeline.from_pretrained(
    args.pretrained_model_name_or_path, transformer=transformer, torch_dtype=weight_dtype
)
if args.lora_ckpt_path is not None:
    pipeline.load_lora_weights(os.path.join(args.lora_ckpt_path, "default"), adapter_name="default")
    pipeline.set_adapters("default")
    if os.path.exists(os.path.join(args.lora_ckpt_path, "cond")):
        pipeline.load_lora_weights(os.path.join(args.lora_ckpt_path, "cond"), adapter_name="cond")
        pipeline.set_adapters(["cond", "default"])
pipeline = pipeline.to("cuda")

num_inference_steps = args.num_inference_steps or (4 if pipeline.config.is_distilled else 50)

anno_files = sorted(os.path.join(args.seg_mask_path, f) for f in os.listdir(args.seg_mask_path) if f.endswith(".json"))
for anno_file in anno_files:
    batch = load_seg_map_and_prompt(
        seg_map_path=anno_file.replace(".json", ".png"),
        seg_anno_path=anno_file,
        cond_scale_factor=args.cond_scale_factor,
    )
    generator = None
    if batch["seed"] is not None:
        generator = [torch.Generator("cuda").manual_seed(batch["seed"] + i) for i in range(args.num_images_per_prompt)]

    images = pipeline(
        global_prompt=batch["global_caption"],
        regional_prompts=batch["regional_captions"],
        regional_labels=batch["label"],
        cond=None if args.no_cond else (batch["cond_pixel_values"][None] + 1) / 2.0,  # [-1,1] -> [0,1]
        attention_mask_method=args.attention_mask_method,
        is_filter_cond_token=True,
        cond2image_attention_weight=args.cond2image_attention_weight,
        regional_text_position=args.regional_text_position,
        height=batch["image_height"],
        width=batch["image_width"],
        cond_scale_factor=args.cond_scale_factor,
        num_images_per_prompt=args.num_images_per_prompt,
        guidance_scale=args.guidance_scale,
        negative_use_cond=args.negative_with_cond,
        num_inference_steps=num_inference_steps,
        generator=generator,
        max_sequence_length=args.max_sequence_length,
        regional_max_sequence_length=args.regional_max_sequence_length,
    ).images

    for i, image in enumerate(images):
        image.save(os.path.join(args.output_path, os.path.splitext(batch["image_name"])[0] + f"_{i}.jpg"), quality=95)
    print("done", batch["image_name"])
