"""Seg2Any の部品を推論時に 1 つずつ外して生成する (FLUX.3 Base + ramen LoRA)。

    cd ~/Seg2Any/"[10-25]exp_ablation" && HF_HOME=/export/ssd3/lan-y/huggingface \
        ../.venv-flux3/bin/python gen_ablation.py --name no_cond --no_cond

--mask_method   hard (SAA + AIA、推論の既定) / base (SAA のみ。ramen LoRA の学習時の設定)
--hard_range    AIA を掛けるブロック (両端含む)。既定 None = [5, 18]
--no_cond       輪郭 cond トークンを入れない
--no_lora       LoRA を載せない (素の Base 版 + 領域アテンションマスクだけ)。cond も自動で外す
--inputs        regional caption を変えた入力 (make_inputs.py) を渡すと、caption の効き方を調べられる
CFG のネガティブ側には cond を入れない (今の既定)。
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from infer import load_seg_map_and_prompt  # noqa: E402
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel  # noqa: E402
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--inputs", default=os.path.join(HERE, "inputs_val10"))
p.add_argument("--out_root", default=os.path.join(HERE, "results"))
p.add_argument("--model", default="black-forest-labs/flux-3-image-klein-base")
p.add_argument("--lora", default="/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000")
p.add_argument("--steps", type=int, default=50)
p.add_argument("--cfg", type=float, default=3.5)
p.add_argument("--mask_method", default="hard", choices=["hard", "base"])
p.add_argument("--hard_range", type=int, nargs=2, default=None)
p.add_argument("--no_cond", action="store_true")
p.add_argument("--no_lora", action="store_true")
args = p.parse_args()
if args.no_lora:
    args.no_cond = True  # 学習前の cond 分岐は意味のある特徴を作らない

out = os.path.join(args.out_root, args.name)
os.makedirs(out, exist_ok=True)
json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=1)

tr = Flux3RegionalTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained(args.model, transformer=tr, torch_dtype=torch.bfloat16)
if not args.no_lora:
    pipe.load_lora_weights(os.path.join(args.lora, "default"), adapter_name="default")
    pipe.load_lora_weights(os.path.join(args.lora, "cond"), adapter_name="cond")
    pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda")

stems = sorted(f[:-5] for f in os.listdir(args.inputs) if f.endswith(".json"))
times = {}
for s in stems:
    b = load_seg_map_and_prompt(seg_map_path=os.path.join(args.inputs, s + ".png"),
                                seg_anno_path=os.path.join(args.inputs, s + ".json"), cond_scale_factor=1)
    torch.cuda.synchronize()
    t0 = time.time()
    img = pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
               cond=None if args.no_cond else (b["cond_pixel_values"][None] + 1) / 2.0,
               attention_mask_method=args.mask_method, hard_attn_block_range=args.hard_range,
               height=b["image_height"], width=b["image_width"], num_inference_steps=args.steps,
               guidance_scale=args.cfg, negative_use_cond=False,
               generator=torch.Generator("cuda").manual_seed(b["seed"])).images[0]
    torch.cuda.synchronize()
    times[s] = round(time.time() - t0, 2)
    img.save(os.path.join(out, s + "_0.jpg"), quality=95)
    print("done", s, times[s], flush=True)
json.dump(times, open(os.path.join(out, "times.json"), "w"), indent=1)
