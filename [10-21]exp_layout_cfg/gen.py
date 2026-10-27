"""条件を 1 つ指定して inputs の全マスクを生成する。

    cd ~/Seg2Any/"[10-21]exp_layout_cfg" && HF_HOME=/export/ssd3/lan-y/huggingface ../.venv-flux3/bin/python gen.py \
        --name lay7 --s_txt 3.5 --s_lay 7

--old_neg：旧既定 (ネガティブ側にも cond。= テキストだけを強める CFG)。元の Flux3RegionalPipeline.__call__ で生成
--check  ：s_txt = s_lay のとき、元の __call__ (今の既定) と画素一致するかを 1 枚で確かめる
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from infer import load_seg_map_and_prompt  # noqa: E402
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel  # noqa: E402
from pipeline_decoupled import Flux3DecoupledGuidancePipeline  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--inputs", default=os.path.join(HERE, "inputs_val10"))
p.add_argument("--out_root", default=os.path.join(HERE, "results"))
p.add_argument("--model", default="black-forest-labs/flux-3-image-klein-base")
p.add_argument("--lora", default="/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000")
p.add_argument("--steps", type=int, default=20)
p.add_argument("--s_txt", type=float, default=3.5)
p.add_argument("--s_lay", type=float, default=3.5)
p.add_argument("--s_lay_late", type=float, default=None)
p.add_argument("--lay_switch", type=float, default=1.0)
p.add_argument("--old_neg", action="store_true")
p.add_argument("--check", action="store_true")
p.add_argument("--shard", default="0/1", help="i/n：stems を n 分割した i 番目だけ生成 (GPU 並列用)")
args = p.parse_args()

out = os.path.join(args.out_root, args.name)
os.makedirs(out, exist_ok=True)
si, sn = map(int, args.shard.split("/"))
if si == 0:
    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=1)

tr = Flux3RegionalTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3DecoupledGuidancePipeline.from_pretrained(args.model, transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(args.lora, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(args.lora, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda")

stems = sorted(f[:-5] for f in os.listdir(args.inputs) if f.endswith(".json"))
stems = stems[si::sn]
if args.check:
    stems = stems[:1]
times = {}
for s in stems:
    b = load_seg_map_and_prompt(seg_map_path=os.path.join(args.inputs, s + ".png"),
                                seg_anno_path=os.path.join(args.inputs, s + ".json"), cond_scale_factor=1)
    common = dict(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"],
                  regional_labels=b["label"], cond=(b["cond_pixel_values"][None] + 1) / 2.0,
                  height=b["image_height"], width=b["image_width"], num_inference_steps=args.steps)
    gen = lambda: torch.Generator("cuda").manual_seed(b["seed"])  # noqa: E731
    torch.cuda.synchronize()
    t0 = time.time()
    if args.old_neg:
        img = pipe(**common, guidance_scale=args.s_txt, negative_use_cond=True, generator=gen()).images[0]
    else:
        img = pipe.generate(**common, s_txt=args.s_txt, s_lay=args.s_lay, s_lay_late=args.s_lay_late,
                            lay_switch=args.lay_switch, generator=gen()).images[0]
    torch.cuda.synchronize()
    times[s] = round(time.time() - t0, 2)
    img.save(os.path.join(out, s + "_0.jpg"), quality=95)
    print("done", s, times[s], flush=True)
    if args.check:
        ref = pipe(**common, guidance_scale=args.s_txt, negative_use_cond=False, generator=gen()).images[0]
        d = np.abs(np.asarray(img, np.int16) - np.asarray(ref, np.int16))
        print("CHECK max abs pixel diff vs original pipeline:", d.max(), "mean", d.mean())
json.dump(times, open(os.path.join(out, "times.json" if sn == 1 else f"times_{si}.json"), "w"), indent=1)
