"""意味付き cond トークン版の LoRA で生成する。条件は ../[10-25]exp_ablation/gen_ablation.py の full_hard と同じ
(FLUX.3 Base、512²、50 ステップ、CFG 3.5、ネガティブ側に cond なし、hard マスク)。

    cd ~/Seg2Any/"[10-25]exp_semcond" && HF_HOME=/export/ssd3/lan-y/huggingface ../.venv-flux3/bin/python gen_semcond.py \
        --name semcond_full --ckpt /export/space0/lan-y/seg2any_ckpt/ramen_flux3_semcond/checkpoint-3000
--inputs に ../[10-25]exp_ablation/inputs_val10_classname などを渡すと、regional caption を変えたときの崩れ方を調べられる
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from infer import load_seg_map_and_prompt  # noqa: E402
from semcond import load_sem_proj, semcond_for_pipeline_call  # noqa: E402
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel  # noqa: E402
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--ckpt", required=True, help="default/ cond/ cond_sem_proj.safetensors を含む checkpoint")
p.add_argument("--inputs", default=os.path.join(os.path.dirname(HERE), "[10-25]exp_ablation/inputs_val10"))
p.add_argument("--out_root", default=os.path.join(HERE, "results"))
p.add_argument("--model", default="black-forest-labs/flux-3-image-klein-base")
p.add_argument("--steps", type=int, default=50)
p.add_argument("--cfg", type=float, default=3.5)
p.add_argument("--mask_method", default="hard", choices=["hard", "base"])
p.add_argument("--no_sem", action="store_true", help="W を読まずに生成 (学習した LoRA だけの効果を見る)")
args = p.parse_args()

out = os.path.join(args.out_root, args.name)
os.makedirs(out, exist_ok=True)
json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=1)

tr = Flux3RegionalTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained(args.model, transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(args.ckpt, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(args.ckpt, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda")
if not args.no_sem:
    load_sem_proj(tr, args.ckpt)

stems = sorted(f[:-5] for f in os.listdir(args.inputs) if f.endswith(".json"))
times = {}
for s in stems:
    b = load_seg_map_and_prompt(seg_map_path=os.path.join(args.inputs, s + ".png"),
                                seg_anno_path=os.path.join(args.inputs, s + ".json"), cond_scale_factor=1)
    cond = (b["cond_pixel_values"][None] + 1) / 2.0
    torch.cuda.synchronize()
    t0 = time.time()
    if not args.no_sem:
        tr._cond_sem = semcond_for_pipeline_call(pipe, b["global_caption"], b["regional_captions"], b["label"], cond,
                                                 b["image_height"], b["image_width"])
    img = pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
               cond=cond, attention_mask_method=args.mask_method, height=b["image_height"], width=b["image_width"],
               num_inference_steps=args.steps, guidance_scale=args.cfg, negative_use_cond=False,
               generator=torch.Generator("cuda").manual_seed(b["seed"])).images[0]
    torch.cuda.synchronize()
    times[s] = round(time.time() - t0, 2)
    img.save(os.path.join(out, s + "_0.jpg"), quality=95)
    print("done", s, times[s], flush=True)
json.dump(times, open(os.path.join(out, "times.json"), "w"), indent=1)
