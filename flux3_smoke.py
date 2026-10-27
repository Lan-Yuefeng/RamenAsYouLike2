"""FLUX 3 Image Klein の素の推論が新 venv で動くかの確認用 (Seg2Any 移植の Step 1)。

    HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 .venv-flux3/bin/python flux3_smoke.py
    HF_HOME=... .venv-flux3/bin/python flux3_smoke.py --model black-forest-labs/flux-3-image-klein-base --steps 50 --out result_f3/flux3_smoke/flux3_smoke_base.png
"""
import argparse
import time

import torch
from diffusers import ModularPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="black-forest-labs/flux-3-image-klein")
parser.add_argument("--prompt", default="a bowl of tonkotsu ramen with chashu, ajitama egg, green onions and nori, top view, food photography")
parser.add_argument("--size", type=int, default=1024)
parser.add_argument("--steps", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="result_f3/flux3_smoke/flux3_smoke.png")
args = parser.parse_args()

t0 = time.time()
pipe = ModularPipeline.from_pretrained(args.model)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")
print(f"load: {time.time() - t0:.1f}s, mem {torch.cuda.memory_allocated() / 2**30:.1f}GiB")

for i in range(2):  # 1回目はウォームアップ
    torch.cuda.synchronize()
    t0 = time.time()
    image = pipe(
        prompt=args.prompt,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        output="images",
    )[0]
    torch.cuda.synchronize()
    print(f"run{i}: {time.time() - t0:.2f}s")

print(f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
image.save(args.out)
print("saved", args.out)
