"""FLUX.2 [klein] 4B の素の推論が新 venv で動くかの確認用 (Seg2Any 移植の Step 1)。

    HF_HOME=/export/ssd3/lan-y/huggingface .venv-flux2/bin/python flux2_smoke.py
"""
import argparse
import time

import torch
from diffusers import Flux2KleinPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
parser.add_argument("--prompt", default="a bowl of tonkotsu ramen with chashu, ajitama egg, green onions and nori, top view, food photography")
parser.add_argument("--size", type=int, default=1024)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="result_f2/flux2_smoke.png")
args = parser.parse_args()

t0 = time.time()
pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
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
    ).images[0]
    torch.cuda.synchronize()
    print(f"run{i}: {time.time() - t0:.2f}s")

print(f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
image.save(args.out)
print("saved", args.out)
