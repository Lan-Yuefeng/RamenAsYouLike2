"""transformer 1 回の時間を条件別に測る: LoRA 有無 / 領域マスク有無 / cond 有無。1 例 (ramen val55 の 2 番目, txt640 img1024 cond434)。"""
import os, sys, time, glob
import torch
sys.path.insert(0, os.path.expanduser("~/Seg2Any"))
from infer import load_seg_map_and_prompt
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline

BASE = "black-forest-labs/flux-3-image-klein"
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
tr = Flux3RegionalTransformer2DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)

captured = {}
_fwd = pipe.transformer.forward
def grab(*a, **k):
    captured.setdefault("k", k); return _fwd(*a, **k)
pipe.transformer.forward = grab
path = sorted(glob.glob(os.path.expanduser("~/Seg2Any/examples/examples_Ramen_val55/*.png")))[1]
b = load_seg_map_and_prompt(path, path[:-4] + ".json", 1)
pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
     cond=(b["cond_pixel_values"][None] + 1) / 2.0, height=512, width=512, num_inference_steps=1)
pipe.transformer.forward = _fwd
K = captured["k"]
am = K["joint_attention_kwargs"]["attention_mask"]
print("mask", tuple(am.shape), am.dtype, "hard", None if K["joint_attention_kwargs"]["hard_attention_mask"] is None else tuple(K["joint_attention_kwargs"]["hard_attention_mask"].shape))

@torch.no_grad()
def bench(kw, n=10):
    for _ in range(3): pipe.transformer(**kw)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): pipe.transformer(**kw)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

nomask = {**K, "joint_attention_kwargs": {**K["joint_attention_kwargs"], "attention_mask": None, "hard_attention_mask": None}}
nocond = {**nomask, "cond_hidden_states": None, "cond_ids": None}
print(f"LoRA on , mask on , cond on : {bench(K):.0f} ms")
print(f"LoRA on , mask off, cond on : {bench(nomask):.0f} ms")
pipe.transformer.disable_adapters() if hasattr(pipe.transformer, "disable_adapters") else pipe.disable_lora()
print(f"LoRA off, mask on , cond on : {bench(K):.0f} ms")
print(f"LoRA off, mask off, cond on : {bench(nomask):.0f} ms")
print(f"LoRA off, mask off, cond off: {bench(nocond):.0f} ms")
