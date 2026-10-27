"""cache_flux3.py の ①③ を学習なしで試す。蒸留版 + LoRA、4 ステップ、512²、val10 (デモと同じ設定)。
条件 (実行順。① は重みを書き換えるので後半にまとめる):
  base        : 今のまま
  oneway_ref  : 一方向化 + 定数 modulation を毎ステップ全部計算 (③ の検算用。probe の oneway_both_fixtemb 相当)
  cache       : ③ (K/V キャッシュ)、LoRA 未融合
  fuse        : ① だけ
  fuse_cache  : ① + ③
出力: result/cache_<name>/<stem>_0.jpg、cache_timing.json (1 枚の時間と transformer 1 回の時間、画像どうしの PSNR)"""
import os, sys, glob, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/Seg2Any"))
from infer import load_seg_map_and_prompt
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline
from cache_flux3 import CachedForward, fuse_default_lora

D = os.path.dirname(os.path.abspath(__file__))
BASE = "black-forest-labs/flux-3-image-klein"
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
STEPS = 4
tr = Flux3RegionalTransformer2DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
tr = pipe.transformer
orig_forward = tr.forward

# transformer 1 回ごとの時間 (同期して測る)
calls = []
def timed(fn):
    def f(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize(); calls.append(time.perf_counter() - t)
        return out
    return f

paths = sorted(glob.glob(os.path.join(glob.escape(D), "val10", "*.png")))
batches = [load_seg_map_and_prompt(p, p[:-4] + ".json", 1) for p in paths]

def gen(b, cf=None):
    if cf is not None:
        cf.reset()
    g = [torch.Generator("cuda").manual_seed(b["seed"] or 0)]
    return pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
                cond=(b["cond_pixel_values"][None] + 1) / 2.0, attention_mask_method="hard", is_filter_cond_token=True,
                height=512, width=512, cond_scale_factor=1, num_inference_steps=STEPS, generator=g,
                max_sequence_length=512, regional_max_sequence_length=50).images[0]

imgs, timing = {}, {}
def run(name, cf=None):
    tr.forward = timed(cf if cf is not None else orig_forward)
    gen(batches[0], cf); gen(batches[0], cf)  # warmup
    out = os.path.join(D, "result", f"cache_{name}"); os.makedirs(out, exist_ok=True)
    totals, per_call = [], []
    imgs[name] = []
    for p, b in zip(paths, batches):
        calls.clear()
        torch.cuda.synchronize(); t = time.perf_counter()
        im = gen(b, cf)
        torch.cuda.synchronize(); totals.append(time.perf_counter() - t)
        per_call.append(list(calls))
        im.save(os.path.join(out, os.path.basename(p)[:-4] + "_0.jpg"), quality=95)
        imgs[name].append(np.asarray(im).astype(np.float64))
    pc = np.array(per_call)
    timing[name] = {"total_s": float(np.mean(totals)), "transformer_s": float(pc.sum(1).mean()),
                    "call_ms_by_step": [float(x) for x in pc.mean(0) * 1000]}
    print(name, json.dumps(timing[name]), flush=True)

def psnr(a, b):
    return float(np.mean([10 * np.log10(255 ** 2 / max(np.mean((x - y) ** 2), 1e-10)) for x, y in zip(imgs[a], imgs[b])]))

run("base")
run("oneway_ref", CachedForward(tr, use_cache=False))
run("cache", CachedForward(tr, use_cache=True))
print("fused modules:", fuse_default_lora(tr), flush=True)
run("fuse")
run("fuse_cache", CachedForward(tr, use_cache=True))

# probe (pipeline 本体 + マスク書換え) の画像と、自前の一方向化実装が一致するか
probe = [np.asarray(__import__("PIL.Image").Image.open(
    os.path.join(D, "result", "probe_oneway_both_fixtemb", os.path.basename(p)[:-4] + "_0.jpg"))).astype(np.float64)
    for p in paths]
imgs["probe_jpg"] = probe
timing["psnr"] = {
    "fuse_vs_base": psnr("fuse", "base"),
    "cache_vs_oneway_ref": psnr("cache", "oneway_ref"),
    "fuse_cache_vs_oneway_ref": psnr("fuse_cache", "oneway_ref"),
    "oneway_ref_vs_probe_jpg": psnr("oneway_ref", "probe_jpg"),
    "oneway_ref_vs_base": psnr("oneway_ref", "base"),
}
timing["gpu"] = torch.cuda.get_device_name()
print(json.dumps(timing["psnr"], indent=1))
json.dump(timing, open(os.path.join(D, f"cache_timing_{timing['gpu'].split()[-1]}.json"), "w"), indent=1)
