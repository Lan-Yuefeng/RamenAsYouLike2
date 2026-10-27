"""cache_flux3.py の検算。同じ GPU・同じ入力で transformer 1 回の出力を比べる。
probe 方式 (元の forward + マスク書換え + modulation 固定フック) と CachedForward(use_cache=False) が一致するか、
4 ステップ全体で キャッシュあり / なし が一致するか、LoRA 融合の前後で一致するか。"""
import os, sys, glob
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/Seg2Any"))
from infer import load_seg_map_and_prompt
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline
from cache_flux3 import CachedForward, fuse_default_lora, _oneway

D = os.path.dirname(os.path.abspath(__file__))
BASE = "black-forest-labs/flux-3-image-klein"
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
tr = Flux3RegionalTransformer2DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
tr = pipe.transformer
orig_forward = tr.forward

# 4 ステップ分の transformer 入力を記録 (base で生成したときのもの)
rec = []
def grab(*a, **k):
    rec.append({kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in k.items()})
    return orig_forward(*a, **k)
path = sorted(glob.glob(os.path.join(glob.escape(D), "val10", "*.png")))[1]
b = load_seg_map_and_prompt(path, path[:-4] + ".json", 1)
tr.forward = grab
pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
     cond=(b["cond_pixel_values"][None] + 1) / 2.0, attention_mask_method="hard", is_filter_cond_token=True,
     height=512, width=512, num_inference_steps=4, generator=[torch.Generator("cuda").manual_seed(0)],
     max_sequence_length=512, regional_max_sequence_length=50)
tr.forward = orig_forward
n_txt, n_img = rec[0]["encoder_hidden_states"].shape[1], rec[0]["hidden_states"].shape[1]

# probe 方式
def const_act(x):
    z = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    t = tr.timestep_embedder(tr.time_proj(z).to(x.dtype)) + tr.target_timestep_embedder(tr.time_proj(z).to(x.dtype))
    return F.silu(t)[:, None, :]
hooks = []
def probe_on(cache_txt):
    for name in (("txt", "image_cond") if cache_txt else ("image_cond",)):
        for mods in (tr.early_stream_modulations, tr.single_stream_modulations):
            hooks.append(mods[name].register_forward_pre_hook(lambda m, a: (const_act(a[0]),)))
def probe_off():
    for h in hooks: h.remove()
    hooks.clear()
def probe_kw(k, cache_txt):
    j = k["joint_attention_kwargs"]
    ow = {kk: _oneway(j[kk], n_txt, n_img, cache_txt) for kk in ("attention_mask", "hard_attention_mask")}
    return {**k, "joint_attention_kwargs": {**j, **ow}}

def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())

@torch.no_grad()
def compare(tag, cache_txt=True):
    probe_on(cache_txt)
    ref = [orig_forward(**probe_kw(k, cache_txt))[0] for k in rec]
    probe_off()
    nocache = CachedForward(tr, use_cache=False, cache_txt=cache_txt)
    mine = [nocache(**k)[0] for k in rec]
    cache = CachedForward(tr, use_cache=True, cache_txt=cache_txt)
    cached = [cache(**k)[0] for k in rec]
    for i in range(4):
        print(f"[{tag}] step{i}  probe vs mine(no cache) {rel(mine[i], ref[i]):.2e}   "
              f"cache vs no cache {rel(cached[i], mine[i]):.2e}", flush=True)
    return ref, cached

with torch.no_grad():
    base0 = [orig_forward(**k)[0] for k in rec]
ref_u, cached_u = compare("unfused")
compare("unfused cond only", cache_txt=False)
print("fused modules:", fuse_default_lora(tr))
with torch.no_grad():
    base1 = [orig_forward(**k)[0] for k in rec]
ref_f, cached_f = compare("fused")
compare("fused cond only", cache_txt=False)
for i in range(4):
    print(f"step{i}  base fused vs unfused {rel(base1[i], base0[i]):.2e}   oneway probe fused vs unfused "
          f"{rel(ref_f[i], ref_u[i]):.2e}   cache fused vs unfused {rel(cached_f[i], cached_u[i]):.2e}")
