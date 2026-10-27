"""予備実験 (学習なし): 後半 joint blocks でテキスト / cond が画像を見ないようにしたら、今の重みでどれだけ崩れるか。
マスクの書き換えだけで「一方向」を再現する (速度はまだ変わらない。品質の確認用)。
variant:
  oneway_txt   : txt の行 -> img の列を遮断
  oneway_cond  : cond の行 -> img の列を遮断
  oneway_both  : 両方
  oneway_both_fixtemb : 両方 + txt / cond の modulation をステップに依らない定数 (飛び先 timestep=0) に固定
mode blocks はもともとモード内だけなので影響しない。蒸留版 + LoRA、4 ステップ、val10。"""
import os, sys, glob
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/Seg2Any"))
from infer import load_seg_map_and_prompt
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline

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

state = {"txt": False, "cond": False, "fixtemb": False}
_prep = pipe.prepare_attention_mask
def prep(txt_seq_lens, **kw):
    masks = _prep(txt_seq_lens=txt_seq_lens, **kw)
    n_txt = sum(txt_seq_lens[0]); n_img = kw["height"] * kw["width"]
    out = []
    for m in masks:
        if m is None:
            out.append(None); continue
        m = m[:, :1].clone()  # 全 head 共通なので 1 枚だけ書き換えて expand し直す
        if state["txt"]:
            m[:, :, :n_txt, n_txt:n_txt + n_img] = float("-inf")
        if state["cond"]:
            m[:, :, n_txt + n_img:, n_txt:n_txt + n_img] = float("-inf")
        out.append(m.expand(-1, kw["num_attention_heads"], -1, -1))
    return tuple(out)
pipe.prepare_attention_mask = prep

# txt / cond の modulation 入力を silu(temb(0) + target_temb(0)) に差し替える
@torch.no_grad()
def const_act(x):
    z = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    t = tr.timestep_embedder(tr.time_proj(z).to(x.dtype)) + tr.target_timestep_embedder(tr.time_proj(z).to(x.dtype))
    return F.silu(t)[:, None, :]
def pre_hook(mod, args):
    return (const_act(args[0]),) if state["fixtemb"] else None
for name in ("txt", "image_cond"):
    tr.early_stream_modulations[name].register_forward_pre_hook(pre_hook)
    tr.single_stream_modulations[name].register_forward_pre_hook(pre_hook)

VARIANTS = {
    "oneway_txt": dict(txt=True, cond=False, fixtemb=False),
    "oneway_cond": dict(txt=False, cond=True, fixtemb=False),
    "oneway_both": dict(txt=True, cond=True, fixtemb=False),
    "oneway_both_fixtemb": dict(txt=True, cond=True, fixtemb=True),
}
paths = sorted(glob.glob(os.path.join(glob.escape(D), "val10", "*.png")))
for name, v in VARIANTS.items():
    state.update(v)
    out = os.path.join(D, "result", f"probe_{name}"); os.makedirs(out, exist_ok=True)
    for path in paths:
        b = load_seg_map_and_prompt(path, path[:-4] + ".json", 1)
        g = [torch.Generator("cuda").manual_seed(b["seed"] or 0)]
        im = pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
                  cond=(b["cond_pixel_values"][None] + 1) / 2.0, attention_mask_method="hard", is_filter_cond_token=True,
                  height=512, width=512, cond_scale_factor=1, num_inference_steps=STEPS, generator=g,
                  max_sequence_length=512, regional_max_sequence_length=50).images[0]
        im.save(os.path.join(out, os.path.basename(path)[:-4] + "_0.jpg"), quality=95)
    print("done", name, flush=True)
