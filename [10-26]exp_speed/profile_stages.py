"""デモと同じ設定 (蒸留版 + LoRA, 512², cond_scale_factor=1, hard) で処理ごとの時間を測る。
各段の前後で cuda.synchronize して壁時計で測る。val55 の先頭 10 例。"""
import os, sys, time, json, glob, argparse
import torch
sys.path.insert(0, os.path.expanduser("~/Seg2Any"))
from infer import load_seg_map_and_prompt
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=10)
p.add_argument("--steps", type=int, nargs="+", default=[16, 8, 4, 2, 1])
p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "profile_stages.json"))
args = p.parse_args()

BASE = "black-forest-labs/flux-3-image-klein"
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
tr = Flux3RegionalTransformer2DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

T = {}
def timed(name, fn):
    def w(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize(); T.setdefault(name, []).append(time.perf_counter() - t)
        return r
    return w

pipe.encode_all_prompt = timed("text_encode", pipe.encode_all_prompt)
pipe.encode_vae_image = timed("vae_encode", pipe.encode_vae_image)
pipe.decode_latents = timed("vae_decode", pipe.decode_latents)
pipe.prepare_attention_mask = timed("attn_mask", pipe.prepare_attention_mask)
pipe.filter_cond_token = timed("filter_cond", pipe.filter_cond_token)
SEQ = {}
_fwd = pipe.transformer.forward
def fwd(*a, **k):
    SEQ.update(txt=k["encoder_hidden_states"].shape[1], img=k["hidden_states"].shape[1],
               cond=0 if k.get("cond_hidden_states") is None else k["cond_hidden_states"].shape[1])
    return _fwd(*a, **k)
pipe.transformer.forward = timed("transformer_step", fwd)

def run(path, steps):
    b = load_seg_map_and_prompt(path, path[:-4] + ".json", 1)
    g = [torch.Generator("cuda").manual_seed(b["seed"] or 0)]
    torch.cuda.synchronize(); t = time.perf_counter()
    pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
         cond=(b["cond_pixel_values"][None] + 1) / 2.0, attention_mask_method="hard", is_filter_cond_token=True,
         height=b["image_height"], width=b["image_width"], cond_scale_factor=1, num_inference_steps=steps,
         generator=g, max_sequence_length=512, regional_max_sequence_length=50)
    torch.cuda.synchronize()
    return time.perf_counter() - t, len(b["regional_captions"])

paths = sorted(glob.glob(os.path.expanduser("~/Seg2Any/examples/examples_Ramen_val55/*.png")))[: args.n]
run(paths[0], 2); run(paths[0], 2)  # warmup
res = {"gpu": torch.cuda.get_device_name(0), "per_steps": {}}
for steps in args.steps:
    T.clear(); totals = []; seqs = []; nreg = []
    for pth in paths:
        tot, nr = run(pth, steps); totals.append(tot); seqs.append(dict(SEQ)); nreg.append(nr)
    mean = lambda x: sum(x) / len(x)
    r = {"total": mean(totals), **{k: sum(v) / len(paths) for k, v in T.items()},
         "transformer_per_call": mean(T["transformer_step"]), "seq": seqs, "n_regions": nreg}
    r["other"] = r["total"] - sum(r[k] for k in T)
    res["per_steps"][steps] = r
    print(f"steps={steps:2d} total={r['total']:.3f}s  " + "  ".join(f"{k}={r[k]:.3f}" for k in T) +
          f"  other={r['other']:.3f}  per_call={r['transformer_per_call']*1000:.0f}ms", flush=True)
print("seq lens (first 3):", res["per_steps"][args.steps[0]]["seq"][:3])
print("peak mem GiB:", torch.cuda.max_memory_allocated() / 2**30)
json.dump(res, open(args.out, "w"), indent=1)
