"""自己蒸留した LoRA で val10 を生成する。推論はデモに入れる形 (① default LoRA 融合 + ③ txt / cond の K/V キャッシュ)、
4 ステップ、512²。出力: result/<name>/<stem>_0.jpg と result/<name>/timing.json
使い方: python gen_val10.py <name> <lora_dir> [<name> <lora_dir> ...]"""
import os, sys, glob, json, time
import torch
sys.path.insert(0, os.path.expanduser("~/Seg2Any"))
sys.path.insert(0, os.path.expanduser("~/Seg2Any/[10-26]exp_speed"))
from infer import load_seg_map_and_prompt
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline
import src.models.transformer_flux3 as T
from cache_flux3 import CachedForward, fuse_default_lora

D = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.expanduser("~/Seg2Any/[10-26]exp_speed/val10")
BASE = "black-forest-labs/flux-3-image-klein"
jobs = list(zip(sys.argv[1::2], sys.argv[2::2]))
paths = sorted(glob.glob(os.path.join(glob.escape(VAL), "*.png")))
batches = [load_seg_map_and_prompt(p, p[:-4] + ".json", 1) for p in paths]
orig_select = T.select_lora

for name, lora in jobs:
    T.select_lora = orig_select  # 前のジョブの融合を引きずらないよう毎回読み直す
    tr = Flux3RegionalTransformer2DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = Flux3RegionalPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(os.path.join(lora, "default"), adapter_name="default")
    pipe.load_lora_weights(os.path.join(lora, "cond"), adapter_name="cond")
    pipe.set_adapters(["cond", "default"])
    pipe = pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
    fuse_default_lora(pipe.transformer)
    cf = CachedForward(pipe.transformer, use_cache=True, cache_txt=True)
    pipe.transformer.forward = cf
    out = os.path.join(D, "result", name); os.makedirs(out, exist_ok=True)

    def gen(b):
        cf.reset()
        g = [torch.Generator("cuda").manual_seed(b["seed"] or 0)]
        return pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
                    cond=(b["cond_pixel_values"][None] + 1) / 2.0, attention_mask_method="hard", is_filter_cond_token=True,
                    height=512, width=512, cond_scale_factor=1, num_inference_steps=4, generator=g,
                    max_sequence_length=512, regional_max_sequence_length=50).images[0]

    gen(batches[0])  # warmup
    ts = []
    for p, b in zip(paths, batches):
        torch.cuda.synchronize(); t = time.perf_counter()
        im = gen(b)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        im.save(os.path.join(out, os.path.basename(p)[:-4] + "_0.jpg"), quality=95)
    json.dump({"lora": lora, "gpu": torch.cuda.get_device_name(), "sec_per_image": sum(ts) / len(ts)},
              open(os.path.join(out, "timing.json"), "w"), indent=1)
    print("done", name, f"{sum(ts) / len(ts):.3f}s/img", flush=True)
    del pipe, tr, cf
    torch.cuda.empty_cache()
