"""蒸留版 + LoRA でステップ数だけ変えて val10 を生成 (デモと同じ設定)。result/steps<N>/<stem>_0.jpg"""
import os, sys, glob
import torch
sys.path.insert(0, os.path.expanduser("~/Seg2Any"))
from infer import load_seg_map_and_prompt
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline
D = os.path.dirname(os.path.abspath(__file__))
BASE = "black-forest-labs/flux-3-image-klein"
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
tr = Flux3RegionalTransformer2DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
for steps in [16, 8, 4, 2, 1]:
    out = os.path.join(D, "result", f"steps{steps}"); os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(glob.escape(D), "val10", "*.png"))):
        b = load_seg_map_and_prompt(path, path[:-4] + ".json", 1)
        g = [torch.Generator("cuda").manual_seed(b["seed"] or 0)]
        im = pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
                  cond=(b["cond_pixel_values"][None] + 1) / 2.0, attention_mask_method="hard", is_filter_cond_token=True,
                  height=512, width=512, cond_scale_factor=1, num_inference_steps=steps, generator=g,
                  max_sequence_length=512, regional_max_sequence_length=50).images[0]
        im.save(os.path.join(out, os.path.basename(path)[:-4] + "_0.jpg"), quality=95)
    print("done", steps, flush=True)
