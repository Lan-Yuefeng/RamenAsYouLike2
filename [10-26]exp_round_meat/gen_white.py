"""「white」仮説: チャーシューの regional caption "browned white surface" と卵の "glossy white surface" が似ているせいで、
丸いチャーシューが卵になるのでは。卵っぽくなった 2 例 (ramen_0014 の楕円 2.5%, デモの 027) を 6 seed で、
caption そのまま / "white" を消した / 丸い形を言葉で補った の 3 通りで生成する。"""
import json, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from infer import load_seg_map_and_prompt
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline
INP = os.path.join(HERE, "inputs"); OUT = os.path.join(HERE, "results_white")
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
ORIG = "A slice of chashu pork with a soft fat rim and browned white surface."
CAPS = {"orig": ORIG,
        "nowhite": "A slice of chashu pork with a soft fat rim and browned surface.",
        "rolled": "A round slice of rolled chashu pork, pink meat with a spiral of fat and a browned edge."}
CASES = ["ramen_0014_ell2.5", "demo027"]
SEEDS = [1000, 1001, 1002, 1003, 1004, 1005]
tr = Flux3RegionalTransformer2DModel.from_pretrained("black-forest-labs/flux-3-image-klein", subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained("black-forest-labs/flux-3-image-klein", transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"]); pipe = pipe.to("cuda")
for case in CASES:
    b = load_seg_map_and_prompt(os.path.join(INP, case + ".png"), os.path.join(INP, case + ".json"), 1)
    assert ORIG in b["regional_captions"]
    for cn, cap in CAPS.items():
        os.makedirs(os.path.join(OUT, cn), exist_ok=True)
        rc = [cap if c == ORIG else c for c in b["regional_captions"]]
        for sd in SEEDS:
            img = pipe(global_prompt=b["global_caption"], regional_prompts=rc, regional_labels=b["label"],
                       cond=(b["cond_pixel_values"][None] + 1) / 2.0, attention_mask_method="hard", is_filter_cond_token=True,
                       height=512, width=512, cond_scale_factor=1, num_inference_steps=16,
                       generator=torch.Generator("cuda").manual_seed(sd),
                       max_sequence_length=512, regional_max_sequence_length=50).images[0]
            img.save(os.path.join(OUT, cn, f"{case}_{sd}.jpg"), quality=95)
        print("done", case, cn, flush=True)
