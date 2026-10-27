"""「消し残しの卵」仮説の検証。val10 の ell2.5_noegg (丸いチャーシュー、卵なし) に対して:
  sliver_cap     チャーシューの右下に卵 (whole) の細い切れ端 (~150-250px) を置き、global caption に "a boiled egg" を足す (デモの自動 caption と同じ状態)
  cap_only       切れ端なし、global caption に "a boiled egg" だけ足す
  sliver_nocap   切れ端あり、global caption は卵なし
比較は results/ell2.5_noegg。seed は各 json の seed。"""
import json, os, sys
import numpy as np, cv2, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from infer import load_seg_map_and_prompt
from dataset.ramen_captions import CLASS_COLORS, REGION_TEXT
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline
INP = os.path.join(HERE, "inputs"); OUT = os.path.join(HERE, "results")
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
MEAT, EGG = np.array(CLASS_COLORS["meat"]), np.array(CLASS_COLORS["egg"])
stems = sorted(f[:-9] for f in os.listdir(INP) if f.endswith("_orig.png") and f.startswith("ramen_"))
for s in stems:
    m = np.array(Image.open(os.path.join(INP, f"{s}_ell2.5_noegg.png")).convert("RGB")); a = json.load(open(os.path.join(INP, f"{s}_ell2.5_noegg.json")))
    num, lab, st, cen = cv2.connectedComponentsWithStats((m == MEAT).all(-1).astype(np.uint8))
    i = 1 + st[1:, cv2.CC_STAT_AREA].argmax(); ring = cv2.dilate((lab == i).astype(np.uint8), np.ones((9, 9), np.uint8)) & (lab != i)
    yy, xx = np.mgrid[:512, :512]; cx, cy = cen[i]
    ang = np.degrees(np.arctan2(yy - cy, xx - cx))
    sl = ring.astype(bool) & (ang > 20) & (ang < 70)
    c = m.copy(); c[sl] = EGG
    cap_egg = a["caption"].replace("topped with ", "topped with a boiled egg, ")
    def put(tag, img, cap):
        Image.fromarray(img).save(os.path.join(INP, f"{s}_{tag}.png")); b = json.loads(json.dumps(a)); b["caption"] = cap
        if (img == EGG).all(-1).any(): b["segments_info"].append({"color": EGG.tolist(), "text": REGION_TEXT["egg"]})
        json.dump(b, open(os.path.join(INP, f"{s}_{tag}.json"), "w"), indent=1)
    put("sliver_cap", c, cap_egg); put("cap_only", m, cap_egg); put("sliver_nocap", c, a["caption"])
    print(s, "sliver px", sl.sum(), "| has 'topped with':", "topped with " in a["caption"])
tr = Flux3RegionalTransformer2DModel.from_pretrained("black-forest-labs/flux-3-image-klein", subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained("black-forest-labs/flux-3-image-klein", transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"]); pipe = pipe.to("cuda")
for name in ["sliver_cap", "cap_only", "sliver_nocap"]:
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    for s in stems:
        b = load_seg_map_and_prompt(os.path.join(INP, f"{s}_{name}.png"), os.path.join(INP, f"{s}_{name}.json"), 1)
        img = pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
                   cond=(b["cond_pixel_values"][None] + 1) / 2.0, attention_mask_method="hard", is_filter_cond_token=True,
                   height=512, width=512, cond_scale_factor=1, num_inference_steps=16,
                   generator=torch.Generator("cuda").manual_seed(b["seed"]),
                   max_sequence_length=512, regional_max_sequence_length=50).images[0]
        img.save(os.path.join(OUT, name, s + ".jpg"), quality=95)
    print("done", name, flush=True)
