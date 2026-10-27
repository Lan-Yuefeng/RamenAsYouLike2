"""チャーシューを丸く描くと卵になる問題の再現と原因の切り分け (FLUX.3 klein 蒸留版 + ramen LoRA、デモと同じ設定)。

val55 のうちチャーシューを含む先頭 10 枚で、一番大きいチャーシュー領域を同じ面積の円に置き換える。
条件:
  orig            元のマスク
  circle          円 (輪郭 cond も領域マスクも円)
  circle_noegg    円 + 卵/半熟卵をスープに塗り替え、global caption からも卵を消す (卵の言葉が画像内にない)
  condC_labO      輪郭 cond だけ円、領域マスク・caption は元のまま (輪郭の形の効果)
  condO_labC      輪郭 cond は元のまま、領域マスクだけ円 (アテンションマスクの形の効果)
  circle_cap_meatonly  円 + global caption を「チャーシューだけ」に (global caption の具材列挙の影響)
"""
import json, os, sys, re
import numpy as np, cv2, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from infer import load_seg_map_and_prompt
from dataset.ramen_captions import CLASS_COLORS
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline

SRC = os.path.join(os.path.dirname(HERE), "examples/examples_Ramen_val55")
INP = os.path.join(HERE, "inputs"); OUT = os.path.join(HERE, "results")
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
MEAT, SOUP = np.array(CLASS_COLORS["meat"]), np.array(CLASS_COLORS["soup"])
EGGS = [np.array(CLASS_COLORS[c]) for c in ("egg", "cut_egg")]
os.makedirs(INP, exist_ok=True)

def is_col(m, c): return (m == c).all(-1)

def make_inputs():
    stems = []
    for f in sorted(os.listdir(SRC)):
        if not f.endswith(".json"): continue
        a = json.load(open(os.path.join(SRC, f)))
        if tuple(MEAT) not in [tuple(s["color"]) for s in a["segments_info"]]: continue
        stems.append(f[:-5])
        if len(stems) == 10: break
    for s in stems:
        a = json.load(open(os.path.join(SRC, s + ".json")))
        m = np.array(Image.open(os.path.join(SRC, s + ".png")).convert("RGB"))
        Image.fromarray(m).save(os.path.join(INP, f"{s}_orig.png")); json.dump(a, open(os.path.join(INP, f"{s}_orig.json"), "w"), indent=1)
        num, lab, stats, cent = cv2.connectedComponentsWithStats(is_col(m, MEAT).astype(np.uint8))
        i = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
        r = np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi); cx, cy = cent[i]
        c = m.copy(); c[lab == i] = SOUP
        yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
        c[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = MEAT
        Image.fromarray(c).save(os.path.join(INP, f"{s}_circle.png")); json.dump(a, open(os.path.join(INP, f"{s}_circle.json"), "w"), indent=1)
        n = c.copy()
        for e in EGGS: n[is_col(n, e)] = SOUP
        b = json.loads(json.dumps(a))
        b["caption"] = re.sub(r"(a halved boiled egg|a boiled egg), ", "", b["caption"]).replace(", a halved boiled egg", "").replace(", a boiled egg", "")
        Image.fromarray(n).save(os.path.join(INP, f"{s}_circlenoegg.png")); json.dump(b, open(os.path.join(INP, f"{s}_circlenoegg.json"), "w"), indent=1)
        b = json.loads(json.dumps(a))
        b["caption"] = re.sub(r"topped with [^.]*\.", "topped with sliced chashu pork.", b["caption"])
        json.dump(b, open(os.path.join(INP, f"{s}_circlemeatonly.json"), "w"), indent=1)
    return stems

stems = make_inputs()
print(stems)
L = lambda s, k, j=None: load_seg_map_and_prompt(os.path.join(INP, f"{s}_{k}.png"), os.path.join(INP, f"{s}_{j or k}.json"), 1)
COND = {
    "orig": lambda s: (L(s, "orig"), L(s, "orig")),
    "circle": lambda s: (L(s, "circle"), L(s, "circle")),
    "circle_noegg": lambda s: (L(s, "circlenoegg"), L(s, "circlenoegg")),
    "condC_labO": lambda s: (L(s, "circle"), L(s, "orig")),
    "condO_labC": lambda s: (L(s, "orig"), L(s, "circle")),
    "circle_cap_meatonly": lambda s: (L(s, "circle", "circlemeatonly"),) * 2,
}
tr = Flux3RegionalTransformer2DModel.from_pretrained("black-forest-labs/flux-3-image-klein", subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained("black-forest-labs/flux-3-image-klein", transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"]); pipe = pipe.to("cuda")
for name, fn in COND.items():
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    for s in stems:
        bc, bl = fn(s)
        img = pipe(global_prompt=bl["global_caption"], regional_prompts=bl["regional_captions"], regional_labels=bl["label"],
                   cond=(bc["cond_pixel_values"][None] + 1) / 2.0, attention_mask_method="hard", is_filter_cond_token=True,
                   height=512, width=512, cond_scale_factor=1, num_inference_steps=16,
                   generator=torch.Generator("cuda").manual_seed(bl["seed"]),
                   max_sequence_length=512, regional_max_sequence_length=50).images[0]
        img.save(os.path.join(OUT, name, s + ".jpg"), quality=95)
    print("done", name, flush=True)
