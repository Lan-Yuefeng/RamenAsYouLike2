"""円の大きさを振る。一番大きいチャーシュー領域を、その重心に置いた縦横比 0.75 の楕円 (面積は画像の p%) に置き換える。
ell{p}        卵/半熟卵はそのまま
ell{p}_noegg  卵/半熟卵をスープに塗り、global caption からも消す
学習データの面積の中央値: 卵 2.1%、半熟卵 4.4%、チャーシュー 4.9% (shape_prior.py)。"""
import json, os, sys, re
import numpy as np, cv2, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from infer import load_seg_map_and_prompt
from dataset.ramen_captions import CLASS_COLORS
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline
INP = os.path.join(HERE, "inputs"); OUT = os.path.join(HERE, "results")
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
MEAT, SOUP = np.array(CLASS_COLORS["meat"]), np.array(CLASS_COLORS["soup"])
EGGS = [np.array(CLASS_COLORS[c]) for c in ("egg", "cut_egg")]
stems = sorted(f[:-9] for f in os.listdir(INP) if f.endswith("_orig.png"))
PS = [float(x) for x in (sys.argv[1:] or ["1", "2", "3"])]
names = []
for s in stems:
    a = json.load(open(os.path.join(INP, s + "_orig.json")))
    m = np.array(Image.open(os.path.join(INP, s + "_orig.png")).convert("RGB"))
    meat = (m == MEAT).all(-1)
    num, lab, st, cent = cv2.connectedComponentsWithStats(meat.astype(np.uint8))
    i = 1 + st[1:, cv2.CC_STAT_AREA].argmax(); cx, cy = cent[i]
    yy, xx = np.mgrid[:m.shape[0], :m.shape[1]]
    for p in PS:
        A = p / 100 * m.size / 3; b = np.sqrt(A / np.pi / 0.75); a_ = 0.75 * b  # 横長 b, 縦 a_
        c = m.copy(); c[lab == i] = SOUP
        c[((xx - cx) / b) ** 2 + ((yy - cy) / a_) ** 2 <= 1] = MEAT
        tag = f"ell{p:g}"
        Image.fromarray(c).save(os.path.join(INP, f"{s}_{tag}.png")); json.dump(a, open(os.path.join(INP, f"{s}_{tag}.json"), "w"), indent=1)
        n = c.copy()
        for e in EGGS: n[(n == e).all(-1)] = SOUP
        b2 = json.loads(json.dumps(a))
        b2["caption"] = re.sub(r"(a halved boiled egg|a boiled egg), ", "", b2["caption"]).replace(", a halved boiled egg", "").replace(", a boiled egg", "")
        Image.fromarray(n).save(os.path.join(INP, f"{s}_{tag}_noegg.png")); json.dump(b2, open(os.path.join(INP, f"{s}_{tag}_noegg.json"), "w"), indent=1)
        names += [tag, tag + "_noegg"]
names = list(dict.fromkeys(names))
tr = Flux3RegionalTransformer2DModel.from_pretrained("black-forest-labs/flux-3-image-klein", subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained("black-forest-labs/flux-3-image-klein", transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"]); pipe = pipe.to("cuda")
for name in names:
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
