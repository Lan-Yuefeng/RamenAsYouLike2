"""1024² 推論 (学習は 512²) を val の先頭 10 枚で 512² 推論と比べる。

    HF_HOME=/export/ssd3/lan-y/huggingface .venv-flux3/bin/python scripts/flux3_eval_val10_1024.py

指標は scripts/flux2_eval_val55.py と同じ (生成画像は 512 に縮めて、512 の mask と比べる)。10 枚なので KID / FID は出さない。
パネルは例ごとに 1 枚 (panel_<stem>.jpg)。1 マス 1024 px で、1024² の生成画像は縮小しない (512² のものは拡大して並べる)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # リポジトリ直下 (src/ dataset/ utils/ infer.py) を import できるように
import argparse
import json
import os
import statistics

import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

from utils.boundary_metric import boundary_score, load_mask, load_rgb

parser = argparse.ArgumentParser()
parser.add_argument("--seg_mask_path", default="examples_Ramen_val55", help="512 の mask (先頭 n 枚を使う)")
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--result_root", default="result_f3/val10_flux3_1024")
parser.add_argument("--conditions", default="flux3_base_cfg35_50step_512,flux3_base_cfg35_50step_1024,"
                    "flux3_base_cfg35_50step_1024_csf2,flux3_distilled_512,flux3_distilled_1024,flux3_distilled_1024_csf2")
parser.add_argument("--clip", default="openai/clip-vit-large-patch14")
args = parser.parse_args()

conds = args.conditions.split(",")
stems = sorted(f[:-5] for f in os.listdir(args.seg_mask_path) if f.endswith(".json"))[: args.n]
captions = {s: json.load(open(os.path.join(args.seg_mask_path, s + ".json")))["caption"] for s in stems}
real = lambda s: os.path.join(args.result_root, "real_512", s + ".png")  # noqa: E731
gen = lambda c, s: os.path.join(args.result_root, c, s + "_0.jpg")  # noqa: E731

device = "cuda"
clip = CLIPModel.from_pretrained(args.clip, torch_dtype=torch.float16).to(device).eval()
proc = CLIPProcessor.from_pretrained(args.clip)


@torch.no_grad()
def img_emb(paths):
    x = proc(images=[Image.open(p).convert("RGB") for p in paths], return_tensors="pt")["pixel_values"]
    feat = clip.visual_projection(clip.vision_model(pixel_values=x.to(device, torch.float16)).pooler_output)
    return torch.nn.functional.normalize(feat.float(), dim=-1)


@torch.no_grad()
def txt_emb(texts):
    x = proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    return torch.nn.functional.normalize(clip.text_projection(clip.text_model(**x).pooler_output).float(), dim=-1)


T = txt_emb([captions[s] for s in stems])
R = img_emb([real(s) for s in stems])
summary = {}
for c in conds:
    E = img_emb([gen(c, s) for s in stems])
    sc = [boundary_score(load_rgb(gen(c, s)), load_mask(os.path.join(args.seg_mask_path, s + ".png"))) for s in stems]
    vals = {"recall": [r for r, _ in sc], "precision": [p for _, p in sc],
            "clip_t": ((E * T).sum(-1) * 100).tolist(), "clip_i": ((E * R).sum(-1) * 100).tolist()}
    summary[c] = {k: round(statistics.mean(v), 4) for k, v in vals.items()}
    summary[c]["recall_per_image"] = [round(r, 3) for r in vals["recall"]]
    print(c, {k: v for k, v in summary[c].items() if k != "recall_per_image"}, flush=True)
json.dump({"stems": stems, "summary": summary}, open(os.path.join(args.result_root, "summary.json"), "w"), indent=1)

# 例ごとのパネル (1 行): mask, 元写真, 各条件
S = 1024
cols = ["mask", "real photo"] + conds
for s in stems:
    panel = Image.new("RGB", (S * len(cols), 40 + S), "white")
    draw = ImageDraw.Draw(panel)
    paths = [os.path.join(args.seg_mask_path, s + ".png"), real(s)] + [gen(c, s) for c in conds]
    for j, (name, p) in enumerate(zip(cols, paths)):
        im = Image.open(p).convert("RGB")
        if im.size != (S, S):
            im = im.resize((S, S), Image.NEAREST if name == "mask" else Image.LANCZOS)
        panel.paste(im, (j * S, 40))
        draw.text((j * S + 8, 10), f"{name}  ({Image.open(p).size[0]}px)", fill="black")
    panel.save(os.path.join(args.result_root, f"panel_{s}.jpg"), quality=92)
print("panels ->", args.result_root)
