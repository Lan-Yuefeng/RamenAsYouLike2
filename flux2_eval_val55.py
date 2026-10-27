"""val 55 枚での評価。result_f2/run_val55_gen.sh で生成した各条件を、同じ指標で比べる。

    HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 .venv-flux2/bin/python flux2_eval_val55.py

指標:
  boundary recall / precision : マスク境界と画像エッジの一致度 (utils/boundary_metric.py)。recall が主指標
  CLIP-T  : global caption と生成画像の CLIP 類似度 (×100)。caption に沿っているか
  CLIP-I  : 同じマスクの元写真と生成画像の CLIP 画像埋め込みの類似度 (×100)。元写真に近い見た目か
  KID / FID : val の元写真 55 枚 (512² に変形) との分布の距離 (clean-fid)。写真らしさ。
              55 枚では FID はばらつきが大きいので参考値、少数サンプル向きの KID を主に見る
出力: result_f2/val55/metrics_per_image.csv, summary.json, README.md, panel.jpg
"""
import argparse
import csv
import json
import os
import statistics

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

from utils.boundary_metric import boundary_score, load_mask, load_rgb

RAMEN_IMAGES = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555/Ramen_images"

parser = argparse.ArgumentParser()
parser.add_argument("--seg_mask_path", default="examples_Ramen_val55")
parser.add_argument("--result_root", default="result_f2/val55")
parser.add_argument("--conditions", default="flux1,flux2_base_cfg4,flux2_base_cfg4_negnocond,flux2_distilled")
parser.add_argument("--clip", default="openai/clip-vit-large-patch14")
args = parser.parse_args()

conds = args.conditions.split(",")
stems = sorted(f[:-5] for f in os.listdir(args.seg_mask_path) if f.endswith(".json"))
captions = {s: json.load(open(os.path.join(args.seg_mask_path, s + ".json")))["caption"] for s in stems}

# 元写真を 512² に変形して 1 フォルダに (学習・マスクと同じく縦横比は潰す)
real_dir = os.path.join(args.result_root, "real_512")
os.makedirs(real_dir, exist_ok=True)
for s in stems:
    out = os.path.join(real_dir, s + ".png")
    if not os.path.exists(out):
        Image.open(os.path.join(RAMEN_IMAGES, s.split("_")[1] + ".png")).convert("RGB").resize((512, 512), Image.BICUBIC).save(out)


def gen_path(cond, s):
    return os.path.join(args.result_root, cond, s + "_0.jpg")


for c in conds:
    missing = [s for s in stems if not os.path.exists(gen_path(c, s))]
    assert not missing, f"{c}: {len(missing)} 枚足りない (例 {missing[:3]})"

# ---------------- CLIP ----------------
device = "cuda"
clip = CLIPModel.from_pretrained(args.clip, torch_dtype=torch.float16).to(device).eval()
proc = CLIPProcessor.from_pretrained(args.clip)


@torch.no_grad()
def img_emb(paths):
    out = []
    for i in range(0, len(paths), 16):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 16]]
        x = proc(images=ims, return_tensors="pt")["pixel_values"].to(device, torch.float16)
        # transformers 5 では get_image_features の戻り値が変わったので、射影まで明示的に計算する
        feat = clip.visual_projection(clip.vision_model(pixel_values=x).pooler_output)
        out.append(torch.nn.functional.normalize(feat.float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def txt_emb(texts):
    x = proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    feat = clip.text_projection(clip.text_model(**x).pooler_output)
    return torch.nn.functional.normalize(feat.float(), dim=-1)


T = txt_emb([captions[s] for s in stems])
R = img_emb([os.path.join(real_dir, s + ".png") for s in stems])

rows = []
for c in ["real_photo"] + conds:
    paths = [os.path.join(real_dir, s + ".png") if c == "real_photo" else gen_path(c, s) for s in stems]
    E = img_emb(paths)
    clip_t = (E * T).sum(-1) * 100
    clip_i = (E * R).sum(-1) * 100
    for i, s in enumerate(stems):
        rec, pre = boundary_score(load_rgb(paths[i]), load_mask(os.path.join(args.seg_mask_path, s + ".png")))
        rows.append({"condition": c, "image": s, "recall": round(rec, 4), "precision": round(pre, 4),
                     "clip_t": round(clip_t[i].item(), 3), "clip_i": round(clip_i[i].item(), 3)})
    print(c, "per-image metrics done", flush=True)

# ---------------- KID / FID ----------------
from cleanfid import fid  # noqa: E402

dist = {}
for c in conds:
    d = os.path.join(args.result_root, c)
    dist[c] = {"kid_x1000": round(float(fid.compute_kid(d, real_dir, mode="clean", verbose=False)) * 1000, 3),
               "fid": round(float(fid.compute_fid(d, real_dir, mode="clean", verbose=False)), 2)}
    print(c, dist[c], flush=True)

# ---------------- 集計 ----------------
def mean_sd(vals):
    return round(statistics.mean(vals), 4), round(statistics.stdev(vals), 4)


summary = {}
for c in ["real_photo"] + conds:
    r = [x for x in rows if x["condition"] == c]
    summary[c] = {k: mean_sd([x[k] for x in r]) for k in ["recall", "precision", "clip_t", "clip_i"]}
    summary[c].update(dist.get(c, {}))

# FLUX.1 との画像ごとの差 (対応のある比較)。recall が FLUX.1 を上回った枚数
paired = {}
for c in conds if "flux1" in conds else []:
    if c == "flux1":
        continue
    a = {x["image"]: x["recall"] for x in rows if x["condition"] == "flux1"}
    b = {x["image"]: x["recall"] for x in rows if x["condition"] == c}
    diffs = [b[s] - a[s] for s in stems]
    paired[c] = {"mean_diff_recall_vs_flux1": round(statistics.mean(diffs), 4),
                 "n_better_than_flux1": sum(d > 0 for d in diffs), "n": len(diffs)}

with open(os.path.join(args.result_root, "metrics_per_image.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
json.dump({"n_images": len(stems), "summary": summary, "paired_vs_flux1": paired},
          open(os.path.join(args.result_root, "summary.json"), "w"), indent=1)

# 一覧 (先頭 12 枚)
cols = ["mask", "real photo"] + conds
s_ = 224
show = stems[:12]
panel = Image.new("RGB", (s_ * len(cols), 24 + s_ * len(show)), "white")
draw = ImageDraw.Draw(panel)
for j, c in enumerate(cols):
    draw.text((j * s_ + 4, 6), c, fill="black")
for i, s in enumerate(show):
    ims = [Image.open(os.path.join(args.seg_mask_path, s + ".png")), Image.open(os.path.join(real_dir, s + ".png"))]
    ims += [Image.open(gen_path(c, s)) for c in conds]
    for j, im in enumerate(ims):
        panel.paste(im.convert("RGB").resize((s_, s_)), (j * s_, 24 + i * s_))
panel.save(os.path.join(args.result_root, "panel.jpg"), quality=90)

for c, m in summary.items():
    print(c, m)
print(json.dumps(paired, indent=1))
