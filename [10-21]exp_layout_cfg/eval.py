"""results/ の各条件を評価する (KID / FID は出さない)。

    cd ~/Seg2Any/"[10-21]exp_layout_cfg" && ../.venv-flux3/bin/python eval.py --conditions base_s3.5,oldneg_s3.5,...

指標は flux2_eval_val55.py と同じ：境界 recall / precision、CLIP-T (global caption)、CLIP-I (元写真)。
mIoU (--seg_model を渡したとき)：生成画像を seg_train.py のラーメン用 SegFormer で分割し、入力マスクと比べる。
  データセット全体の混同行列から求める mIoU と、クラスごとの IoU。real_photo 行は実写での値 (評価器の上限の目安)
秒/枚は gen.py の times*.json (各ジョブの 1 枚目はウォームアップ込みなので除く)。
1 つ目の条件との対応のある比較 (画像ごとの差の平均と、符号付き順位検定の p 値) を paired.csv に出す。
出力：results/summary.csv、results/panel.jpg (1 マス 512 px、縮小なし)
"""
import argparse
import csv
import json
import os
import statistics
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from utils.boundary_metric import boundary_score, load_mask, load_rgb, mask_boundary  # noqa: E402
from seg_common import CLASS_NAMES, NUM_CLASSES, color_mask_to_index, confusion, miou_from_conf, normalize  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--inputs", default=os.path.join(HERE, "inputs_val10"))
p.add_argument("--root", default=os.path.join(HERE, "results"))
p.add_argument("--conditions", required=True)
p.add_argument("--real_dir", default=os.path.join(os.path.dirname(HERE), "result_f2/val55/real_512"))
p.add_argument("--clip", default="openai/clip-vit-large-patch14")
p.add_argument("--panel", default="panel.jpg")
p.add_argument("--panel_n", type=int, default=12, help="パネルに載せる枚数 (先頭から)")
p.add_argument("--seg_model", default=None, help="例 /export/ssd3/lan-y/seg2any_ckpt/ramen_segformer_b4")
args = p.parse_args()

conds = args.conditions.split(",")
stems = sorted(f[:-5] for f in os.listdir(args.inputs) if f.endswith(".json"))
captions = {s: json.load(open(os.path.join(args.inputs, s + ".json")))["caption"] for s in stems}
def gen(c, s):
    if c == "real_photo":
        return os.path.join(args.real_dir, s + ".png")
    return os.path.join(args.root, c, s + "_0.jpg")

device = "cuda"
clip = CLIPModel.from_pretrained(args.clip, torch_dtype=torch.float16).to(device).eval()
proc = CLIPProcessor.from_pretrained(args.clip)


@torch.no_grad()
def img_emb(paths):
    x = proc(images=[Image.open(q).convert("RGB") for q in paths], return_tensors="pt")["pixel_values"]
    f = clip.visual_projection(clip.vision_model(pixel_values=x.to(device, torch.float16)).pooler_output)
    return torch.nn.functional.normalize(f.float(), dim=-1)


@torch.no_grad()
def txt_emb(texts):
    x = proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    return torch.nn.functional.normalize(clip.text_projection(clip.text_model(**x).pooler_output).float(), dim=-1)


T = txt_emb([captions[s] for s in stems])
R = img_emb([os.path.join(args.real_dir, s + ".png") for s in stems])

seg = None
if args.seg_model:
    from transformers import SegformerForSemanticSegmentation
    seg = SegformerForSemanticSegmentation.from_pretrained(args.seg_model).to(device).eval()
    gt_idx = {s: color_mask_to_index(load_mask(os.path.join(args.inputs, s + ".png"))) for s in stems}


@torch.no_grad()
def seg_pred(path):
    x = torch.from_numpy(load_rgb(path)).permute(2, 0, 1)[None].float().div(255).to(device)
    lg = seg(pixel_values=normalize(x)).logits
    return torch.nn.functional.interpolate(lg, size=(512, 512), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()


rows, summary, per_class = [], [], {}
base = None
for c in conds + (["real_photo"] if seg is not None else []):
    E = img_emb([gen(c, s) for s in stems])
    ct, ci = ((E * T).sum(-1) * 100).tolist(), ((E * R).sum(-1) * 100).tolist()
    rec, pre, mious = [], [], []
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for i, s in enumerate(stems):
        r, q = boundary_score(load_rgb(gen(c, s)), load_mask(os.path.join(args.inputs, s + ".png")))
        rec.append(r), pre.append(q)
        row = dict(condition=c, image=s, recall=round(r, 4), precision=round(q, 4),
                   clip_t=round(ct[i], 3), clip_i=round(ci[i], 3))
        if seg is not None:
            cf = confusion(seg_pred(gen(c, s)), gt_idx[s])
            conf += cf
            row["miou"] = round(miou_from_conf(cf)[0], 4)
            mious.append(row["miou"])
        rows.append(row)
    # 参考として流用した既存出力 (ref_*) と real_photo には時間の記録が無い
    tfiles = [f for f in os.listdir(os.path.join(args.root, c)) if f.startswith("times")] if c != "real_photo" else []
    tv = [v for f in tfiles for v in list(json.load(open(os.path.join(args.root, c, f))).values())[1:]]
    sec = statistics.mean(tv) if tv else float("nan")
    if base is None:
        base = rec
    srow = dict(condition=c, recall=round(statistics.mean(rec), 4), precision=round(statistics.mean(pre), 4),
                clip_t=round(statistics.mean(ct), 2), clip_i=round(statistics.mean(ci), 2))
    if seg is not None:
        m, iou = miou_from_conf(conf)
        srow["miou"] = round(m, 4)
        per_class[c] = iou
    srow.update(n_recall_up_vs_first=sum(a > b for a, b in zip(rec, base)), sec_per_img=round(sec, 2))
    summary.append(srow)

with open(os.path.join(args.root, "metrics_per_image.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(), w.writerows(rows)
with open(os.path.join(args.root, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader(), w.writerows(summary)
print(f"{'condition':22s} recall  prec   CLIP-T CLIP-I  mIoU   up/n  s/img")
for r in summary:
    print(f"{r['condition']:22s} {r['recall']:.3f}  {r['precision']:.3f}  {r['clip_t']:.2f}  {r['clip_i']:.2f}"
          f"  {r.get('miou', float('nan')):.3f}  {r['n_recall_up_vs_first']:>3d}    {r['sec_per_img']}")

if per_class:
    with open(os.path.join(args.root, "per_class_iou.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition"] + CLASS_NAMES)
        for c, iou in per_class.items():
            w.writerow([c] + [round(float(v), 4) for v in iou])
    print("per-class IoU:", " ".join(CLASS_NAMES))
    for c, iou in per_class.items():
        print(f"{c:22s}", " ".join(f"{v:.3f}" for v in iou))

# 1 つ目の条件との対応のある比較
from scipy.stats import wilcoxon  # noqa: E402

keys = ["recall", "precision", "clip_t", "clip_i"] + (["miou"] if seg is not None else [])
paired = []
for c in conds[1:]:
    for k in keys:
        a = [r[k] for r in rows if r["condition"] == conds[0]]
        b = [r[k] for r in rows if r["condition"] == c]
        d = [y - x for x, y in zip(a, b)]
        paired.append(dict(condition=c, metric=k, mean_diff=round(statistics.mean(d), 4),
                           n_up=sum(v > 0 for v in d), n=len(d),
                           p_wilcoxon=float(f"{wilcoxon(b, a).pvalue:.2g}") if any(d) else 1.0))
with open(os.path.join(args.root, "paired.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(paired[0]))
    w.writeheader(), w.writerows(paired)
print(f"paired vs {conds[0]}:")
for r in paired:
    print(f"  {r['condition']:22s} {r['metric']:9s} diff {r['mean_diff']:+.4f}  up {r['n_up']}/{r['n']}  p={r['p_wilcoxon']}")

# パネル：マスク | 元写真 | 各条件 (生成画像にマスク境界を赤で重ねたもの)。512 px のまま
S = 512
cols = ["mask", "real"] + conds
stems = stems[:args.panel_n]
panel = Image.new("RGB", (S * len(cols), 30 + S * len(stems)), "white")
d = ImageDraw.Draw(panel)
for j, c in enumerate(cols):
    d.text((j * S + 6, 8), c, fill="black")
for i, s in enumerate(stems):
    m = load_mask(os.path.join(args.inputs, s + ".png"))
    bd = mask_boundary(m)
    ims = [Image.fromarray(m), Image.open(os.path.join(args.real_dir, s + ".png")).convert("RGB")]
    for c in conds:
        a = load_rgb(gen(c, s)).copy()
        a[bd] = (255, 0, 0)
        ims.append(Image.fromarray(a))
    for j, im in enumerate(ims):
        panel.paste(im.resize((S, S), Image.NEAREST) if im.size != (S, S) else im, (j * S, 30 + i * S))
panel.save(os.path.join(args.root, args.panel), quality=90)
