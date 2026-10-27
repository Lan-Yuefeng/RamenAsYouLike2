"""Boundary IoU (Cheng et al. 2021)：セグメンテーション結果と入力マスクを、境界から d px 以内の帯だけで比べる IoU。

境界 recall (Canny エッジ) はエッジが増えるだけでも上がる。こちらは「ラーメン用 SegFormer が読み取った領域の形」が
マスクの境界に沿っているかを見るので、エッジの量には左右されない。クラスごとにデータセット全体で集計し、
正解に出てくるクラスで平均する。

    cd ~/Seg2Any/"[10-21]exp_layout_cfg" && ../.venv-flux3/bin/python boundary_iou.py --root results_val55 --conditions a,b,...
出力: <root>/boundary_iou.csv
"""
import argparse
import csv
import os

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seg_common import CLASS_NAMES, NUM_CLASSES, color_mask_to_index, normalize
from utils.boundary_metric import load_mask, load_rgb

p = argparse.ArgumentParser()
p.add_argument("--inputs", default="inputs_val55")
p.add_argument("--root", required=True)
p.add_argument("--conditions", required=True)
p.add_argument("--real_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result_f2/val55/real_512"))
p.add_argument("--seg_model", default="/export/ssd3/lan-y/seg2any_ckpt/ramen_segformer_b4")
p.add_argument("--d", default="3,8", help="境界帯の幅 (px、512² 上)。カンマ区切りで複数")
args = p.parse_args()

ds = [int(x) for x in args.d.split(",")]
stems = sorted(f[:-5] for f in os.listdir(args.inputs) if f.endswith(".json"))
seg = SegformerForSemanticSegmentation.from_pretrained(args.seg_model).cuda().eval()


@torch.no_grad()
def pred(path):
    x = torch.from_numpy(load_rgb(path)).permute(2, 0, 1)[None].float().div(255).cuda()
    lg = seg(pixel_values=normalize(x)).logits
    return torch.nn.functional.interpolate(lg, size=(512, 512), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()


def inner_band(binary, d):
    """領域のうち、輪郭から d px 以内の部分 (Boundary IoU の定義どおり、領域の内側の帯)"""
    b = binary.astype(np.uint8)
    er = cv2.erode(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return binary & ~(er > 0)


rows = []
for c in args.conditions.split(","):
    inter = {d: np.zeros(NUM_CLASSES) for d in ds}
    union = {d: np.zeros(NUM_CLASSES) for d in ds}
    present = np.zeros(NUM_CLASSES, bool)
    for s in stems:
        gt = color_mask_to_index(load_mask(os.path.join(args.inputs, s + ".png")))
        path = os.path.join(args.real_dir, s + ".png") if c == "real_photo" else os.path.join(args.root, c, s + "_0.jpg")
        pr = pred(path)
        valid = gt != 255
        for k in range(NUM_CLASSES):
            g, q = (gt == k) & valid, (pr == k) & valid
            if not (g.any() or q.any()):
                continue
            present[k] |= g.any()
            for d in ds:
                gb, qb = inner_band(g, d), inner_band(q, d)
                inter[d][k] += (gb & qb).sum()
                union[d][k] += (gb | qb).sum()
    r = dict(condition=c)
    for d in ds:
        iou = np.where(present, inter[d] / np.maximum(union[d], 1), np.nan)
        r[f"bIoU_d{d}"] = round(float(np.nanmean(iou)), 4)
        for k in range(NUM_CLASSES):
            if present[k]:
                r[f"d{d}_{CLASS_NAMES[k]}"] = round(float(iou[k]), 3)
    rows.append(r)
    print(c, {k: v for k, v in r.items() if k.startswith("bIoU")}, flush=True)

with open(os.path.join(args.root, "boundary_iou.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(), w.writerows(rows)
for r in rows:
    print(r["condition"], {k: v for k, v in r.items() if k.startswith("d3_")})
