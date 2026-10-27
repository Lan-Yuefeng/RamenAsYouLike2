"""infer_flux2.py の出力ディレクトリから、マスクの重ね合わせと比較パネルを作る。

    .venv-flux2/bin/python make_result_panels.py result_f2/cfg3.0_steps16 [--title "CFG 3.0, 16 steps"]

<dir>/input/*.png (マスク) と <dir>/generated/*_0.jpg (生成) を読み、次を書き出す:
  overlay/<stem>_overlay.jpg : 生成画像にマスク色を 45% で重ね、領域境界を白線で描いたもの
  panel_mask_gt_gen.jpg      : 各行 マスク | GT (元写真, 生成と同じサイズに変形) | 生成  (GT が無い入力では作らない)
  panel_overlay.jpg          : 各行 マスク | 生成 | マスクを重ねた生成
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image, ImageDraw

RAMEN_IMAGES = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555/Ramen_images"

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("--title", default="")
parser.add_argument("--alpha", type=float, default=0.45)
args = parser.parse_args()

os.makedirs(os.path.join(args.dir, "overlay"), exist_ok=True)


def overlay(gen, mask):
    g = np.asarray(gen, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    o = (1 - args.alpha) * g + args.alpha * m
    # 4 近傍で色が変わる画素を境界とみなす
    edge = np.zeros(m.shape[:2], bool)
    dy = (m[1:] != m[:-1]).any(-1)
    dx = (m[:, 1:] != m[:, :-1]).any(-1)
    edge[1:] |= dy
    edge[:-1] |= dy
    edge[:, 1:] |= dx
    edge[:, :-1] |= dx
    o[edge] = 255
    return Image.fromarray(o.clip(0, 255).astype(np.uint8))


rows = []
for mp in sorted(glob.glob(os.path.join(args.dir, "input", "*.png"))):
    stem = os.path.splitext(os.path.basename(mp))[0]
    gen = Image.open(os.path.join(args.dir, "generated", f"{stem}_0.jpg")).convert("RGB")
    mask = Image.open(mp).convert("RGB").resize(gen.size, Image.NEAREST)
    gt_path = os.path.join(RAMEN_IMAGES, stem.split("_")[-1] + ".png") if stem.startswith("ramen_") else ""
    gt = Image.open(gt_path).convert("RGB").resize(gen.size) if os.path.exists(gt_path) else None
    ov = overlay(gen, mask)
    ov.save(os.path.join(args.dir, "overlay", f"{stem}_overlay.jpg"), quality=95)
    rows.append((mask, gt, gen, ov))


def panel(cols, idx, out):
    # 各画像は縮小せず生成画像と同じ解像度のまま並べる (ユーザー指定。パネルが大きくなってもよい)
    head = 28
    c = max(r[2].width for r in rows)
    p = Image.new("RGB", (c * len(cols), head + sum(r[2].height for r in rows)), "white")
    d = ImageDraw.Draw(p)
    for j, name in enumerate(cols):
        d.text((j * c + 4, 6), name + (f"  ({args.title})" if args.title and j == len(cols) - 1 else ""), fill="black")
    y = head
    for r in rows:
        for j, k in enumerate(idx):
            p.paste(r[k], (j * c, y))
        y += r[2].height
    p.save(os.path.join(args.dir, out), quality=90)


if all(r[1] is not None for r in rows):
    panel(["mask", "ground truth", "generated"], [0, 1, 2], "panel_mask_gt_gen.jpg")
panel(["mask", "generated", "mask overlaid on generated"], [0, 2, 3], "panel_overlay.jpg")
print(f"{len(rows)} rows -> {args.dir}")
