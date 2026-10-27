"""Compose [ground truth | semantic mask | generated...] strips for a result directory.

    PYTHONPATH=. .venv/bin/python examples_UECFoodPix/compose_panels.py \
        --result_dir /export/ssd3/lan-y/seg2any_ckpt/uecfoodpix_infer/ckpt5000 \
        --label "checkpoint-5000"

The ground truth comes from examples_UECFoodPix/gt/, written by make_examples.py, so the
comparison is against the held-out test image the mask was taken from.
"""
import argparse
import glob
import os
from collections import defaultdict

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
GAP, LABEL_H, BG = 8, 28, (255, 255, 255)


def label_strip(w, text):
    s = Image.new("RGB", (w, LABEL_H), BG)
    ImageDraw.Draw(s).text((4, 8), text, fill=(0, 0, 0))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--label", default="generated")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    out_dir = a.out_dir or a.result_dir

    gen = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.result_dir, "*_[0-9].jpg"))):
        gen[os.path.basename(f).rsplit("_", 1)[0]].append(f)
    if not gen:
        raise SystemExit(f"no generated images in {a.result_dir}")

    for stem, files in sorted(gen.items()):
        tiles, labels = [], []
        gt = os.path.join(HERE, "gt", f"{stem}.jpg")
        if os.path.exists(gt):
            tiles.append(Image.open(gt).convert("RGB")); labels.append("ground truth")
        mask = os.path.join(HERE, f"{stem}.png")
        if os.path.exists(mask):
            tiles.append(Image.open(mask).convert("RGB")); labels.append("mask")
        for i, f in enumerate(sorted(files)):
            tiles.append(Image.open(f).convert("RGB")); labels.append(f"{a.label} #{i}")

        h = max(t.height for t in tiles)
        tiles = [t.resize((int(t.width * h / t.height), h), Image.BICUBIC) for t in tiles]
        W = sum(t.width for t in tiles) + GAP * (len(tiles) - 1)
        canvas = Image.new("RGB", (W, h + LABEL_H), BG)
        x = 0
        for t, lab in zip(tiles, labels):
            canvas.paste(label_strip(t.width, lab), (x, 0))
            canvas.paste(t, (x, LABEL_H))
            x += t.width + GAP
        p = os.path.join(out_dir, f"{stem}_panel.png")
        canvas.save(p)
        print("  wrote", p)


if __name__ == "__main__":
    main()
