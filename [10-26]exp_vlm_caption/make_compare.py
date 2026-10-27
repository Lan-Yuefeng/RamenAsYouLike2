"""手書きテンプレートと VLM caption を画像ごとに並べたパネルを作る (2026-10-26)。

左に正解画像（原寸）、右に領域ごとの caption（上: 手書き、下: VLM）。
  python make_compare.py --captions captions_trial --out compare_trial
"""
import argparse
import glob
import json
import os
import sys
import textwrap

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset.ramen_captions import CLASS_COLORS, global_caption, regional_caption  # noqa: E402

DATASET_ROOT = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    font = ImageFont.truetype(FONT, 18)
    bold = ImageFont.truetype(FONT.replace("Sans.ttf", "Sans-Bold.ttf"), 18)
    wrap, line_h, text_w = 90, 24, 1000

    for path in sorted(glob.glob(os.path.join(args.captions, "*.json"))):
        d = json.load(open(path))
        photo = Image.open(os.path.join(DATASET_ROOT, "Ramen_images", d["image"])).convert("RGB")
        cmask = Image.open(os.path.join(DATASET_ROOT, "Ramen_color_masks", d["image"])).convert("RGB")

        rows = [("global", None, global_caption(list(d["regions"]), d["soup"]), d["global"])]
        rows += [(c, CLASS_COLORS[c], regional_caption(c, d["soup"]), v) for c, v in d["regions"].items()]
        blocks = []
        for name, color, hand, vlm in rows:
            lines = [(("H: " if i == 0 else "   ") + l, "black") for i, l in enumerate(textwrap.wrap(hand, wrap))]
            lines += [(("V: " if i == 0 else "   ") + l, (0, 0, 160)) for i, l in enumerate(textwrap.wrap(vlm, wrap))]
            blocks.append((name, color, lines))
        text_h = sum(line_h * (len(b[2]) + 1) + 8 for b in blocks) + 40

        W = photo.width * 2 + text_w + 40
        H = max(photo.height, text_h)
        canvas = Image.new("RGB", (W, H), "white")
        canvas.paste(photo, (0, 0))
        canvas.paste(cmask, (photo.width, 0))
        draw = ImageDraw.Draw(canvas)
        x, y = photo.width * 2 + 20, 10
        draw.text((x, y), f"{d['image']}  soup={d['soup']}   H=handwritten  V=VLM", font=bold, fill="black")
        y += line_h + 8
        for name, color, lines in blocks:
            if color:
                draw.rectangle([x, y + 3, x + 16, y + 19], fill=color, outline="black")
            draw.text((x + 24, y), name, font=bold, fill="black")
            y += line_h
            for l, fill in lines:
                draw.text((x, y), l, font=font, fill=fill)
                y += line_h
            y += 8
        canvas.save(os.path.join(args.out, d["image"]))
        print("saved", d["image"])


if __name__ == "__main__":
    main()
