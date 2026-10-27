"""Compose [real | semantic mask | generated] triptychs for a result directory.

For every ramen_XXXX_0.jpg produced by infer.py, place next to it a
ramen_XXXX_panel.png holding, left to right:
  - the real UEC-Ramen555 photo (squashed to square, the way the model saw it in training)
  - the semantic color mask that conditioned generation
  - the generated image

Run:
    .venv/bin/python examples_Ramen/compose_panels.py \
        --result_dir ./result/ramen_3000 \
        --mask_dir ./examples_Ramen \
        --label "step 3000"
"""
import argparse
import glob
import os

from PIL import Image, ImageDraw

SRC_IMAGES = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555/Ramen_images"

GAP = 8            # white gutter between panels
LABEL_H = 28       # caption strip height
BG = (255, 255, 255)


def label_strip(width, text):
    strip = Image.new("RGB", (width, LABEL_H), BG)
    d = ImageDraw.Draw(strip)
    # default PIL font; anchor text roughly centered
    tw = d.textlength(text)
    d.text(((width - tw) / 2, 6), text, fill=(0, 0, 0))
    return strip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--mask_dir", default=os.path.join(os.path.dirname(__file__)))
    ap.add_argument("--label", default="generated")
    args = ap.parse_args()

    gens = sorted(glob.glob(os.path.join(args.result_dir, "ramen_*_0.jpg")))
    if not gens:
        raise SystemExit(f"no ramen_*_0.jpg found in {args.result_dir}")

    for gen_path in gens:
        stem = os.path.basename(gen_path)[: -len("_0.jpg")]  # ramen_0007
        idx = stem.split("_")[1]                             # 0007

        gen = Image.open(gen_path).convert("RGB")
        size = gen.size  # square, e.g. 512x512

        mask_path = os.path.join(args.mask_dir, stem + ".png")
        mask = Image.open(mask_path).convert("RGB").resize(size, Image.NEAREST)

        real_path = os.path.join(SRC_IMAGES, idx + ".png")
        # resize real to the square generation size, matching training's squash so the
        # comparison is like-for-like rather than penalising the model for the crop.
        real = Image.open(real_path).convert("RGB").resize(size, Image.BICUBIC)

        panels = [("Real", real), ("Semantic mask", mask), (f"Generated ({args.label})", gen)]
        w, h = size
        canvas = Image.new("RGB", (w * 3 + GAP * 2, h + LABEL_H), BG)
        for i, (title, img) in enumerate(panels):
            x = i * (w + GAP)
            canvas.paste(label_strip(w, title), (x, 0))
            canvas.paste(img, (x, LABEL_H))

        out = os.path.join(args.result_dir, stem + "_panel.png")
        canvas.save(out)
        print("wrote", os.path.relpath(out))


if __name__ == "__main__":
    main()
