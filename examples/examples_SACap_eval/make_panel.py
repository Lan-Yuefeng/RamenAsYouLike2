"""result_f2/sacap_flux2/<examples dir>/ の生成結果を 1 枚の比較画像にまとめる。

    python examples_SACap_eval/make_panel.py examples_SACap_eval
列: マスク | 元写真 (あれば) | Base LoRA なし | Base + LoRA | 蒸留版 + LoRA
"""
import glob
import os
import sys

from PIL import Image, ImageDraw

ex = sys.argv[1]
res = f"result_f2/sacap_flux2/{ex}"
cols = [("mask", None), ("GT", None), ("Base, no LoRA", "base_nolora"), ("Base + LoRA (50 step)", "base_lora"), ("Distilled + LoRA (4 step)", "distill_lora")]
H = 320
rows = []
for png in sorted(glob.glob(f"{ex}/*.png")):
    stem = os.path.splitext(os.path.basename(png))[0]
    gt = glob.glob(f"{ex}/gt/{stem}.*")
    tiles = []
    for name, sub in cols:
        if name == "mask":
            p = png
        elif name == "GT":
            p = gt[0] if gt else None
        else:
            p = f"{res}/{sub}/{stem}_0.jpg"
        if p is None or not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        im = im.resize((round(im.width * H / im.height), H))
        tiles.append((name, im))
    rows.append(tiles)

width = max(sum(t.width for _, t in r) + 8 * (len(r) - 1) for r in rows)
panel = Image.new("RGB", (width, len(rows) * (H + 28)), "white")
d = ImageDraw.Draw(panel)
y = 0
for r in rows:
    x = 0
    for name, im in r:
        d.text((x + 4, y + 6), name, fill="black")
        panel.paste(im, (x, y + 24))
        x += im.width + 8
    y += H + 28
panel.save(f"{res}/panel.jpg", quality=90)
print(f"{res}/panel.jpg", panel.size)
