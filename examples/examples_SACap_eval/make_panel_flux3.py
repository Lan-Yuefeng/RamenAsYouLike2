"""SACap で学習した LoRA の FLUX.2 / FLUX.3 生成結果を、画像ごとに 1 枚の比較パネルにする (縮小・非可逆圧縮なし、PNG)。

    python examples_SACap_eval/make_panel_flux3.py [examples_SACap_eval | examples]
列: マスク | 元写真 | FLUX.2 Base+LoRA | FLUX.3 Base LoRA なし | FLUX.3 Base+LoRA | FLUX.3 蒸留版+LoRA
出力: result_f3/sacap_flux3/<examples dir>/panel_<stem>.png
元写真 (examples_SACap_eval のみ) (約 2250x1500) は原寸のまま一番下の行に置く。生成画像はすべて 1024x672 の原寸。
"""
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFont

EX = sys.argv[1] if len(sys.argv) > 1 else "examples_SACap_eval"
F2 = f"result_f2/sacap_flux2/{EX}"
F3 = f"result_f3/sacap_flux3/{EX}"
GAP, LABEL = 12, 48
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
except OSError:
    FONT = ImageFont.load_default()

rows_spec = [
    [("mask", "{ex}/{s}.png"), ("FLUX.2 Base + LoRA (50 step)", "{f2}/base_lora/{s}_0.jpg"),
     ("FLUX.3 Base, no LoRA (50 step)", "{f3}/base_nolora/{s}_0.jpg")],
    [("FLUX.3 Base + LoRA (50 step)", "{f3}/base_lora/{s}_0.jpg"),
     ("FLUX.3 Distilled + LoRA (16 step)", "{f3}/distill_lora/{s}_0.jpg")],
    [("GT (original size)", "{ex}/gt/{s}.jpg")],
]

for png in sorted(glob.glob(f"{EX}/*.png")):
    s = os.path.splitext(os.path.basename(png))[0]
    rows = []
    for spec in rows_spec:
        tiles = []
        for name, pat in spec:
            p = pat.format(ex=EX, f2=F2, f3=F3, s=s)
            if os.path.exists(p):
                tiles.append((name, Image.open(p).convert("RGB")))
        if tiles:
            rows.append(tiles)
    W = max(sum(t.width for _, t in r) + GAP * (len(r) - 1) for r in rows)
    H = sum(LABEL + max(t.height for _, t in r) + GAP for r in rows)
    panel = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(panel)
    y = 0
    for r in rows:
        x = 0
        for name, im in r:
            d.text((x + 6, y + 8), name, fill="black", font=FONT)
            panel.paste(im, (x, y + LABEL))
            x += im.width + GAP
        y += LABEL + max(t.height for _, t in r) + GAP
    out = f"{F3}/panel_{s}.png"
    panel.save(out)
    print(out, panel.size)
