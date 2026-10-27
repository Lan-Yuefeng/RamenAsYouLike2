"""UECFoodPix 皿版の比較パネル: マスク | GT | FLUX.2 Base | FLUX.3 Base | FLUX.3 蒸留版。

各画像は縮小しない (マスク・GT は元の解像度、生成は 512² のまま)。上揃えで横に並べ、下にクラス名の凡例。
FLUX.2 は自作の皿マスクで生成した既存の result_f2/UECFOODPIX_flux2/uecfoodpix_plate_flux2_test10_base を使う。
この 10 枚では自作マスクと共有マスク (FLUX.3 が使う) の一致率は 99.8% 以上、クラス構成も同じ。

    /export/ssd2/lan-y/venvs/Seg2Any/.venv-flux3/bin/python scripts/make_uecfoodpix_flux3_compare.py
出力: result_f3/UECFOODPIX_flux3/compare_test10/panel_<stem>.jpg と all_panels.jpg
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # リポジトリ直下 (src/ dataset/ utils/ infer.py) を import できるように
import colorsys
import os

import numpy as np
from PIL import Image, ImageDraw

from dataset import uecfoodpix_plate_captions as plate_vocab

STEMS = "16164 3061 3317 3709 3710 3770 4357 5599 6286 6910".split()
IMG_ROOT = "/export/ssd2/lan-y/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/test/img"
MASK_ROOT = "/export/ssd2/lan-y/seg2any_data/uecfoodpix/mask_plate_lab/test"
COLUMNS = [
    ("FLUX.2 Base 20step CFG4", "result_f2/UECFOODPIX_flux2/uecfoodpix_plate_flux2_test10_base/gen"),
    ("FLUX.3 Base 20step CFG3.5", "result_f3/UECFOODPIX_flux3/plate_test10_base20/gen"),
    ("FLUX.3 distilled 16step", "result_f3/UECFOODPIX_flux3/plate_test10_distill16/gen"),
]
OUT = "result_f3/UECFOODPIX_flux3/compare_test10"


def colour(class_id):
    """scripts/gen_uecfoodpix_flux2_test.py と同じ配色。0 (テーブル) は黒、103 (皿) は薄い灰色"""
    if class_id == 0:
        return (0, 0, 0)
    if class_id == plate_vocab.PLATE_ID:
        return (200, 200, 200)
    h = (class_id * 47) % 180 + 10
    s = 200 + (class_id * 13) % 55
    v = 170 + (class_id * 29) % 85
    r, g, b = colorsys.hsv_to_rgb(h / 255.0, s / 255.0, v / 255.0)
    return (int(r * 255), int(g * 255), int(b * 255))


PALETTE = np.array([colour(i) for i in range(256)], dtype=np.uint8)


def make_panel(stem):
    segm = np.array(Image.open(f"{MASK_ROOT}/{stem}.png"))
    tiles = [("mask", Image.fromarray(PALETTE[segm])), ("GT", Image.open(f"{IMG_ROOT}/{stem}.jpg").convert("RGB"))]
    tiles += [(name, Image.open(f"{d}/{stem}.jpg").convert("RGB")) for name, d in COLUMNS]

    gap, head, foot = 8, 22, 26
    W = sum(t.width for _, t in tiles) + gap * (len(tiles) - 1)
    H = max(t.height for _, t in tiles)
    panel = Image.new("RGB", (W, head + H + foot), "white")
    d = ImageDraw.Draw(panel)
    x = 0
    for name, t in tiles:
        panel.paste(t, (x, head))
        d.text((x + 4, 5), name, fill="black")
        x += t.width + gap
    x = 4
    for i in np.unique(segm):
        d.rectangle([x, head + H + 7, x + 12, head + H + 19], fill=colour(int(i)), outline="gray")
        label = plate_vocab.regional_caption(int(i))
        d.text((x + 16, head + H + 7), label, fill="black")
        x += 16 + int(d.textlength(label)) + 12
    d.text((W - 60, head + H + 7), stem, fill="gray")
    return panel


os.makedirs(OUT, exist_ok=True)
panels = []
for stem in STEMS:
    p = make_panel(stem)
    p.save(f"{OUT}/panel_{stem}.jpg", quality=92)
    panels.append(p)
W = max(p.width for p in panels)
all_panels = Image.new("RGB", (W, sum(p.height for p in panels)), "white")
y = 0
for p in panels:
    all_panels.paste(p, (0, y))
    y += p.height
all_panels.save(f"{OUT}/all_panels.jpg", quality=90)
print(OUT, all_panels.size)
