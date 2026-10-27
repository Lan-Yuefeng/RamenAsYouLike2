"""自己蒸留の結果パネル。行 = val10、列 = 入力マスク / ① のみ (ref_fuse) / ①+③ 学習前 (sd_0) / ①+③ 自己蒸留 2000 step (原寸 512)。"""
import os, glob
from PIL import Image, ImageDraw
D = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.expanduser("~/Seg2Any/[10-26]exp_speed/val10")
cols = [("mask", None), ("1 fuse (no cache)", "ref_fuse"), ("1+3 before training", "sd_0"), ("1+3 self-distill 2000", "sd_2000")]
stems = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(glob.escape(VAL), "*.png")))]
S, H = 512, 40
panel = Image.new("RGB", (S * len(cols), H + S * len(stems)), "white")
d = ImageDraw.Draw(panel)
for c, (name, _) in enumerate(cols):
    d.text((c * S + 10, 12), name, fill="black")
for r, st in enumerate(stems):
    for c, (_, sub) in enumerate(cols):
        p = os.path.join(VAL, st + ".png") if sub is None else os.path.join(D, "result", sub, st + "_0.jpg")
        panel.paste(Image.open(p).convert("RGB").resize((S, S), Image.NEAREST), (c * S, H + r * S))
panel.save(os.path.join(D, "panel_selfdistill_val10.jpg"), quality=92)
