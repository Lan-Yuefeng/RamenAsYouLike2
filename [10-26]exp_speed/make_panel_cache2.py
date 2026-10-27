"""run_cache2.py の結果パネル。列 = 入力マスク / base / ① / ①+② / ①+③ (原寸 512)。"""
import os, glob
from PIL import Image, ImageDraw
D = os.path.dirname(os.path.abspath(__file__))
cols = [("mask", None), ("base", "cache_base"), ("1 fuse", "cache_fuse"), ("1+2 fuse_cond_cache", "cache_fuse_cond_cache"), ("1+3 fuse_cache", "cache_fuse_cache")]
stems = [os.path.basename(p)[:-4] for p in sorted(glob.glob(os.path.join(glob.escape(D), "val10", "*.png")))]
S, H = 512, 40
panel = Image.new("RGB", (S * len(cols), H + S * len(stems)), "white")
d = ImageDraw.Draw(panel)
for c, (name, _) in enumerate(cols):
    d.text((c * S + 10, 12), name, fill="black")
for r, st in enumerate(stems):
    for c, (_, sub) in enumerate(cols):
        p = os.path.join(D, "val10", st + ".png") if sub is None else os.path.join(D, "result", sub, st + "_0.jpg")
        panel.paste(Image.open(p).convert("RGB").resize((S, S), Image.NEAREST), (c * S, H + r * S))
panel.save(os.path.join(D, "panel_cache2_val10.jpg"), quality=92)
