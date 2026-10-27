"""大きさ実験の行パネル (rows_size/)。列 = 楕円マスク 0.7% / 結果 ell0.7 / ell0.7_noegg / 1.5 / ... (原寸)。"""
import os
from PIL import Image, ImageDraw
H = os.path.dirname(os.path.abspath(__file__))
PS = ["0.7", "1.5", "2.5"]
stems = sorted(f[:-4] for f in os.listdir(os.path.join(H, "results/ell0.7")))
S, T = 512, 28
os.makedirs(os.path.join(H, "rows_size"), exist_ok=True)
for s in stems:
    cols = [("orig result", f"results/orig/{s}.jpg")]
    for p in PS:
        cols += [(f"mask ell{p}", f"inputs/{s}_ell{p}.png"), (f"ell{p}", f"results/ell{p}/{s}.jpg"), (f"ell{p}_noegg", f"results/ell{p}_noegg/{s}.jpg")]
    r = Image.new("RGB", (S * len(cols), S + T), "white"); d = ImageDraw.Draw(r)
    for j, (n, f) in enumerate(cols):
        r.paste(Image.open(os.path.join(H, f)).convert("RGB"), (j * S, T)); d.text((j * S + 6, 6), f"{s} {n}", fill="black")
    r.save(os.path.join(H, "rows_size", s + ".jpg"), quality=90)
