"""行 = 画像、列 = 元マスク / 円マスク / 各条件。原寸で並べる。panel.jpg と行ごとの rows/*.jpg を出す。"""
import os, sys
from PIL import Image, ImageDraw
H = os.path.dirname(os.path.abspath(__file__))
conds = sys.argv[1:] or ["orig", "circle", "circle_noegg", "condC_labO", "condO_labC", "circle_cap_meatonly"]
stems = sorted(f[:-4] for f in os.listdir(os.path.join(H, "results", conds[0])))
cols = [("mask orig", lambda s: f"inputs/{s}_orig.png"), ("mask circle", lambda s: f"inputs/{s}_circle.png")] + \
       [(c, lambda s, c=c: f"results/{c}/{s}.jpg") for c in conds]
S, T = 512, 28
os.makedirs(os.path.join(H, "rows"), exist_ok=True)
rows = []
for s in stems:
    r = Image.new("RGB", (S * len(cols), S + T), "white"); d = ImageDraw.Draw(r)
    for j, (name, f) in enumerate(cols):
        r.paste(Image.open(os.path.join(H, f(s))).convert("RGB").resize((S, S), Image.NEAREST), (j * S, T))
        d.text((j * S + 6, 6), f"{s}  {name}", fill="black")
    r.save(os.path.join(H, "rows", s + ".jpg"), quality=90); rows.append(r)
P = Image.new("RGB", (rows[0].width, len(rows) * rows[0].height), "white")
for i, r in enumerate(rows): P.paste(r, (0, i * r.height))
P.save(os.path.join(H, "panel.jpg"), quality=90)
