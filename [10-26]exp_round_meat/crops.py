"""楕円の周り 192px を切り出して並べる (確認用 crops.jpg)。行 = 画像、列 = 条件。"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw
H = os.path.dirname(os.path.abspath(__file__))
conds = sys.argv[1:] or ["ell0.7", "ell0.7_noegg", "ell1.5", "ell1.5_noegg", "ell2.5", "ell2.5_noegg"]
stems = sorted(f[:-4] for f in os.listdir(os.path.join(H, "results", conds[0])))
C, T = 192, 16
P = Image.new("RGB", (C * len(conds), (C + T) * len(stems)), "white"); d = ImageDraw.Draw(P)
for i, s in enumerate(stems):
    m = np.array(Image.open(os.path.join(H, f"inputs/{s}_ell1.5.png")).convert("RGB"))
    ys, xs = np.where((m == (0, 63, 255)).all(-1)); cy, cx = int(ys.mean()), int(xs.mean())
    x0, y0 = min(max(cx - C // 2, 0), 512 - C), min(max(cy - C // 2, 0), 512 - C)
    for j, c in enumerate(conds):
        im = Image.open(os.path.join(H, f"results/{c}/{s}.jpg")).crop((x0, y0, x0 + C, y0 + C))
        P.paste(im, (j * C, i * (C + T) + T)); d.text((j * C + 3, i * (C + T) + 2), f"{s[-4:]} {c}", fill="black")
P.save(os.path.join(H, sys.argv[0].replace(".py", "") + "_" + "_".join(conds)[:40] + ".jpg"), quality=92)
print(P.size)
