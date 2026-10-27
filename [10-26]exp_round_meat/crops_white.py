"""gen_white.py の結果の切り出し。行 = (例, caption)、列 = seed。"""
import os
import numpy as np
from PIL import Image, ImageDraw
H = os.path.dirname(os.path.abspath(__file__))
C, T = 192, 16
CEN = {"ramen_0014_ell2.5": None, "demo027": (415, 262)}
m = np.array(Image.open(os.path.join(H, "inputs/ramen_0014_ell2.5.png")).convert("RGB"))
ys, xs = np.where((m == (0, 63, 255)).all(-1)); CEN["ramen_0014_ell2.5"] = (int(xs.mean()), int(ys.mean()))
caps, seeds = ["orig", "nowhite", "rolled"], range(1000, 1006)
P = Image.new("RGB", (C * 6, (C + T) * 6), "white"); d = ImageDraw.Draw(P); i = 0
for case, (cx, cy) in CEN.items():
    x0, y0 = min(max(cx - C // 2, 0), 512 - C), min(max(cy - C // 2, 0), 512 - C)
    for cn in caps:
        for j, sd in enumerate(seeds):
            P.paste(Image.open(os.path.join(H, f"results_white/{cn}/{case}_{sd}.jpg")).crop((x0, y0, x0 + C, y0 + C)), (j * C, i * (C + T) + T))
            d.text((j * C + 3, i * (C + T) + 2), f"{case[-9:]} {cn} s{sd}", fill="black")
        i += 1
P.save(os.path.join(H, "crops_white.jpg"), quality=92)
