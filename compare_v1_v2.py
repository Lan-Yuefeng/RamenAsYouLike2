"""v1 (result_f2/cfg3.0_steps16) と v2 (result_f2/v2_cfg3.0_steps16) の比較パネルと境界 recall。
各行: マスク | GT | v1 | v2。同じ入力・seed・CFG 3.0・16 step。"""
import glob, os
import numpy as np
from PIL import Image, ImageDraw
from utils.boundary_metric import boundary_score, load_rgb

RAMEN_IMAGES = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555/Ramen_images"
V1, V2, OUT = "result_f2/cfg3.0_steps16", "result_f2/v2_cfg3.0_steps16", "result_f2/v2_cfg3.0_steps16/compare_v1_v2.jpg"
c, head = 512, 28  # 縮小しない (生成と同じ 512²)
stems = [os.path.splitext(os.path.basename(p))[0] for p in sorted(glob.glob(f"{V2}/input/*.png"))]
p = Image.new("RGB", (4 * c, head + c * len(stems)), "white")
d = ImageDraw.Draw(p)
for j, n in enumerate(["mask", "ground truth", "v1 ckpt-3000", "v2 ckpt-3000"]):
    d.text((j * c + 4, 6), n, fill="black")
rec = {"v1": [], "v2": []}
for i, s in enumerate(stems):
    mask = Image.open(f"{V2}/input/{s}.png").convert("RGB")
    gt = Image.open(os.path.join(RAMEN_IMAGES, s.split("_")[-1] + ".png")).convert("RGB")
    for j, im in enumerate([mask, gt, Image.open(f"{V1}/generated/{s}_0.jpg"), Image.open(f"{V2}/generated/{s}_0.jpg")]):
        im = im.convert("RGB")
        p.paste(im if im.size == (c, c) else im.resize((c, c), Image.NEAREST if j == 0 else Image.BICUBIC), (j * c, head + i * c))
    m = np.asarray(mask.resize((512, 512), Image.NEAREST))
    for k, v in [("v1", V1), ("v2", V2)]:
        rec[k].append(boundary_score(load_rgb(f"{v}/generated/{s}_0.jpg"), m, 3)[0])
p.save(OUT, quality=90)
for k in rec:
    print(k, "boundary recall mean %.3f" % np.mean(rec[k]), " ".join("%.2f" % x for x in rec[k]))
