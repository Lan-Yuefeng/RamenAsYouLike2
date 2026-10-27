"""UECFOODPIXCOMPLETE の料理マスクと UECFOODPIX_Plate の皿マスクを 1 枚の 1ch PNG にまとめる。

    .venv-flux2/bin/python dataset/make_uecfoodpix_plate_masks.py

UECFOODPIX_Plate/<料理クラス id>/<stem>.png は 473x473 の 2 値 RGBA (R=1 が皿)。同じ画像が、写っている料理クラスの
数だけフォルダに入っていて中身が少しずつ違うので、全フォルダの和を取って元画像サイズに NEAREST で戻す。
料理画素は料理マスクを優先する (重なりは 0.05% 未満)。

出力 <OUT>/{train,test}/<stem>.png : 0=テーブル等, 1..102=料理, 103=皿・器・お盆 (dataset/uecfoodpix_plate_captions.py)
"""
import collections
import os

import numpy as np
from joblib import Parallel, delayed
from PIL import Image

ROOT = "/export/ssd3/lan-y/UECFOODPIXCOMPLETE/data"
PLATE = "/export/ssd3/lan-y/UECFOODPIX_Plate"
OUT = "/export/ssd3/lan-y/seg2any_data/uecfoodpix/mask_plate"
PLATE_ID = 103


def build(stem, split, plate_dirs):
    food = np.array(Image.open(f"{ROOT}/UECFoodPIXCOMPLETE/{split}/mask/{stem}.png"))[..., 0]
    h, w = food.shape
    plate = np.zeros((h, w), bool)
    for c in plate_dirs:
        m = Image.open(f"{PLATE}/{c}/{stem}.png")
        plate |= np.array(m.resize((w, h), Image.NEAREST))[..., 0] > 0
    out = food.copy()
    out[plate & (food == 0)] = PLATE_ID
    Image.fromarray(out.astype(np.uint8), mode="L").save(f"{OUT}/{split}/{stem}.png")
    return float((out == PLATE_ID).mean()), len(plate_dirs) > 0


def main():
    where = collections.defaultdict(list)
    for c in os.listdir(PLATE):
        for f in os.listdir(f"{PLATE}/{c}"):
            where[f[:-4]].append(c)
    for split, txt in [("train", "train9000.txt"), ("test", "test1000.txt")]:
        os.makedirs(f"{OUT}/{split}", exist_ok=True)
        stems = [l.strip() for l in open(f"{ROOT}/{txt}") if l.strip()]
        res = Parallel(n_jobs=16)(delayed(build)(s, split, where.get(s, [])) for s in stems)
        frac = np.array([r[0] for r in res])
        print(f"{split}: {len(stems)} masks, plate mask missing for {sum(not r[1] for r in res)}, "
              f"plate pixels mean {frac.mean():.3f}, images with no plate pixels {(frac == 0).sum()}")


if __name__ == "__main__":
    main()
