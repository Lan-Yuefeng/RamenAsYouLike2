"""SACap-eval (test/) から推論用の例 (色分けマスク png + caption json + 元画像) を作る。

    python examples_SACap_eval/make_examples.py --num 12 --seed 0

- マスクは元のアスペクト比のまま長辺 1024、各辺 32 の倍数に NEAREST で縮小 (cond_scale_factor=2 で割り切れるように)
- 重なりは面積の大きい順に塗る (小さい物体が上に残る)
- 色はパレットから一意に割り当てる。黒 (0,0,0) は未ラベル扱いなので使わない
"""
import argparse
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from PIL import Image

SACAP = "/export/ssd3/lan-y/SACap-1M"
OUT = os.path.dirname(os.path.abspath(__file__))


def palette(n, rng):
    colors = set()
    while len(colors) < n:
        c = tuple(int(x) for x in rng.integers(40, 256, 3))
        colors.add(c)
    return list(colors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--long_side", type=int, default=1024)
    args = ap.parse_args()

    df = pd.read_parquet(f"{SACAP}/annotations/anno_test.parquet")
    rows = df.sample(n=args.num, random_state=args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(OUT, "gt"), exist_ok=True)

    for _, r in rows.iterrows():
        stem = os.path.splitext(r.imagename)[0]
        with open(f"{SACAP}/test/{stem}.json") as f:
            anns = json.load(f)
        caps = {s["anno_id"]: s["caption"] for s in r.segments_info}
        segs = [(a["id"], mask_util.decode(a["segmentation"]) == 1) for a in anns["annotations"] if a["id"] in caps]
        segs.sort(key=lambda x: -x[1].sum())

        h, w = segs[0][1].shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        colors = palette(len(segs), rng)
        info = []
        for (aid, m), c in zip(segs, colors):
            canvas[m] = c
            info.append({"color": list(c), "text": caps[aid]})

        s = args.long_side / max(h, w)
        nw, nh = max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)
        mask = Image.fromarray(canvas).resize((nw, nh), Image.NEAREST)
        # 縮小で消えた領域は json から外す (infer 側は色が無い領域を黙って無視するが、念のため)
        present = {tuple(c) for c in np.unique(np.array(mask).reshape(-1, 3), axis=0)}
        info = [x for x in info if tuple(x["color"]) in present]

        mask.save(os.path.join(OUT, f"{stem}.png"))
        with open(os.path.join(OUT, f"{stem}.json"), "w") as f:
            json.dump({"caption": r.caption, "seed": random.Random(stem).randint(0, 10000), "segments_info": info}, f, indent=2, ensure_ascii=False)
        shutil.copy(f"{SACAP}/test/{r.imagename}", os.path.join(OUT, "gt", r.imagename))
        print(stem, (nw, nh), len(info), "regions")


if __name__ == "__main__":
    main()
