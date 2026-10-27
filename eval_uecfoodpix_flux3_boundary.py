"""UECFoodPix 皿版 test10 の境界 recall / precision (utils/boundary_metric.py、flux2_eval_val55.py と同じ指標)。

    /export/ssd2/lan-y/venvs/Seg2Any/.venv-flux3/bin/python eval_uecfoodpix_flux3_boundary.py

- 比べる mask は学習・生成の入力と同じ 512² (id マップを nearest-exact で縮小。Dataset と同じ)
- recall は境界の種類別にも出す: food = 料理 (1..102) が接する境界、plate|table = 皿 (103) とテーブル (0) の境界だけ
- FLUX.2 皿版の入力は自作の皿マスクなので、FLUX.2 は自作マスクでも測る (共有マスクとの一致率は 99.8% 以上)
- 参考に GT 写真 (512² に縮小) も同じ指標で測る
出力: result_f3/UECFOODPIX_flux3/compare_test10/boundary.json と boundary_<stem>.jpg (生成画像に mask 境界を赤で重ねたもの、512² のまま)
"""
import json
import os
import statistics

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from utils.boundary_metric import image_edges, load_rgb

STEMS = "16164 3061 3317 3709 3710 3770 4357 5599 6286 6910".split()
IMG_ROOT = "/export/ssd2/lan-y/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/test/img"
MASK_LAB = "/export/ssd2/lan-y/seg2any_data/uecfoodpix/mask_plate_lab/test"
MASK_OWN = "/export/ssd3/lan-y/seg2any_data/uecfoodpix/mask_plate/test"  # FLUX.2 皿版の入力
CONDS = {
    "FLUX.2 Base 20step CFG4": ("result_f2/UECFOODPIX_flux2/uecfoodpix_plate_flux2_test10_base/gen", MASK_OWN),
    "FLUX.3 Base 20step CFG3.5": ("result_f3/UECFOODPIX_flux3/plate_test10_base20/gen", MASK_LAB),
    "FLUX.3 distilled 16step": ("result_f3/UECFOODPIX_flux3/plate_test10_distill16/gen", MASK_LAB),
}
OUT = "result_f3/UECFOODPIX_flux3/compare_test10"
TOL = 3
PLATE = 103


def load_ids(path, size=512):
    segm = np.array(Image.open(path))
    segm = segm[..., 0] if segm.ndim == 3 else segm
    t = torch.from_numpy(segm.astype(np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="nearest-exact")[0, 0].long().numpy()


def boundaries(ids):
    """境界画素 (右か下の隣と id が違う画素) と、その境界が料理に接するか"""
    b = np.zeros(ids.shape, bool)
    food = np.zeros(ids.shape, bool)
    is_food = (ids >= 1) & (ids < PLATE)
    for dy, dx in ((0, 1), (1, 0)):
        a, c = ids[: ids.shape[0] - dy, : ids.shape[1] - dx], ids[dy:, dx:]
        diff = a != c
        f = diff & (is_food[: ids.shape[0] - dy, : ids.shape[1] - dx] | is_food[dy:, dx:])
        b[: ids.shape[0] - dy, : ids.shape[1] - dx] |= diff
        food[: ids.shape[0] - dy, : ids.shape[1] - dx] |= f
    for m in (b, food):
        m[[0, -1], :] = False
        m[:, [0, -1]] = False
    return b, food


def score(img, ids, tol=TOL):
    b, food = boundaries(ids)
    e = image_edges(img)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    e_d = cv2.dilate(e.astype(np.uint8), k) > 0
    b_d = cv2.dilate(b.astype(np.uint8), k) > 0
    rec = lambda m: float((m & e_d).sum() / max(1, m.sum()))  # noqa: E731
    return {"recall": rec(b), "recall_food": rec(food), "recall_plate_table": rec(b & ~food),
            "precision": float((e & b_d).sum() / max(1, e.sum()))}


def overlay(img, ids):
    b, _ = boundaries(ids)
    b = cv2.dilate(b.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    out = img.copy()
    out[b] = (255, 0, 0)
    return Image.fromarray(out)


if __name__ == "__main__":
    rows = {name: [] for name in ["GT photo (512²)"] + list(CONDS)}
    for s in STEMS:
        lab = load_ids(f"{MASK_LAB}/{s}.png")
        tiles = []
        gt = load_rgb(f"{IMG_ROOT}/{s}.jpg")
        rows["GT photo (512²)"].append(score(gt, lab))
        tiles.append(("GT photo (512²)", overlay(gt, lab)))
        for name, (d, mask_root) in CONDS.items():
            img = load_rgb(f"{d}/{s}.jpg")
            r = score(img, load_ids(f"{mask_root}/{s}.png"))
            if mask_root != MASK_LAB:
                r["recall_lab_mask"] = score(img, lab)["recall"]
            rows[name].append(r)
            tiles.append((f"{name}  recall {r['recall']:.3f}", overlay(img, lab)))
        panel = Image.new("RGB", (512 * len(tiles) + 8 * (len(tiles) - 1), 22 + 512), "white")
        dr = ImageDraw.Draw(panel)
        for i, (t, im) in enumerate(tiles):
            panel.paste(im, (i * 520, 22))
            dr.text((i * 520 + 4, 5), t, fill="black")
        panel.save(f"{OUT}/boundary_{s}.jpg", quality=92)

    summary = {}
    for name, rs in rows.items():
        summary[name] = {k: round(statistics.mean(r[k] for r in rs), 3) for k in rs[0]}
        summary[name]["recall_per_image"] = dict(zip(STEMS, (round(r["recall"], 3) for r in rs)))
    json.dump(summary, open(f"{OUT}/boundary.json", "w"), indent=1)

    keys = ["recall", "recall_food", "recall_plate_table", "precision"]
    print(f"{'':28s}" + "".join(f"{k:>20s}" for k in keys))
    for name, v in summary.items():
        print(f"{name:28s}" + "".join(f"{v[k]:>20.3f}" for k in keys) + (f"   (lab mask recall {v['recall_lab_mask']:.3f})" if "recall_lab_mask" in v else ""))
    print()
    print("per image recall:", "  ".join(f"{s}" for s in STEMS))
    for name, v in summary.items():
        print(f"{name:28s}", "  ".join(f"{x:.2f}" for x in v["recall_per_image"].values()))
