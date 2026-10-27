"""Build Seg2Any examples for a fixed list of image names (the cho pix2pix
comparison set, i.e. cho's original test-set images that still have per-label
A_cp annotations and exist in the current UEC-Ramen555 555-image release).

Unlike make_examples.py this does NOT restrict to val.txt only -- some of the
requested names were part of Seg2Any's LoRA train split. Each output is
annotated (via a sidecar note printed to stdout) with whether Seg2Any saw it
during training, so the comparison stays honest.

Run:
    .venv/bin/python examples_Ramen/make_examples_cho_compare.py
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dataset.ramen_captions import CLASS_COLORS, global_caption, load_soup_labels  # noqa: E402

SRC = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555"
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SPLIT = os.path.join(HERE, "..", "data", "ramen", "train.txt")
VAL_SPLIT = os.path.join(HERE, "..", "data", "ramen", "val.txt")
DST = os.path.join(HERE, "..", "examples_Ramen_ChoCompare")
RESOLUTION = 512

NAMES = [
    "0513.png", "0514.png", "0515.png", "0516.png", "0517.png", "0518.png",
    "0519.png", "0520.png", "0521.png", "0522.png", "0523.png", "0524.png",
    "0528.png", "0553.png",
]


def main():
    soup_labels = load_soup_labels(SRC)
    val_names = set(ln.strip() for ln in open(VAL_SPLIT) if ln.strip())
    train_names = set(ln.strip() for ln in open(TRAIN_SPLIT) if ln.strip())

    by_color = {v: k for k, v in CLASS_COLORS.items()}
    os.makedirs(DST, exist_ok=True)
    for f in os.listdir(DST):
        if f.startswith("ramen_") and f.endswith((".png", ".json")):
            os.remove(os.path.join(DST, f))

    for name in NAMES:
        seen = "val(unseen-by-seg2any)" if name in val_names else (
            "train(seen-by-seg2any)" if name in train_names else "NOT-IN-CURRENT-555")

        mask = Image.open(os.path.join(SRC, "Ramen_color_masks", name)).convert("RGB")
        mask = mask.resize((RESOLUTION, RESOLUTION), Image.NEAREST)
        arr = np.array(mask)

        present = [tuple(c) for c in np.unique(arr.reshape(-1, 3), axis=0).tolist()]
        assert set(present) <= set(by_color), f"{name}: off-palette after resize {set(present)-set(by_color)}"

        soup_type = soup_labels[name]
        cls_present = [by_color[c] for c in present]

        anno = {
            "caption": global_caption(cls_present, soup_type),
            "seed": 1000 + int(name.split(".")[0]),
            "segments_info": [
                {"color": list(CLASS_COLORS[c]), "text": __import__("dataset.ramen_captions", fromlist=["regional_caption"]).regional_caption(c, soup_type)}
                for c in cls_present
            ],
        }

        stem = f"ramen_{name.split('.')[0]}"
        mask.save(os.path.join(DST, stem + ".png"))
        with open(os.path.join(DST, stem + ".json"), "w") as f:
            json.dump(anno, f, indent=2)
        print(f"{stem}  soup={soup_type}  seg2any_split={seen}  classes={cls_present}")


if __name__ == "__main__":
    main()
