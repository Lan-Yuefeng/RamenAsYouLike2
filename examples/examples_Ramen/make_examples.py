"""Build Seg2Any examples_Ramen/ from held-out UEC-Ramen555 color masks.

Only samples from data/ramen/val.txt (the 55 images excluded from training) are
used, so inference is never run on a mask the LoRA has already seen.

Captions come from dataset/ramen_captions.py -- the same module the training
dataset uses -- so the prompts here are worded exactly like the training ones.

Masks are emitted at 512x512 to match `resolution: 512` in config/seg2any_ramen.yaml.
The dataset is a mix of 4:3 and 16:9, and training squashes everything to a square,
so inference must be squashed the same way; resize the generated image back to the
mask's original aspect ratio afterwards.

Run:
    .venv/bin/python examples_Ramen/make_examples.py
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dataset.ramen_captions import (  # noqa: E402
    CLASS_COLORS,
    TOPPING_ORDER,
    UTENSIL_ORDER,
    global_caption,
    load_soup_labels,
    regional_caption,
)

SRC = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555"
VAL_SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ramen", "val.txt")

# --resolution matches the LoRA's training resolution (512). --out defaults to this
# directory. Emit a larger set (e.g. 1024) into a separate dir so the 512 examples the
# inference command points at are never overwritten.
_p = argparse.ArgumentParser()
_p.add_argument("--resolution", type=int, default=512)
_p.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
_p.add_argument("--num", type=int, default=10)
_args = _p.parse_args()

RESOLUTION = _args.resolution
DST = _args.out
NUM_EXAMPLES = _args.num


def pick(names, soup_labels, info):
    """Greedily pick NUM_EXAMPLES val masks covering every soup type and class."""
    chosen = []
    want_cls = set(TOPPING_ORDER) | set(UTENSIL_ORDER)
    want_soup = {soup_labels[n] for n in names}
    while len(chosen) < NUM_EXAMPLES and len(chosen) < len(names):
        best, best_score = None, None
        for n in names:
            if n in chosen:
                continue
            score = (
                len(info[n] & want_cls) * 2
                + (2 if soup_labels[n] in want_soup else 0)
                + len(info[n]) * 0.01  # tie-break toward richer masks
            )
            if best_score is None or score > best_score:
                best, best_score = n, score
        chosen.append(best)
        want_cls -= info[best]
        want_soup.discard(soup_labels[best])
    return sorted(chosen)


def main():
    soup_labels = load_soup_labels(SRC)
    with open(VAL_SPLIT) as f:
        val_names = [ln.strip() for ln in f if ln.strip()]

    by_color = {v: k for k, v in CLASS_COLORS.items()}
    info = {}
    for n in val_names:
        arr = np.array(Image.open(os.path.join(SRC, "Ramen_color_masks", n)).convert("RGB"))
        present = {tuple(c) for c in np.unique(arr.reshape(-1, 3), axis=0).tolist()}
        assert present <= set(by_color), f"{n}: off-palette colors {present - set(by_color)}"
        info[n] = {by_color[c] for c in present}

    os.makedirs(DST, exist_ok=True)
    for f in os.listdir(DST):  # drop any previous examples in this output dir only
        if f.startswith("ramen_") and f.endswith((".png", ".json")):
            os.remove(os.path.join(DST, f))

    for name in pick(val_names, soup_labels, info):
        mask = Image.open(os.path.join(SRC, "Ramen_color_masks", name)).convert("RGB")
        # NEAREST is mandatory: infer.py matches palette colors exactly, so any
        # interpolated in-between pixel silently drops that region.
        mask = mask.resize((RESOLUTION, RESOLUTION), Image.NEAREST)
        arr = np.array(mask)

        present = [tuple(c) for c in np.unique(arr.reshape(-1, 3), axis=0).tolist()]
        assert set(present) <= set(by_color), f"{name}: off-palette after resize"

        soup_type = soup_labels[name]
        cls_present = [by_color[c] for c in present]

        anno = {
            "caption": global_caption(cls_present, soup_type),
            "seed": 1000 + int(name.split(".")[0]),
            "segments_info": [
                {"color": list(CLASS_COLORS[c]), "text": regional_caption(c, soup_type)}
                for c in cls_present
            ],
        }

        stem = f"ramen_{name.split('.')[0]}"
        mask.save(os.path.join(DST, stem + ".png"))
        with open(os.path.join(DST, stem + ".json"), "w") as f:
            json.dump(anno, f, indent=2)
        print(f"{stem}  soup={soup_type}  classes={cls_present}")


if __name__ == "__main__":
    main()
