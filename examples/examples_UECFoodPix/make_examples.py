"""Build infer.py inputs from the held-out UECFOODPIXCOMPLETE test split.

    PYTHONPATH=. .venv/bin/python examples_UECFoodPix/make_examples.py

For each selected test image this writes, next to this file:
    <stem>.png   RGB colour-coded segmentation map (one flat colour per class)
    <stem>.json  {"caption", "seed", "segments_info":[{"color","text"}]}
    gt/<stem>.jpg  the ground-truth photo, for side-by-side comparison

Two details keep inference in-distribution with the training run:

  * "caption": null. Training used global_caption=None (the ADE20K-style choice), so
    encode_all_prompt sets the global text length to 0 and builds the pooled CLIP vector
    from the joined regional prompts. Putting a sentence here instead would feed the model
    global text tokens it never saw while training.
  * Region text comes from dataset.uecfoodpix_captions.regional_caption, the same function
    the dataset used, so id 101 reads "food" here exactly as it did in training.

Masks are resized to 512x512 (nearest) to match the training resolution.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.uecfoodpix_captions import CLASS_ID_TO_NAME, regional_caption

ROOT = '/export/ssd3/lan-y/UECFOODPIXCOMPLETE/data'
OUT = os.path.dirname(os.path.abspath(__file__))
RES = 512

# Hand-picked from test1000.txt for coverage: multi-dish set-meal layouts first, then
# simpler plates, then a couple of single-dish images.
STEMS = None  # filled in by pick()


def colour(class_id):
    """Deterministic, well-separated RGB per class id. 0 -> black, as in examples_Ramen."""
    if class_id == 0:
        return (0, 0, 0)
    h = (class_id * 47) % 180 + 10           # spread hues, avoid the very ends
    s = 200 + (class_id * 13) % 55
    v = 170 + (class_id * 29) % 85
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 255.0, s / 255.0, v / 255.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def pick(n_multi=5, n_single=3):
    """Choose test images: the most multi-dish ones, plus a few single-dish ones."""
    stems = [l.strip() for l in open(f'{ROOT}/test1000.txt') if l.strip()]
    scored = []
    for s in stems:
        m = np.array(Image.open(f'{ROOT}/UECFoodPIXCOMPLETE/test/mask/{s}.png'))[:, :, 0]
        ids = [int(i) for i in np.unique(m)]
        scored.append((len([i for i in ids if i != 0]), s, ids))
    scored.sort(key=lambda x: -x[0])
    multi = scored[:n_multi]
    # Diversify the single-dish picks: one image per class, not three of the same croissant.
    single, seen = [], set()
    for x in scored:
        if x[0] != 1:
            continue
        cls = next(i for i in x[2] if i != 0)
        if cls in seen:
            continue
        seen.add(cls)
        single.append(x)
        if len(single) == n_single:
            break
    return multi + single


def main():
    os.makedirs(f'{OUT}/gt', exist_ok=True)
    chosen = pick()
    for seed_i, (n_food, stem, ids) in enumerate(chosen):
        m = np.array(Image.open(f'{ROOT}/UECFoodPIXCOMPLETE/test/mask/{stem}.png'))[:, :, 0]
        m = np.array(Image.fromarray(m).resize((RES, RES), Image.NEAREST))

        rgb = np.zeros((RES, RES, 3), dtype=np.uint8)
        segments_info = []
        for i in sorted(int(x) for x in np.unique(m)):
            c = colour(i)
            rgb[m == i] = c
            segments_info.append({"color": list(c), "text": regional_caption(CLASS_ID_TO_NAME[i])})

        Image.fromarray(rgb).save(f'{OUT}/{stem}.png')
        json.dump({"caption": None, "seed": 1000 + seed_i, "segments_info": segments_info},
                  open(f'{OUT}/{stem}.json', 'w'), indent=1, ensure_ascii=False)

        Image.open(f'{ROOT}/UECFoodPIXCOMPLETE/test/img/{stem}.jpg').convert('RGB') \
             .resize((RES, RES), Image.BICUBIC).save(f'{OUT}/gt/{stem}.jpg')

        print(f"  {stem}: {n_food} dishes -> " + ", ".join(s['text'] for s in segments_info))
    print(f"\n{len(chosen)} examples written to {OUT}")


if __name__ == '__main__':
    main()
