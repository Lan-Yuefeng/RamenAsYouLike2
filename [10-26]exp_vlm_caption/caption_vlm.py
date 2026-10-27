"""UEC-Ramen555 の領域 caption を公開 VLM (Qwen3-VL) で付ける (2026-10-26)。

ramen_captions.py の手書きテンプレートを置き換える候補。VLM には正解画像・領域を強調した画像・
クラス名・スープ種ラベルを見せる (学習時だけ使える情報)。出力は 1 画像 1 JSON:
  {"image", "soup", "global": str, "regions": {class_name: str}}

使い方:
  python caption_vlm.py --split ../data/ramen/train.txt --limit 10 --out captions_trial
  既存の JSON のスープだけ作り直す: --only_class soup
  複数 GPU は --shard i --num_shards n を GPU ごとに起動する
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset.ramen_captions import CLASS_ID_TO_NAME, load_soup_labels  # noqa: E402

DATASET_ROOT = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

SOUP_HINT = {
    "salt": "shio (salt) ramen broth",
    "soysauce": "shoyu (soy sauce) ramen broth",
    "miso": "miso ramen broth",
    "porkbone": "tonkotsu (pork bone) ramen broth",
    "spicy": "spicy ramen broth",
}

CLASS_HINT = {
    "background": "the background around the bowl (table, tray, etc.)",
    "bowl": "the ramen bowl itself",
    "soup": "the broth with noodles",
    "spoon": "a spoon",
    "chopsticks": "chopsticks",
    "cut_egg": "a halved boiled egg",
    "egg": "a whole boiled egg",
    "seaweed": "nori seaweed",
    "meat": "chashu pork",
    "menma": "menma (bamboo shoots)",
    "naruto": "naruto fish cake",
}

REGION_PROMPT = (
    "Image 1 is a photo of a bowl of ramen. Image 2 is a crop of the same photo showing only one "
    "region; everything outside the region is painted flat gray.\n"
    "The region is: {hint}.{note}\n"
    "Describe only what is inside the region, as it looks in this photo: its color, texture, shape "
    "and surface details. Do not mention the gray fill or anything outside the region. Write one "
    "sentence of at most 25 words, starting with the object name."
)

# caption の書き出し。v1 では塩・豚骨のスープ caption の約半分に種類名が入らなかったので、必ず名前から書かせる
SOUP_NAME = {
    "salt": "shio (salt)",
    "soysauce": "shoyu (soy sauce)",
    "miso": "miso",
    "porkbone": "tonkotsu (pork bone)",
    "spicy": "spicy",
}

# 器と背景は中身まで書きがちなので個別に釘を刺す
CLASS_NOTE = {
    "background": " Describe only the table or surface around the bowl; do not mention the bowl or the food.",
    "bowl": " Describe only the bowl's material, color, pattern and rim; do not mention the food or broth in it.",
}

GLOBAL_PROMPT = (
    "This is a photo of a bowl of {soup}. The visible items are: {items}.\n"
    "Write a caption for this photo in 2 sentences, at most 60 words: the kind of ramen and how "
    "the broth looks, the toppings and their arrangement, the bowl, the background and lighting. "
    "Describe what you see; do not mention items that are not listed."
)


def region_crop(image, mask, margin=0.1, min_side=128):
    """領域外を灰色で塗り、領域の外接矩形（少し余白あり）で切り出す。"""
    arr = np.asarray(image).copy()
    arr[~mask] = 128
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    my, mx = int((ys.max() - ys.min()) * margin) + 4, int((xs.max() - xs.min()) * margin) + 4
    crop = Image.fromarray(arr[max(ys.min() - my, 0): ys.max() + my + 1, max(xs.min() - mx, 0): xs.max() + mx + 1])
    if min(crop.size) < min_side:  # 小さい具材は拡大して見せる
        s = min_side / min(crop.size)
        crop = crop.resize((round(crop.width * s), round(crop.height * s)), Image.BICUBIC)
    return crop


def load_model():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    return model, processor


@torch.no_grad()
def ask(model, processor, images, text, max_new_tokens=120, repetition_penalty=1.0):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=repetition_penalty)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--repetition_penalty", type=float, default=1.0,
                    help="同じ文字列を繰り返す崩れ (0096 の bowl) を作り直すときだけ 1.1 にする")
    ap.add_argument("--only_class", default=None,
                    help="既存の JSON のこのクラスだけ作り直す (v2 でスープだけ直したときに使った)")
    ap.add_argument("--max_side", type=int, default=768, help="VLM に渡す画像の長辺")
    args = ap.parse_args()

    names = [l.strip() for l in open(args.split) if l.strip()]
    if args.limit:
        names = names[: args.limit]
    names = names[args.shard :: args.num_shards]
    soups = load_soup_labels(DATASET_ROOT)
    os.makedirs(args.out, exist_ok=True)
    model, processor = load_model()

    for name in names:
        dst = os.path.join(args.out, name.replace(".png", ".json"))
        if args.only_class:
            prev = json.load(open(dst))
            if prev.get("redone") == args.only_class:
                continue
        elif os.path.exists(dst):
            continue
        image = Image.open(os.path.join(DATASET_ROOT, "Ramen_images", name)).convert("RGB")
        segm = np.array(Image.open(os.path.join(DATASET_ROOT, "Ramen_masks", name)))
        scale = args.max_side / max(image.size)
        if scale < 1:
            size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(size, Image.BICUBIC)
            segm = np.array(Image.fromarray(segm).resize(size, Image.NEAREST))
        soup = soups[name]

        regions = {}
        for label_id in np.unique(segm).tolist():
            cls = CLASS_ID_TO_NAME[label_id]
            if args.only_class and cls != args.only_class:
                continue
            note = (f" This is a bowl of {SOUP_HINT[soup]}. Start the sentence with "
                    f"\"The {SOUP_NAME[soup]} broth\"." if cls == "soup" else CLASS_NOTE.get(cls, ""))
            crop = region_crop(image, segm == label_id)
            regions[cls] = ask(model, processor, [image, crop],
                               REGION_PROMPT.format(hint=CLASS_HINT[cls], note=note),
                               repetition_penalty=args.repetition_penalty)
        if args.only_class:
            prev["regions"].update(regions)
            prev["redone"] = args.only_class
            json.dump(prev, open(dst, "w"), ensure_ascii=False, indent=1)
            print(name, soup, regions, flush=True)
            continue
        items = ", ".join(CLASS_HINT[c] for c in regions if c not in ("background", "bowl", "soup"))
        glob = ask(model, processor, [image],
                   GLOBAL_PROMPT.format(soup=SOUP_HINT[soup], items=items or "none"), 160,
                   args.repetition_penalty)

        json.dump({"image": name, "soup": soup, "global": glob, "regions": regions},
                  open(dst, "w"), ensure_ascii=False, indent=1)
        print(name, soup, json.dumps(regions, ensure_ascii=False)[:200], flush=True)


if __name__ == "__main__":
    main()
