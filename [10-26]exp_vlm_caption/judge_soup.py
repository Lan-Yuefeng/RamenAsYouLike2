"""画像のスープ種を VLM に 5 択で答えさせる (2026-10-26)。生成画像の「塩と入れたら塩に見えるか」の評価用。

まず本物の写真 (val55) で正解率を測り、判定器として使えるか確かめる。
  python judge_soup.py --images /export/space0/lan-y/UEC-Ramen555/UEC-Ramen555/Ramen_images \
      --split ../data/ramen/val.txt --out judge_real_val55.json
生成画像は --pattern 'ramen_{stem}_0.jpg' でファイル名を合わせる。
"""
import argparse
import json
import os
import sys
from collections import Counter

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset.ramen_captions import SOUP_TYPES, load_soup_labels  # noqa: E402

DATASET_ROOT = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555"

# 選択肢の記号 -> スープ種。記号で答えさせると出力の揺れを拾いやすい
CHOICES = {
    "A": ("salt", "shio (salt) ramen: clear, pale golden broth"),
    "B": ("soysauce", "shoyu (soy sauce) ramen: clear, brown broth"),
    "C": ("miso", "miso ramen: opaque, yellowish to reddish-brown broth"),
    "D": ("porkbone", "tonkotsu (pork bone) ramen: creamy, milky white broth"),
    "E": ("spicy", "spicy ramen: red broth with chili oil"),
}
PROMPT = (
    "What kind of ramen is this? Judge from the broth.\n"
    + "\n".join(f"{k}. {v[1]}" for k, v in CHOICES.items())
    + "\nAnswer with a single letter."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--pattern", default="{stem}.png",
                    help="画像のファイル名。生成画像なら ramen_{stem}_0.jpg")
    ap.add_argument("--max_side", type=int, default=768)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    soups = load_soup_labels(DATASET_ROOT)
    names = [l.strip() for l in open(args.split) if l.strip()]

    results = {}
    for name in names:
        image = Image.open(os.path.join(args.images, args.pattern.format(stem=name[:-4]))).convert("RGB")
        image.thumbnail((args.max_side, args.max_side), Image.BICUBIC)
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
        ans = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = CHOICES.get(ans[:1].upper(), ("invalid",))[0]
        results[name] = {"gt": soups[name], "pred": pred, "raw": ans}

    acc = sum(r["gt"] == r["pred"] for r in results.values()) / len(results)
    conf = Counter((r["gt"], r["pred"]) for r in results.values())
    print(f"{args.model}  n={len(results)}  acc={acc:.3f}")
    print("gt \\ pred".ljust(10) + "".join(s[:8].rjust(9) for s in SOUP_TYPES + ["invalid"]))
    for g in SOUP_TYPES:
        print(g.ljust(10) + "".join(str(conf[(g, p)]).rjust(9) for p in SOUP_TYPES + ["invalid"]))
    json.dump({"model": args.model, "acc": acc, "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
