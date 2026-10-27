"""VLM caption から、ユーザーが打ちそうな短い英語プロンプトを作る (2026-10-26)。拡張器の学習データの入力側。

正解画像は見せず、caption の文面だけから Qwen3-VL (テキストのみ) に書かせる。手書きの言い換え表を作らないため。
1 画像 1 回の呼び出しで、全領域についてくわしさの違う 3 通りを出させる:
  name: 物の名前だけ (1-3 語)  /  short: 特徴を 1 つ足す (2-4 語)  /  detail: 6 語まで
出力: <out>/<name>.json = {"image", "soup", "regions": {class: {"name", "short", "detail"}}}
  python make_short_prompts.py --captions captions_all --out short_prompts
"""
import argparse
import json
import os
import re

import torch

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

PROMPT = (
    "Below are captions for regions of a ramen photo, one per line as `region: caption`.\n"
    "For each region, write the short prompts a user might type to ask an image generator for that "
    "region, in plain English, at three levels of detail:\n"
    '- "name": just what it is, 1-3 words (e.g. "chashu pork")\n'
    '- "short": 2-4 words, the name plus one key visual trait\n'
    '- "detail": at most 6 words\n'
    "For the soup region, always keep the kind of ramen broth (e.g. shio, shoyu, miso, tonkotsu, spicy) "
    "in every level. Use only information in the caption.\n"
    "Return only a JSON object mapping each region to {{\"name\": ..., \"short\": ..., \"detail\": ...}}.\n\n"
    "{lines}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    os.makedirs(args.out, exist_ok=True)

    files = sorted(os.listdir(args.captions))[args.shard :: args.num_shards]
    for f in files:
        dst = os.path.join(args.out, f)
        if os.path.exists(dst):
            continue
        d = json.load(open(os.path.join(args.captions, f)))
        lines = "\n".join(f"{c}: {v}" for c, v in d["regions"].items())
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(lines=lines)}]}],
            tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=600, do_sample=False)
        text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            regions = json.loads(m.group(0))
            assert set(regions) == set(d["regions"]), f"regions differ: {set(regions) ^ set(d['regions'])}"
            assert all(set(v) == {"name", "short", "detail"} for v in regions.values())
        except Exception as e:  # 崩れたものは飛ばし、あとで数えて作り直す
            print("FAILED", f, repr(e)[:200], flush=True)
            continue
        json.dump({"image": d["image"], "soup": d["soup"], "regions": regions},
                  open(dst, "w"), ensure_ascii=False, indent=1)
        print(f, json.dumps(regions, ensure_ascii=False)[:160], flush=True)


if __name__ == "__main__":
    main()
