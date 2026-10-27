"""VLM caption の抜けと長さを確かめる (2026-10-26)。

- 原寸マスクにあるクラスが全部 regions にあるか (caption_vlm.py は縮小したマスクで領域を取るので、小さな領域が消えうる)
- regional caption が regional_max_sequence_length (50 トークン) に収まるか。Qwen3 系のトークナイザで数える
  python check_captions.py --captions captions_all
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset.ramen_captions import CLASS_ID_TO_NAME  # noqa: E402

DATASET_ROOT = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--max_tokens", type=int, default=50)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    paths = sorted(glob.glob(os.path.join(args.captions, "*.json")))
    missing, too_long, lens = [], [], []
    for p in paths:
        d = json.load(open(p))
        segm = np.array(Image.open(os.path.join(DATASET_ROOT, "Ramen_masks", d["image"])))
        for cid in np.unique(segm).tolist():
            if CLASS_ID_TO_NAME[cid] not in d["regions"]:
                missing.append((d["image"], CLASS_ID_TO_NAME[cid], int((segm == cid).sum())))
        for c, v in d["regions"].items():
            n = len(tok(v)["input_ids"])
            lens.append(n)
            if n > args.max_tokens:
                too_long.append((d["image"], c, n))
    lens = np.array(lens)
    print(f"{len(paths)} files, {len(lens)} regions, tokens mean {lens.mean():.1f} max {lens.max()}")
    print(f"missing classes: {len(missing)}", missing[:20])
    print(f"over {args.max_tokens} tokens: {len(too_long)}", too_long[:20])


if __name__ == "__main__":
    main()
