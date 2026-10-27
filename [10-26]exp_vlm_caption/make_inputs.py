"""caption の JSON ({"global", "regions"}) から Seg2Any 推論の入力 (examples 形式) を作る (2026-10-26)。

マスク画像と seed は examples/examples_Ramen_val55 のものを使う (exp_ablation の inputs_val10 と同じ画像・seed)。
  python make_inputs.py --captions captions_all --split val10.txt --out inputs/val10_oracle
  python make_inputs.py --short short --split val10.txt --out inputs/val10_short    # 短いプロンプトをそのまま
--short を付けると short_prompts/ のその段階を regional に、全領域をつないだものを global に使う。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from dataset.ramen_captions import CLASS_COLORS  # noqa: E402

VAL55 = os.path.join(os.path.dirname(HERE), "examples", "examples_Ramen_val55")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default=None)
    ap.add_argument("--short", default=None, choices=["name", "short", "detail"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert (args.captions is None) != (args.short is None)
    color2name = {tuple(v): k for k, v in CLASS_COLORS.items()}
    os.makedirs(args.out, exist_ok=True)

    for name in [l.strip() for l in open(args.split) if l.strip()]:
        stem = name.replace(".png", "")
        base = json.load(open(os.path.join(VAL55, f"ramen_{stem}.json")))
        if args.short:
            sp = json.load(open(os.path.join(HERE, "short_prompts", stem + ".json")))["regions"]
            regions = {c: v[args.short] for c, v in sp.items()}
            glob = "A top-down photo of a bowl of ramen: " + ", ".join(regions.values()) + "."
        else:
            d = json.load(open(os.path.join(args.captions, stem + ".json")))
            regions, glob = d["regions"], d["global"]
        segs = []
        for seg in base["segments_info"]:
            segs.append({"color": seg["color"], "text": regions[color2name[tuple(seg["color"])]]})
        json.dump({"caption": glob, "seed": base["seed"], "segments_info": segs},
                  open(os.path.join(args.out, f"ramen_{stem}.json"), "w"), indent=2, ensure_ascii=False)
        link = os.path.join(args.out, f"ramen_{stem}.png")
        if not os.path.lexists(link):
            os.symlink(os.path.join(VAL55, f"ramen_{stem}.png"), link)
    print("ok", args.out)


if __name__ == "__main__":
    main()
