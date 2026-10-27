"""val55 の先頭 10 枚から、regional caption を変えた入力を作る (マスク画像はシンボリックリンク)。

inputs_val10_classname   : regional caption を具材名だけにする ("menma" など。soup は "soup")
inputs_val10_noregional  : regional caption を空文字にする (領域の意味は global caption と輪郭 cond だけで伝わる)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from dataset.ramen_captions import CLASS_COLORS  # noqa: E402

color2name = {tuple(v): k for k, v in CLASS_COLORS.items()}
src = os.path.join(HERE, "inputs_val10")
for variant in ["classname", "noregional"]:
    dst = os.path.join(HERE, f"inputs_val10_{variant}")
    os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(src)):
        if f.endswith(".png"):
            link = os.path.join(dst, f)
            if not os.path.lexists(link):
                os.symlink(os.path.join("../../examples/examples_Ramen_val55", f), link)
        elif f.endswith(".json"):
            a = json.load(open(os.path.join(src, f)))
            for seg in a["segments_info"]:
                seg["text"] = color2name[tuple(seg["color"])].replace("_", " ") if variant == "classname" else ""
            json.dump(a, open(os.path.join(dst, f), "w"), indent=2, ensure_ascii=False)
print("ok")
