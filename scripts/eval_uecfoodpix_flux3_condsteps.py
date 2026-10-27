"""cond (輪郭) を最初の k ステップだけ入れたときの境界 recall と見た目 (UECFoodPix 皿版 test10)。

    /export/ssd2/lan-y/venvs/Seg2Any/.venv-flux3/bin/python scripts/eval_uecfoodpix_flux3_condsteps.py

生成は scripts/gen_uecfoodpix_flux3_test.py --cond_steps k (result_f3/UECFOODPIX_flux3/cond_steps/)。全ステップの分は既存の生成を使う。
指標は scripts/eval_uecfoodpix_flux3_boundary.py と同じ (tol 3 px) に加えて、tol 8 px の recall (おおまかな配置が合っているかの目安)。
出力: result_f3/UECFOODPIX_flux3/cond_steps/summary.json、panel_<stem>.jpg (512² のまま横に並べる。上段 Base 20/18/16/14/12/10/5/2/0、下段 蒸留版 16/8/4)
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # リポジトリ直下 (src/ dataset/ utils/ infer.py) を import できるように
import json
import statistics

from PIL import Image, ImageDraw

from eval_uecfoodpix_flux3_boundary import IMG_ROOT, MASK_LAB, STEMS, load_ids, score
from utils.boundary_metric import load_rgb

R = "result_f3/UECFOODPIX_flux3"
OUT = f"{R}/cond_steps"
ROWS = [
    [("Base cond 20/20", f"{R}/plate_test10_base20/gen")]
    + [(f"Base cond {k}/20", f"{OUT}/base20_cs{k}/gen") for k in (18, 16, 14, 12, 10, 5, 2, 0)],
    [("distilled cond 16/16", f"{R}/plate_test10_distill16/gen")]
    + [(f"distilled cond {k}/16", f"{OUT}/distill16_cs{k}/gen") for k in (8, 4)],
]
KEYS = ["recall", "recall_food", "recall_plate_table", "recall_tol8", "precision"]

res = {name: [] for row in ROWS for name, _ in row}
for s in STEMS:
    ids = load_ids(f"{MASK_LAB}/{s}.png")
    ncol = max(len(r) for r in ROWS) + 1
    panel = Image.new("RGB", (520 * ncol, (22 + 512 + 6) * len(ROWS)), "white")
    dr = ImageDraw.Draw(panel)
    for ri, row in enumerate(ROWS):
        y = ri * (22 + 512 + 6)
        if ri == 0:
            panel.paste(Image.fromarray(load_rgb(f"{IMG_ROOT}/{s}.jpg")), (0, y + 22))
            dr.text((4, y + 5), f"GT {s} (512²)", fill="black")
        for ci, (name, d) in enumerate(row, start=1):
            img = load_rgb(f"{d}/{s}.jpg")
            r = score(img, ids)
            r["recall_tol8"] = score(img, ids, tol=8)["recall"]
            res[name].append(r)
            panel.paste(Image.fromarray(img), (ci * 520, y + 22))
            dr.text((ci * 520 + 4, y + 5), f"{name}  recall {r['recall']:.3f}", fill="black")
    panel.save(f"{OUT}/panel_{s}.jpg", quality=92)

summary = {}
for name, rs in res.items():
    summary[name] = {k: round(statistics.mean(r[k] for r in rs), 3) for k in KEYS}
    summary[name]["recall_per_image"] = dict(zip(STEMS, (round(r["recall"], 3) for r in rs)))
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)

print(f"{'':24s}" + "".join(f"{k:>20s}" for k in KEYS))
for name, v in summary.items():
    print(f"{name:24s}" + "".join(f"{v[k]:>20.3f}" for k in KEYS))
print()
print(f"{'per image recall':24s}", "  ".join(f"{s:>5s}" for s in STEMS))
for name, v in summary.items():
    print(f"{name:24s}", "  ".join(f"{x:5.2f}" for x in v["recall_per_image"].values()))
