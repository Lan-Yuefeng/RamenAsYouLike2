"""train_flux2.py が tensorboard に書いた validation 画像を、指定ステップぶん 1 枚の一覧にする。

    .venv-flux2/bin/python flux2_val_panel.py --step 1000 [--logdir ckpt/ramen_flux2/logs]

各行: 元写真 | マスク + caption | 生成画像。再開でイベントファイルが複数あっても全部探す。
"""
import argparse
import glob
import io
import os

from PIL import Image, ImageDraw
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, required=True)
parser.add_argument("--logdir", default="ckpt/ramen_flux2/logs")
parser.add_argument("--out", default=None)
parser.add_argument("--width", type=int, default=900)
args = parser.parse_args()

rows = {}
for f in sorted(glob.glob(os.path.join(args.logdir, "**", "events.out.tfevents.*"), recursive=True)):
    ea = EventAccumulator(f, size_guidance={"images": 0})
    ea.Reload()
    for tag in ea.Tags().get("images", []):
        for e in ea.Images(tag):
            if e.step == args.step:
                rows[tag] = Image.open(io.BytesIO(e.encoded_image_string)).convert("RGB")
if not rows:
    raise SystemExit(f"step {args.step} の validation 画像がまだ無い")

w = args.width
ims = [r.resize((w, int(r.height * w / r.width))) for r in rows.values()]
panel = Image.new("RGB", (w, 22 + sum(i.height for i in ims)), "white")
ImageDraw.Draw(panel).text(
    (6, 5), f"step {args.step}:  real photo | mask + captions | generated (klein-base, CFG 4, 50 steps)", fill="black")
y = 22
for im in ims:
    panel.paste(im, (0, y))
    y += im.height
out = args.out or f"result_f2/flux2_val/step{args.step}.jpg"
os.makedirs(os.path.dirname(out), exist_ok=True)
panel.save(out, quality=90)
print(f"{len(ims)} rows -> {out}")
