"""KID / FID (val の実写 55 枚、512² に変形したものとの分布の距離)。scripts/flux2_eval_val55.py と同じ設定 (clean-fid)。

    cd ~/Seg2Any/"[10-21]exp_layout_cfg" && CUDA_VISIBLE_DEVICES=0 ../.venv-flux3/bin/python kid.py --root results_val55 --conditions a,b,...
55 枚では FID はばらつきが大きいので参考値。少数サンプル向きの KID を主に見る。出力: <root>/kid.csv
"""
import argparse
import csv
import os

from cleanfid import fid

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--conditions", required=True)
p.add_argument("--real_dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result_f2/val55/real_512"))
args = p.parse_args()

rows = []
for c in args.conditions.split(","):
    d = os.path.join(args.root, c)
    r = dict(condition=c, kid_x1000=round(float(fid.compute_kid(d, args.real_dir, mode="clean", verbose=False)) * 1000, 2),
             fid=round(float(fid.compute_fid(d, args.real_dir, mode="clean", verbose=False)), 2))
    rows.append(r)
    print(r, flush=True)
with open(os.path.join(args.root, "kid.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(), w.writerows(rows)
