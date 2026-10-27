# ③ txt / cond 一方向化 + K/V キャッシュの自己蒸留 (2026-10-27, gp36)

案2 の ③ (exp_speed/cache_flux3.py の CachedForward) は学習なしだと具が過剰・配置崩れになる。これを自己蒸留で取り戻す。

## 方法
- 教師: 蒸留版 + ramen LoRA (/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000)、全結合 attention、LoRA 未融合
- 教師の軌跡: train 500 枚 × seed 0-3 × 4 ステップの transformer 入力と出力を、デモと同じ推論設定 (512²、hard マスク) で保存
  (`precompute_teacher.py`、`run_precompute.sh`) → /export/ssd2/lan-y/seg2any_data/ramen_selfdistill/teacher_steps4 (8000 点、約 15GB)
- 生徒: 教師と同じ LoRA から開始し、`CachedForward(use_cache=False, cache_txt=True, ckpt=True)` で学習
  (joint blocks で txt / cond → img を遮断、txt / cond の modulation は t=0・飛び先 0 に固定)。推論のキャッシュあり計算と完全一致する
- 損失: 教師の状態 x_t での出力 MSE (off-policy)。default / cond 両 LoRA (67.5M)、AdamW lr 5e-5、2000 step、6 GPU (4090x4 + A100x2) × bs1、1.8 s/step、約 67 分
- 出力: /export/space0/lan-y/seg2any_ckpt/ramen_flux3_selfdistill/checkpoint-{500,1000,1500,2000}
- 一括実行: `run_all.sh` (事前計算待ち → `train_selfdistill.py` → `gen_val10.py` → eval.py / boundary_iou.py)

loss: 0.20 (開始直後) → 0.016 (2000 step)。ステップ別では最後のステップ (k=3) が一番大きい

## 結果 (val10、4 ステップ、512²、推論は ① 融合 + ③ キャッシュ)

| 条件 | mIoU | bIoU d3 | bIoU d8 | 境界 recall | CLIP-T | CLIP-I | 1 枚 (4090) |
|---|---|---|---|---|---|---|---|
| ① のみ (ref_fuse, キャッシュなし) | 0.908 | 0.334 | 0.596 | 0.646 | 28.22 | 88.97 | 0.98 s |
| ①+③ 学習前 (sd_0) | 0.433 | 0.067 | 0.112 | 0.463 | 28.47 | 83.94 | 0.69 s |
| ①+③ 500 step | 0.895 | 0.310 | 0.566 | 0.617 | 27.84 | 89.17 | |
| ①+③ 1000 step | 0.899 | 0.301 | 0.559 | 0.631 | 28.20 | 89.20 | |
| ①+③ 1500 step | 0.903 | 0.309 | 0.570 | 0.617 | 28.24 | 89.06 | |
| ①+③ 2000 step | **0.907** | 0.322 | 0.585 | 0.656 | 27.93 | 89.45 | **0.65 s** |

(今のまま = 融合もキャッシュもなし は 4090 で 1.22〜1.25 s。1 枚の時間は exp_speed の計測と `result/t4090_sd2000/timing.json`)

- 2000 step で mIoU は ① とほぼ同じ (-0.001)、bIoU d3 は -0.012、CLIP-I はむしろ +0.5。学習前の崩れ (mIoU 0.43) はほぼ消えた
- 見た目 (`panel_selfdistill_val10.jpg`、列 = mask / ① / ①+③ 学習前 / ①+③ 2000 step) も ① とほぼ同じ絵になる。具の過剰・器の二重化は消えた
- 速度は 4090 で 1.22 s → 0.65 s (1.9 倍)。① だけ (0.98 s) より 0.33 s 速い
- 注意: val10 だけ。境界の細部 (bIoU d3) は少し負けている。教師は ① の未融合版なので、上限は教師の品質

## val55 (2026-10-27, `run_val55.sh` → ~/Seg2Any/result_f3/val55_selfdistill、ここからは `result_val55` で辿れる)

生成は `gen_val.py` (mode: base / fuse / fuse_cache)。4090 x4 で 4 条件を並列に生成 (同じノードで並列なので時間はやや遅め。下表の時間は単独計測の値)。

| 条件 | mIoU | bIoU d3 | bIoU d8 | KID×1000 | FID | CLIP-T | CLIP-I | 1 枚 (4090 単独) |
|---|---|---|---|---|---|---|---|---|
| base (今のまま) | 0.893 | 0.286 | 0.551 | 16.9 | 87.1 | 27.65 | 89.40 | 1.22 s |
| ref_fuse (① のみ) | 0.894 | 0.286 | 0.551 | 16.8 | 86.6 | 27.71 | 89.36 | 0.98 s |
| sd_0 (①+③ 学習前) | 0.437 | 0.064 | 0.109 | 34.0 | 108.8 | 27.81 | 85.14 | 0.69 s |
| sd_2000 (①+③ 自己蒸留) | 0.890 | 0.270 | 0.537 | 15.7 | 85.3 | 27.83 | 89.31 | 0.65 s |

- base との対応のある比較 (paired.csv, Wilcoxon): sd_2000 の mIoU -0.003 (p=0.20)、recall -0.010 (p=0.17)、CLIP-I -0.10 (p=0.68)。有意差なし
- 残る劣化は境界の細部 (bIoU d3 -0.016)。KID / FID はむしろ良い (55 枚なので参考値)
- ① は全指標で base と区別できない (mIoU p=0.44)
