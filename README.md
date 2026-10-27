# RamenAsYouLike2

セグメンテーションマスクから好きなラーメン画像を生成する卒業研究のコードです（作業中）。

[Seg2Any](https://github.com/0xLDF/Seg2Any) (NeurIPS 2025) をベースに、FLUX.2 / FLUX.3 への移植と、ラーメン・料理データでの LoRA 学習を行っています。

## 内容

- `src/`, `train_flux*.py`, `infer_flux*.py` — Seg2Any の FLUX.2 / FLUX.3 版
- `dataset/` — UEC-Ramen555 / UECFoodPix / SACap-1M 用のデータローダと caption
- `config/`, `train_*.sh` — 学習設定
- `[MM-DD]exp_*/` — 個別の実験（レイアウト CFG、切除実験、VLM caption、高速化、自己蒸留など）
- `FLUX2_PORT.md`, `FLUX3_PORT.md` — 移植の記録

データセット・学習済み重み・生成画像はリポジトリに含めていません。

## 謝辞

元のコードは [0xLDF/Seg2Any](https://github.com/0xLDF/Seg2Any) によるものです。
