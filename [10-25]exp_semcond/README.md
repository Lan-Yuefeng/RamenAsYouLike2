# 意味付き cond トークン（exp_semcond）

2026-10-25 開始。`[10-25]exp_ablation` で、Seg2Any の輪郭 cond はクラスを区別せず、「何の領域か」は regional caption の説明文に頼り切っていることがわかった（caption を具材名だけにするとメンマの IoU が 0.85 → 0.12）。
そこで **領域ごとの caption 埋め込みを cond トークンに足し**、cond の側でも意味を伝える。

## 方法
```
cond_i ← x_embedder(cond_i) + W · context_embedder( Σ_r a_ir · e_r )
```
- `e_r`：領域 r の regional caption の Qwen3-VL 埋め込み（20480 次元）を、caption 部分のトークンだけで平均したもの。transformer に渡すテキスト埋め込みをそのまま使うので、追加のエンコードは無い（右パディングの因果 LM なので、caption 部分はパディングの影響を受けない）
- `a_ir`：cond トークン i のパッチ（16×16 px）のうち、領域 r が占める割合。cond トークンは輪郭上にしか無い（間引き）ので、多くは両側の領域を混ぜて持つ（例：404 トークン中 264）
- `context_embedder`：テキスト用の入力射影を凍結のまま共有（20480 → 3072）
- `W`：新しく足す `Linear(3072, 3072)`（9.4M パラメータ、fp32）。**ゼロ初期化**なので、学習開始時の出力は元の Seg2Any と完全に一致する
- 足す場所：cond の最初の mode block（`mode_blocks.image_cond.0`）の入力（forward pre-hook）

## ファイル
| ファイル | 役割 |
|---|---|
| `semcond.py` | 本体（`enable_semcond`、caption の平均埋め込み、cond トークンごとの意味ベクトル、推論用のヘルパー、W の保存と読み込み） |
| `test_semcond.py` | 実装テスト 6 項目（caption の範囲 / 面積比 / ゼロ初期化で元と一致 / 勾配が流れる / 保存と読み込み / 推論で cond の数が一致）。**6/6 PASS** |
| `train_flux3_semcond.py` | `../train_flux3.py` のコピー。`semcond` と書いた 5 か所だけ変更（import、W の追加、保存と読み込み、意味ベクトルの計算） |
| `seg2any_ramen_flux3_semcond.yaml` | ベースライン `config/seg2any_ramen_flux3.yaml` との違いは output_dir・`semcond: True`・validation なし だけ |
| `train_semcond.sh` / `run_train_bg.sh` | 学習の起動（後者は落ちたら latest から再開） |
| `gen_semcond.py` | 生成（ベースラインと同じ条件：50 ステップ、CFG 3.5、hard マスク）。`--no_sem` で W を外す |
| `run_eval_val10.sh` | val の先頭 10 枚 × caption 3 通り（全文 / 具材名だけ / 空）で生成し、評価 |

## 学習
- gp41（A40 × 8）、2026-10-25 11:13 開始、bs 2 × 8 GPU、3000 ステップ、約 4.5 秒/ステップ（約 3.8 時間）
- 出力：`/export/space0/lan-y/seg2any_ckpt/ramen_flux3_semcond/checkpoint-{250,...,3000}`、ログ `train_20261025_1113.log`
- ベースライン（`ramen_flux3`）は gp42 で同じ config・seed で学習したもの。違いは W の有無だけ（GPU の種類は違う）
- スモーク学習（2 GPU、20 ステップ）で、W が 0 から育つこと、checkpoint に `cond_sem_proj.safetensors` が保存されること、再開できることを確認済み

## 結果（2026-10-26、checkpoint-3000、val の先頭 10 枚）
| caption | 条件 | mIoU | bIoU(d3) | recall |
|---|---|---|---|---|
| 全文 | ベースライン | 0.941 | 0.442 | 0.631 |
| 全文 | semcond | 0.934 | 0.393 | 0.726 |
| 全文 | semcond（W を外す） | 0.837 | 0.288 | 0.616 |
| 具材名だけ | ベースライン | 0.820 | 0.382 | 0.713 |
| 具材名だけ | semcond | 0.691 | 0.282 | 0.902 |
| 空 | ベースライン | 0.669 | 0.329 | 0.738 |
| 空 | semcond | 0.306 | 0.074 | 0.249 |

- 全文 caption では mIoU は同等、bIoU はやや低い。仮説（意味が cond 側で補われ、短い caption で崩れにくくなる）は**支持されなかった**
- 具材名だけではメンマが 0.12 → 0.57 と改善した一方、切った卵が丸ごとの卵になり（cut_egg 0.96 → 0.00）、mIoU 全体は下がった（cut_egg / egg はそれぞれ 1 枚だけなので値はぶれやすい）
- 空 caption では 0 ベクトルが W に入り、丼の形ごと崩れる
- W を外すと mIoU が 0.12 下がるので、LoRA は W に頼るように学習されている。学習では regional caption を常に全文で与えているため、具材名だけや空の caption の埋め込みは W にとって学習時に見ていない入力になっている
