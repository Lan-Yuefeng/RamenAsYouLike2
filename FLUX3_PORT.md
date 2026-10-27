# Seg2Any の FLUX 3 Image Klein への移植メモ

Seg2Any の基盤モデルを FLUX.2 [klein] 4B（[FLUX2_PORT.md](FLUX2_PORT.md)）から **FLUX 3 Image Klein（5.6B）** に置き換える作業の記録です。
FLUX.1 版・FLUX.2 版のファイルには手を入れず、`*_flux3.py` として新しく作っています。

最終更新：2026-10-16（Step 4 まで完了）

---

## 1. 現在の状況

| ステップ | 内容 | 状態 |
|---|---|---|
| 1 | 新しい venv を作り、素の klein で画像を生成する | 完了 |
| 2 | transformer に Seg2Any の仕組み（cond トークン、領域マスク、2 種類の LoRA）を移植する | **完了**（テスト 13/13） |
| 3 | 推論パイプラインを移植する（Qwen3-VL、80 トークン単位のパディング、image_rope_scale） | **完了**（テスト 7/7） |
| 4 | 学習スクリプトを移植する | **完了**（テスト 6/6、20 ステップのスモーク学習と再開を確認） |
| 5 | UEC-Ramen555 で LoRA を学習し、FLUX.1 / FLUX.2 版と比べる | **完了**（§7。画質は最良、境界 recall は FLUX.1 / FLUX.2 Base に届かず） |

---

## 2. 環境

- venv：`~/Seg2Any/.venv-flux3`（実体は `/export/ssd3/lan-y/venvs/Seg2Any/.venv-flux3`）
  - **Python 3.11**（diffusers main が 3.11 以上を要求するため。FLUX.2 版は 3.10）
  - torch 2.8.0+cu128、diffusers 0.41.0.dev0（GitHub main から）、transformers 5.18.0。それ以外は `.venv-flux2` と同じ版
- モデル（gated。2026-10-16 に利用規約に同意済み）：ssd3 にキャッシュしてある（diffusers 形式の部分だけ。単一ファイル版と量子化版は落としていない）

| モデル | 用途 | 既定 |
|---|---|---|
| `black-forest-labs/flux-3-image-klein-base` | 蒸留なし。LoRA の学習と丁寧な推論に使う | 50 ステップ、CFG あり |
| `black-forest-labs/flux-3-image-klein` | 蒸留版（flow map）。速い推論に使う | 16 ステップ、CFG なし |

- **実行時の環境変数は `HF_HOME=/export/ssd3/lan-y/huggingface` だけにする**。diffusers dev 版は分割された重みを読むときに Hub に問い合わせるので、`HF_HUB_OFFLINE=1` を付けると transformer の読み込みに失敗する（エラーにならずに `None` になり、あとで `'NoneType' object has no attribute 'dtype'` になる）。
- 素のモデルの読み込みは `ModularPipeline.from_pretrained(repo)` → `load_components(dtype=torch.bfloat16)`（引数名は `torch_dtype` ではない）

### 素のモデルの速度（gp42、1024²、`flux3_smoke.py`）
| | ステップ数 | 1 枚 | ピークメモリ |
|---|---|---|---|
| 蒸留版 | 16 | 4.1 秒 | 21.2 GiB |
| Base 版 | 50 | 22.9 秒 | 21.2 GiB |

---

## 3. FLUX 3 の構造（FLUX.2 との違い）

| 項目 | FLUX.2 klein 4B | FLUX 3 Image Klein |
|---|---|---|
| ブロック | double 5 層 + single 20 層 | **mode blocks 5 層**（txt / image / image_cond がそれぞれ別の重みで、自分のモード内だけで attention）+ **joint single blocks 28 層**（全トークン結合、重み共有、attention と MLP を並列に計算） |
| 参照画像 | 入力の後ろにつなぐだけ | **専用の `image_cond` ストリーム**（入力射影、mode blocks、modulation が専用） |
| modulation | 全体で共有の 3 つの Linear | モードごと（txt / image / image_cond）× 前半用・後半用。txt と image_cond は timestep 0 |
| テキストエンコーダ | Qwen3（3 層を連結、7680 次元） | **Qwen3-VL-4B**（4, 8, …, 32 層目の 8 層を連結、20480 次元） |
| テキストの長さ | 固定長 | **80 の倍数に右パディング**。transformer はパディングも見る。パディング部分の Qwen 側の position id は 1 に固定 |
| 画像の位置 ID | (0, h, w, 0) | **(0, 0.5h, 0.5w, 0)**（`image_rope_scale=0.5`）。参照画像 k は l = -(k+1) |
| 蒸留版 | ステップ数を減らしただけ | flow map：飛び先の timestep の埋め込み（`target_timestep_embedder`）を全モードに足す |
| スケジューラ | — | time shift 6.93 |

---

## 4. Step 2：transformer（`src/models/transformer_flux3.py`）

`Flux3RegionalTransformer2DModel`（`Flux3Transformer2DModel` の子クラス。重みのキーは純正と同一なので `from_pretrained` でそのまま読める）。

- **cond トークンは `image_cond` ストリームに載せる**。FLUX 3 はもともと「timestep 0 のきれいな画像トークン」を参照画像として扱う経路を持っているので、Seg2Any の輪郭 cond（timestep 0）とそのまま対応する。
- **マスク**：pipeline が作る (B, H, S, S) の全体マスク `[txt | img | cond]` を受け取る。
  - mode blocks では、各モードの対角ブロック（txt×txt、img×img、cond×cond）を切り出して使う。こうしないと、前半 5 層で regional caption 同士が混ざってしまう（テスト 8）。
  - joint blocks では全体マスクをそのまま使う。
  - 全体マスクの各対角ブロックには、どこも見られない行がない（各 caption は自分自身を見る、cond の pad は pad 同士を見る）。そのため切り出しても NaN にはならない。
- **hard_attn_block_range**：ブロック番号は mode blocks が 0〜4、joint blocks が 5〜32。既定値は FLUX.1 / FLUX.2 と同じ比率の **[5, 18]**。
- **LoRA の切り替え**：txt と img は `default`、cond は `cond`。mode blocks はモードごとに重みが分かれているので、ブロックに adapter 名を渡す。joint blocks では、[txt, img] を `default` で、cond を `cond` で射影し、attention だけ 1 本の系列で取る。
- LoRA の対象（テストで使ったもの。学習用 config の雛形）
  - default：txt と image の mode blocks、joint blocks の全 Linear、txt と image の modulation（36.3M パラメータ、r=16）
  - cond：image_cond の入力射影、modulation、mode blocks、joint blocks の全 Linear（31.3M）
- QK norm と RoPE は純正と同じく fp32 で計算する。

### テスト（`flux3_step2_test.py`、13/13 PASS）
```bash
HF_HOME=/export/ssd3/lan-y/huggingface .venv-flux3/bin/python flux3_step2_test.py
```
| # | 内容 | 結果 |
|---|---|---|
| — | state_dict のキーと重みが純正と同一 | 357 keys |
| 1 | cond なしなら純正と一致 | 2.5e-5 |
| 1b | cond あり・全許可マスクなら、純正に参照画像として渡したときと一致 | 3.7e-5 |
| 1c | 蒸留版（flow map）も純正と一致 | 7.2e-5 |
| 2 / 3 | cond を隔離すれば cond なしと一致する / 隔離しなければ効く | 0 / 1.0 |
| 4 | hard マスクは指定範囲のブロックにだけ効く | 0 / 0 |
| 5a / 5b | `cond` LoRA は cond なしの出力に漏れない / `default` LoRA は効く | 0 / 0.42 |
| 6 | 勾配チェックポイントありで backward が通り、両方の LoRA に勾配が流れる | B=2、512² で 1.2 秒、11.8 GiB |
| 7 | バッチごとに違う txt_ids を扱える | 0 / 0 |
| 8 | どこからも見られない caption は出力に影響しない（mode blocks にもマスクが効いている） | 0 |

---

## 5. Step 3：推論パイプライン（`src/pipelines/pipeline_flux3.py`、`infer_flux3.py`）

`Flux3RegionalPipeline`。diffusers には FLUX.3 の ModularPipeline しか無いので、公式の各ブロック（`modular_pipelines/flux3/`）の処理を 1 つの `DiffusionPipeline` にまとめ直し、`__call__` は FLUX.2 版と同じ形にした。
`Flux3LoraLoaderMixin` を継承しているので、`load_lora_weights` / `save_lora_weights` は FLUX.2 版と同じように使える。

- **読み込み**：repo に `model_index.json` が無い（ModularPipeline 用の `modular_model_index.json` だけ）ので、`Flux3RegionalPipeline.from_pretrained(repo, transformer=...)` は VAE・Qwen3-VL・processor を 1 つずつ読み、スケジューラは `FlowMatchEulerDiscreteScheduler(shift=6.93)` を作る。`transformer=None` も可（学習時のテキスト専用）。
- **テキスト**：caption ごとに公式と同じ手順でエンコードする（チャットテンプレート → 80 の倍数に右パディング → パディングの position id を 1）。パディングはその caption のトークンとして残し、`txt_seq_lens` にも含める（マスク上も同じ区間）。パディング後の長さが同じ caption はまとめて 1 バッチで通す。
  - ラーメンの caption は global も regional も 80 トークンに収まるので、7 領域なら txt は 8 × 80 = 640 トークン。
  - `text_pad_multiple=1` にするとパディングなし（FLUX.2 版と同じ扱い。比較実験用）。
- **位置 ID**：noisy は (0, 0.5h, 0.5w, 0)、cond は (0, 0.5h, 0.5w, -1)。txt は (0, 0, 0, l) で、l は caption ごとに 0 から振り直す。
- **スケジューラ**：sigma は `linspace(1, 1/N, N)` に shift 6.93。蒸留版は各ステップで飛び先（次の timestep。最後は 0）を `target_timestep` として渡す。
- **既定値**：蒸留版 16 ステップ（CFG なし）、Base 版 50 ステップ・CFG 3.5（公式の値）。CFG のネガティブ側には cond を入れない（FLUX.2 版での結論）。
- 初期ノイズは公式と同じく fp32 で引いてから bf16 にする（seed の互換のため）。

### テスト（`flux3_step3_test.py`、7/7 PASS）
| # | 内容 | 結果 |
|---|---|---|
| 1 | caption ごとの埋め込みが純正の `Flux3TextEncoderStep` と一致する | 相対誤差 1.4e-3（bf16） |
| 2 | 各 caption が 80 の倍数長で、L が区間ごとに 0 から振り直される | lens = [80 × 8] |
| 3 | cond の間引き数と位置 ID (0, 0.5h, 0.5w, -1) | 434 / 1024 トークン |
| 4a | regional caption を完全に隔離すれば、純正パイプラインと同じ画像になる（蒸留版 16 ステップ） | PSNR 40.4 dB |
| 4b | 同上（Base 版 50 ステップ、CFG 3.5） | PSNR 45.7 dB |
| 5a / 5b | ラーメンの例を生成（512²、LoRA なし、hard マスク） | 蒸留版 1.7 秒 / Base 版 9.8 秒、19.5 GiB |

比較画像：`result_f3/flux3_step3_panel.jpg`（上段：純正の蒸留版 | 自前の蒸留版 | 純正の Base 版 | 自前の Base 版、下段：マスク | LoRA なしの蒸留版 | LoRA なしの Base 版）。LoRA なしの出力：`result/flux3_ramen_nolora/`

### 推論
```bash
HF_HOME=/export/ssd3/lan-y/huggingface .venv-flux3/bin/python infer_flux3.py \
    --seg_mask_path examples_Ramen --output_path result/flux3_ramen [--lora_ckpt_path ...] [--no_cond]
# Base 版で丁寧に生成（50 ステップ、CFG 3.5）
... --pretrained_model_name_or_path black-forest-labs/flux-3-image-klein-base
```

---

## 6. Step 4：学習スクリプト（`train_flux3.py`、`config/seg2any_ramen_flux3.yaml`、`train_ramen_flux3.sh`）

`train_flux2.py` をコピーし、FLUX.3 で違う部分だけを書き換えた（データ・損失・最適化・保存の流れは同じ）。

| 項目 | FLUX.2 版 | FLUX.3 版 |
|---|---|---|
| ノイズスケジューラ | repo の scheduler | repo に無いので `FlowMatchEulerDiscreteScheduler(shift=6.93)` を作る。timestep は shift 済みの sigma から一様に取り、weighting は none（**diffusers 公式の `train_dreambooth_lora_flux3_klein.py` と同じ**） |
| VAE | 別に読む | テキスト用パイプラインが読んだものを fp32 にして使う |
| transformer の GPU への移し方 | `.to(device, dtype=bf16)` | **`.to(device)` だけ**。bf16 に cast すると、`from_pretrained` が fp32 に残した QK norm まで bf16 に落ちて推論とずれる（テスト 5） |
| 位置 ID | (0, h, w, 0) | noisy は (0, 0.5h, 0.5w, 0)、cond は (0, 0.5h, 0.5w, -1)（`prepare_image_ids`。推論と同じ関数） |
| テキスト | パディングを除いて連結 | caption ごとに 80 の倍数に右パディング（`model.text_pad_multiple`） |
| guidance | klein には無い | 無い。蒸留版（flow map）を指定したらエラーにする（学習は Base 版） |
| default LoRA | `transformer_blocks` と `single_transformer_blocks` の全 Linear + 変調 | `txt_mode_blocks`・`mode_blocks.image`・`single_transformer_blocks` の全 Linear + txt / image の変調 4 つ |
| cond LoRA | x_embedder・変調・img 側の attn と ff・single の融合 Linear | `x_embedders.image_cond`・image_cond の変調 2 つ・`mode_blocks.image_cond` の全 Linear・joint blocks の全 Linear |
| validation | CFG 4.0 | CFG 3.5（Base 版の既定） |

config は FLUX.2 版 v2（`seg2any_ramen_flux2_v2.yaml`）をもとにした。比べられるように、データ・caption・rank 16・学習率 1e-4・3000 ステップ・bs 2 × 8 GPU・勾配チェックポイントありは同じにしている。出力先は `/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3`。

```bash
bash train_ramen_flux3.sh                                   # 8 GPU
bash train_ramen_flux3.sh --resume_from_checkpoint=latest   # 途中から再開
# GPU を絞るとき (2 GPU 用の accelerate config もある)
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 ACCEL_CONFIG=accelerate_2gpu_config.yaml bash train_ramen_flux3.sh
```

### スモーク学習（2 GPU、20 ステップ、gp42）
- 出力：`/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3_smoke/checkpoint-{10,20,24}`、ログ：`/export/ssd3/lan-y/logs/flux3_smoke_{train,resume}.log`
- 約 1.5 秒/ステップ（bs 2、512²、勾配チェックポイントあり）。loss は 0.6〜0.9
- 保存した LoRA のパラメータ数は 866 テンソル（default + cond）
- `--resume_from_checkpoint=latest` で checkpoint-20 から 24 まで再開できた

### テスト（`flux3_step4_test.py`、6/6 PASS）
| # | 内容 | 結果 |
|---|---|---|
| 1 | 保存した default / cond LoRA が、推論側の `load_lora_weights` で全テンソル一致で読める | 866 テンソル、差 0 |
| 2 | LoRA が効く（LoRA あり vs なし / cond を外す） | 2.4 / 2.1 |
| 3 | cond トークンと位置 ID が学習と推論で一致する | 差 0 |
| 4 | テキスト入力が学習（B=2）と推論（B=1）で一致する | 一致 |
| 5 | 学習と同じ読み込み方で QK norm が fp32 のまま | 86 個すべて fp32 |
| 6 | Base 版で学習した LoRA が蒸留版（flow map）にそのまま載り、生成できる | OK |

---

## 7. Step 5：本番学習と val55 評価

### 学習（gp42、8 GPU、2026-10-16 16:43〜18:22）
- 起動：tmux `flux3_ramen` で `/export/ssd3/lan-y/logs/run_flux3_ramen.sh`（異常終了時は latest から自動再開、最大 3 回。今回は再開なしで rc=0）
- ログ：`/export/ssd3/lan-y/logs/flux3_ramen_train_20261016_1643.log`
- 出力：`/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-{250,...,3000}`
- 3000 ステップ / 1 時間 36 分（1.92 秒/ステップ）、各 GPU 23.7 GB
- loss（500 ステップごとの平均）：0.669 → 0.664 → 0.661 → 0.646 → 0.640 → 0.632

### 生成（`result_f3/run_val55_flux3_gen.sh`、checkpoint-3000、512²、GPU 1 枚ずつ並列）
| 条件 | 55 枚の時間 |
|---|---|
| flux3_base_cfg35_50step | 835 秒 |
| flux3_base_cfg35_20step | 366 秒 |
| flux3_distilled（16 ステップ、CFG なし） | 231 秒 |

### 評価（`flux2_eval_val55.py --result_root result_f3/val55_flux3`）
FLUX.1 / FLUX.2 は `result_f2/val55` の生成物をリンクして同じ条件で再計算した。

| 条件 | 境界 recall | 境界 precision | CLIP-T | CLIP-I | KID×1000 | FID |
|---|---|---|---|---|---|---|
| 実写 | 0.652 | 0.261 | 26.34 | 100 | – | – |
| FLUX.1 | **0.808** | 0.307 | 27.96 | 85.32 | 18.85 | 84.68 |
| FLUX.2 Base CFG4 | 0.772 | 0.311 | 28.06 | 89.12 | 17.92 | 85.56 |
| FLUX.2 蒸留版 | 0.592 | 0.271 | **28.37** | 85.46 | 44.37 | 109.6 |
| FLUX.3 Base CFG3.5 50 ステップ | 0.694 | 0.336 | 27.14 | **90.22** | **11.66** | **75.79** |
| FLUX.3 Base CFG3.5 20 ステップ | 0.672 | **0.379** | 27.21 | 90.08 | 14.96 | 78.39 |
| FLUX.3 蒸留版 | 0.657 | 0.255 | 28.00 | 90.19 | 16.07 | 81.44 |

FLUX.1 より recall が高い画像の数（55 枚中）：FLUX.2 Base 23、FLUX.3 Base 50 ステップ 13、20 ステップ 9、蒸留版 8。

### 所見
- **画質・実写らしさは FLUX.3 が最良**：KID / FID / CLIP-I がすべて 1 位。蒸留版でも FLUX.1 / FLUX.2 Base より KID が良く、FLUX.2 蒸留版（KID 44）のような崩れはない
- **レイアウトの忠実度（境界 recall）は FLUX.1 / FLUX.2 Base より低い**（0.694 vs 0.808 / 0.772）。丼・レンゲ・のりなど大きい領域の位置は合うが、具材の細かい形が mask からずれやすい
- 丼の縁に赤や青の模様を描くことが多い。余計な輪郭になるが、precision はむしろ高い
- 蒸留版は Base 50 ステップより 3.6 倍速く、recall の低下は 0.04 に収まる（FLUX.2 では蒸留版で 0.18 落ちた）
- パネル：`result_f3/val55_flux3/panel.jpg`、画像ごとの値：`metrics_per_image.csv`、`summary.json`

### 次の候補
- 学習を延ばす（loss はまだ下がっている）、または途中のチェックポイント（1500 / 2250 など）で recall を比べる
- hard_attn_block_range（既定 [5,18]）や cond LoRA の対象を変える


---

## 8. 1024² で推論（学習は 512² のまま）

UEC-Ramen555 は短辺の中央値が 450 px（1024 以上は 555 枚中 3 枚）なので、1024² の学習はせず、512² で学習した checkpoint-3000 を 1024² で推論した。
val55 の先頭 10 枚だけで比べる（`result_f3/run_val10_flux3_1024_gen.sh` → `result_f3/val10_flux3_1024`、評価は `flux3_eval_val10_1024.py`。生成画像は 512 に縮めて 512 の mask と比べる。10 枚なので KID / FID は出さない）。
mask は 512 の mask を最近傍で 2 倍にしたもの（`examples_Ramen_val10_1024`）。csf2 は `--cond_scale_factor 2`（輪郭画像を 512 で作り、位置 id を 2 倍して 1024 の格子に合わせる）。

| 条件 | 境界 recall | precision | CLIP-T | CLIP-I | 10 枚の時間 |
|---|---|---|---|---|---|
| Base 50 ステップ 512² | 0.649 | 0.312 | 27.04 | 89.35 | – |
| **Base 50 ステップ 1024²** | **0.693** | 0.240 | 27.21 | 88.67 | 537 秒 |
| Base 50 ステップ 1024² csf2 | 0.405 | 0.179 | 27.90 | 89.50 | 497 秒 |
| 蒸留版 512² | 0.610 | 0.243 | 28.02 | 88.98 | – |
| **蒸留版 1024²** | **0.658** | 0.194 | 28.51 | 87.72 | 144 秒 |
| 蒸留版 1024² csf2 | 0.452 | 0.139 | 28.80 | 87.80 | 130 秒 |

（時間は 4 条件を別々の GPU で同時に走らせたときの値）

- 1024² にすると recall は Base 版で +0.044、蒸留版で +0.048 上がる（10 枚中 Base 版 7 枚、蒸留版 6 枚で上昇）
- 見た目は 1024² のほうがはっきり細かい（麺・ネギ・チャーシューの筋）。学習データが低解像度でも、細部はベースモデルが描いている
- precision が下がるのは、細かい模様のエッジが増えるため（512 に縮めても残る）
- csf2 は recall が大きく下がり、画像もぼやける。輪郭の cond は 1024 の格子で作る（csf1）ほうがよい
- パネル：`result_f3/val10_flux3_1024/panel_<stem>.jpg`（1 マス 1024 px）、拡大比較：`crop_ramen_0007_base50_512up_1024_csf2.jpg`


---

## 9. SACap-1M で LoRA 学習（2026-10-16〜）

FLUX.2 版の SACap 試走（`config/seg2any_sacap_flux2.yaml`、sa_000000-009 の 10,451 枚、1024² / cond 512、hard、rank 64、全体バッチ 16、2000 ステップ）と同じ条件で FLUX.3 版を学習する。

- 新規：`dataset/sacap_1m_flux3_dataset.py`（`SACap1MFlux3Dataset`。FLUX.2 版との違いはバケット用テキスト長だけ：Qwen3-VL のテンプレート込みで caption ごとに 80 の倍数に切り上げ。キャッシュ `SACap-1M/cache/train/qwen3vl_txt_seq_len_g512_r50_p80.parquet`。平均 566 / 最大 1760 トークン）
- `config/seg2any_sacap_flux3.yaml`（FLUX.2 版との差はモデル・LoRA 対象層・hard_attn_block_range [5,18]・text_pad_multiple 80・validation CFG 3.5）、`train_sacap_flux3.sh`
- gp42 での試走：grad ckpt なし・bs2 は 95GB でも OOM。grad ckpt ありで **約 5 秒/ステップ**（FLUX.2 版は gp40 A6000 で 10.3 秒）
- 本番：gp42 8 GPU、tmux `flux3_sacap`（`/export/ssd3/lan-y/logs/run_flux3_sacap.sh`、異常終了時は latest から再開）、ログ `/export/ssd3/lan-y/logs/flux3_sacap_train_20261016_2223.log`、出力 `/export/ssd3/lan-y/seg2any_ckpt/sacap_flux3/checkpoint-{500,...,2000}`
- 学習結果：2000 ステップ / 3 時間 8 分（5.0 秒/ステップ）、2026-10-17 01:32 に正常終了（rc=0、再開なし）。loss は最後 0.57 前後
- 生成（`result_f3/run_sacap_flux3_gen.sh`）：`examples_SACap_eval` の 12 枚（1024x672）、学習と同じ hard / cond_scale_factor 2。Base+LoRA 50 ステップ CFG 3.5、蒸留版+LoRA 16 ステップ、Base LoRA なし → `result_f3/sacap_flux3/examples_SACap_eval/{base_lora,distill_lora,base_nolora}`
- パネル：`examples_SACap_eval/make_panel_flux3.py` → `result_f3/sacap_flux3/examples_SACap_eval/panel_<stem>.png`（縮小なし PNG。FLUX.2 Base+LoRA も並べる。GT は原寸で最下段）
- 見た目：LoRA ありは mask の配置にきちんと従う（LoRA なしは配置を無視）。蒸留版でも崩れない。FLUX.2 版と比べると、イカ（sa_8686177）のように形が caption どおりの物体になる
- `examples/`（landmarks / spiderman / animals / 20_circular_badges）でも同じ 3 条件を生成（`result_f3/run_sacap_flux3_gen_examples.sh` → `result_f3/sacap_flux3/examples/`、パネルは `make_panel_flux3.py examples`）。LoRA ありは 4 例とも mask の形と配置に従い、領域ごとの caption（ピサの斜塔・万里の長城・エッフェル塔・タージマハル、レゴのライオン・氷のペンギン・炎の鷲、A〜T のバッジ）も正しく出る。LoRA なしは配置を無視し、バッジの並びも崩れる。蒸留版でも崩れない


---

## 10. UECFoodPix（皿入り）で LoRA 学習（2026-10-24、gp41）

FLUX.2 皿版（`config/seg2any_uecfoodpix_plate_flux2.yaml`、ckpt `ssd3/seg2any_ckpt/uecfoodpix_plate_flux2`）と比べられるように、
**データの中身・caption・rank・学習率・バッチ・ステップ数を FLUX.2 皿版に揃え**、モデル部分（LoRA 対象層・テキストのパディング・validation）は §6 のラーメン FLUX.3 版と同じにした。

### ファイル

| 種類 | パス |
|---|---|
| Dataset | `dataset/uecfoodpix_plate_flux3_dataset.py`（`UECFoodPixPlateFlux3Dataset`。FLUX.2 皿版を継承し、バケット用テキスト長だけ Qwen3-VL で数え直す） |
| config | `config/seg2any_uecfoodpix_plate_flux3.yaml` |
| 起動 | `train_uecfoodpix_plate_flux3.sh`（`VENV` / `HF_HOME` は既定で ssd2） |
| 学習本体 | `train_flux3.py`（変更なし） |
| caption 語彙 | `dataset/uecfoodpix_plate_captions.py`（FLUX.2 皿版と共通、変更なし） |
| 出力 | `/export/space0/lan-y/seg2any_ckpt/uecfoodpix_plate_flux3/`（`checkpoint-*`、`logs/`（tensorboard）、`train.log`） |

```bash
bash train_uecfoodpix_plate_flux3.sh                                  # 8 GPU
bash train_uecfoodpix_plate_flux3.sh --resume_from_checkpoint=latest  # 再開
```

### データ

| 項目 | 内容 |
|---|---|
| 画像 | UECFOODPIXCOMPLETE（研究室共有 `/export/data/dataset/UECFOOD/UECFOODPIXCOMPLETE` を `/export/ssd2/lan-y/UECFOODPIXCOMPLETE/data/` にコピー）。train 9000 / test 1000（公式分割 `train9000.txt` / `test1000.txt`） |
| マスク | 研究室共有の **`UECFOODPIXCOMPLETEwithPLATE_104`**（`{train,test}/labels/<stem>.png`）を `/export/ssd2/lan-y/seg2any_data/uecfoodpix/mask_plate_lab/{train,test}/` にコピーし、**R チャンネルだけの 1ch PNG に変換**（原本は 3ch 同値。FLUX.2 版の `_cond_token_num` が 1ch 前提のため） |
| ラベル | 0 = テーブル等、1..102 = 料理、103 = 皿・器・お盆（104 値。FLUX.2 皿版の自作マスクと同じ体系） |
| FLUX.2 皿版との違い | FLUX.2 皿版の皿マスクは `UECFOODPIX_Plate` から自作（`make_uecfoodpix_plate_masks.py`）。今回は共有の完成版を使ったので、**皿の輪郭は FLUX.2 皿版と完全には同じでない可能性がある** |
| 皿ラベルのある画像 | train 89.6%、test 99.8% |
| 1 枚あたりの料理数 | train 平均 1.59（最大 9）、test 平均 1.73（最大 7） |
| caption | 領域ごと：料理名 / `table surface and surroundings`（0）/ `plates, bowls and tray`（103）。global：`A photo of a meal: <料理名の列挙>, served on tableware on a table.` |
| バケット | Qwen3-VL トークン長（テンプレート込みで caption ごとに 80 の倍数に切り上げ）と cond トークン数を 50 刻みで分ける。59 バケット。キャッシュ `/export/ssd2/lan-y/seg2any_data/uecfoodpix/cache_flux3/plate_bucket_train9000_512H_512W_g512_r50.parquet` |
| トークン長（train） | テキスト 平均 359 / 中央値 320 / 最小 160 / 最大 960、cond 平均 320 / 中央値 321 / 最小 124 / 最大 716 |
| 前処理 | 512² にリサイズ（BILINEAR、縦横比は保たない）、cond は 1 px の白い輪郭画像（cond_scale_factor 1 → 512²） |

### モデル・LoRA

| 項目 | 値 |
|---|---|
| ベース | `black-forest-labs/flux-3-image-klein-base`（蒸留なし。revision 3f7fa5e） |
| attention mask | `base`（hard_attn_block_range は既定 [5,18]） |
| cond トークン | 使う（`is_use_cond_token`）、値 0 のトークンは捨てる（`is_filter_cond_token`） |
| LoRA rank | **32**（FLUX.1 / FLUX.2 の UECFoodPix 版と同じ。ラーメン版は 16） |
| default LoRA | `all-linear-in-dit-blocks+modulation`（txt / noisy img の mode blocks・joint blocks の全 Linear + 変調） |
| cond LoRA | `x_embedders.image_cond`・image_cond の変調・`mode_blocks.image_cond` の全 Linear・joint blocks の全 Linear |
| LoRA の bias / 初期化 | bias なし / 通常の初期化（gaussian なし） |
| 学習パラメータ数 | 135.07M（全体の 2.38%） |
| テキスト | max_sequence_length 512、regional_max_sequence_length 50、caption ごとに 80 の倍数へ右パディング、txt_ids は caption ごとに 0 から振り直す（restart） |
| ノイズ | `FlowMatchEulerDiscreteScheduler(shift=6.93)`、weighting_scheme none |

### 最適化

| 項目 | 値 |
|---|---|
| オプティマイザ | AdamW（β1 0.9、β2 0.999、weight decay 1e-2、ε 1e-8）、勾配クリップ 1.0 |
| 学習率 | 1e-4、constant、warmup なし |
| バッチ | 2 / GPU × 8 GPU = **全体 16**、勾配累積なし |
| ステップ数 | **5500**（5500 × 16 = 88k サンプル ≒ 9.8 epoch） |
| 精度 | bf16 mixed precision、TF32 なし |
| 勾配チェックポイント | あり（A40 46GB のため） |
| seed | 42 |
| validation | 500 ステップごと、test 側で num_validation_images 8・num_images_per_prompt 4、Base 20 ステップ CFG 3.5（tensorboard に画像） |
| checkpoint | 500 ステップごとに保存。容量削減のため **1000 / 3000 / 5000 / 5500 だけ残した**（他は削除） |

### 環境

| 項目 | 値 |
|---|---|
| マシン | gp41、NVIDIA A40 46GB × 8（`config/accelerate_8gpu_config.yaml`、`NCCL_P2P_LEVEL=NVL`） |
| venv | `/export/ssd2/lan-y/venvs/Seg2Any/.venv-flux3`（ssd3 が落ちていたので作り直し）。Python 3.11、torch 2.8.0+cu128、torchvision 0.23.0、**diffusers 0.41.0.dev0 = main の af4271c（2026-10-15）に固定**、transformers 5.18.0、peft 0.21.2、accelerate 1.15.0 |
| 重み | `HF_HOME=/export/ssd2/lan-y/huggingface`（space0 からコピー。space0 から直接読むと mmap が遅い）。**token もこの HF_HOME に置く必要がある**（無いと gated で 401） |
| 注意 | 以前の `.venv-flux3`（ssd3）の diffusers のコミットは記録が無く、今回のものと同じとは限らない |

### 学習の経過

- 2026-10-24 12:25 開始。速度は 3.3〜3.5 秒/ステップ、GPU メモリは約 20〜23GB/GPU
- validation 1 回あたり約 2.5 分
- loss：最初 0.78 前後 → 5000 ステップ付近で 0.55〜0.8（ばらつき大）
- 2026-10-24 18:06 正常終了（5500 ステップ / 5 時間 40 分、平均 3.7 秒/ステップ、validation 込み）。最終 loss 0.61 前後。checkpoint は 1000 / 3000 / 5000 / 5500 を残した

### 生成と比較（test 10 枚、checkpoint-5500）

- 生成：`gen_uecfoodpix_flux3_test.py`（入力は学習と同じ Dataset の test、512² 正方、seed = stem、attention base）
  - Base + LoRA 20 ステップ CFG 3.5 → `result_f3/UECFOODPIX_flux3/plate_test10_base20/gen`
  - 蒸留版 + LoRA 16 ステップ → `result_f3/UECFOODPIX_flux3/plate_test10_distill16/gen`
  - 速度（gp41 A40、1 枚）：Base 20 ステップ CFG 約 17 秒、蒸留版 16 ステップ 約 8 秒
- FLUX.2 皿版は既存の `result_f2/UECFOODPIX_flux2/uecfoodpix_plate_flux2_test10_base`（自作マスク、Base 20 ステップ CFG 4、base）を流用。この 10 枚では自作マスクと共有マスクの一致率 99.8% 以上・クラス構成も同じなので、入力はほぼ同じ
- パネル：`make_uecfoodpix_flux3_compare.py` → `result_f3/UECFOODPIX_flux3/compare_test10/panel_<stem>.jpg`、`all_panels.jpg`（マスク | GT | FLUX.2 Base | FLUX.3 Base | FLUX.3 蒸留版、縮小なし）
- 見た目：
  - 3 条件ともマスクの配置・皿の形にはよく従う
  - FLUX.3 は料理の描写が caption に忠実：照り焼き魚のタレ（4357。FLUX.2 はカツ・ステーキ風）、エビフライ・ハンバーグのソース（16164）、納豆の豆粒（3061）、マカロニサラダ（4357）
  - 色調が大きく違う：FLUX.2 は暗くコントラストが強い（黒いお盆・暗い背景）。FLUX.3 は明るく、ピンクや薄緑の食堂のお盆が多く、やや白っぽく褪せた感じ
  - 蒸留版は Base とほぼ同じ構図・品質で、少し彩度・コントラストが低い
  - "food"（others）クラスは曖昧で、モデルごとに中身がばらばら（6910 は GT が海苔のパック、FLUX.2 はきゅうり、FLUX.3 は揚げ物）

### 境界 recall（test 10 枚、`eval_uecfoodpix_flux3_boundary.py`）

指標は `utils/boundary_metric.py`（flux2_eval_val55.py と同じ。Canny エッジがマスク境界の 3 px 以内にある割合、512²）。マスクは入力と同じ 512²（nearest-exact）。
境界を「料理が接する境界（food）」と「皿とテーブルの境界（plate|table）」に分けても集計した。FLUX.2 は入力に使った自作マスクで測る（共有マスクで測っても 0.748 でほぼ同じ）。

| 条件 | recall | food | plate\|table | precision | エッジ密度 |
|---|---|---|---|---|---|
| GT 写真（512² に縮小） | 0.563 | 0.631 | 0.517 | 0.300 | 0.053 |
| FLUX.2 Base 20 ステップ CFG 4 | **0.747** | **0.869** | **0.668** | 0.406 | 0.050 |
| FLUX.3 Base 20 ステップ CFG 3.5 | 0.623 | 0.760 | 0.534 | **0.489** | 0.031 |
| FLUX.3 蒸留版 16 ステップ | 0.450 | 0.519 | 0.409 | 0.281 | 0.041 |

- **蒸留版は境界を守れていない**：recall は Base より 0.17 低く、10 枚すべてで Base 以下（food 境界でも 0.76 → 0.52）。見た目の品質が良く見えても、料理・皿の輪郭がマスクからずれる（16164 は皿の縁がマスクより外に出て、ハンバーグのソースもはみ出す。6286 は角皿やお椀の位置がずれる）
- エッジの少なさのせいではない：蒸留版のエッジ密度（0.041）は FLUX.3 Base（0.031）より多いのに recall は低い。エッジはあるが、マスク境界ではない場所にある（precision も 0.28 と低い）
- ラーメン val10（§8）では蒸留版と Base の差は 0.04 だったが、UECFoodPix では 0.17 と大きい
- FLUX.3 Base も FLUX.2 より 0.12 低い（ラーメン val55 と同じ傾向）。とくに皿とテーブルの境界が弱い。FLUX.3 は明るく白っぽい色調で、白い皿と薄い色のお盆の境目のコントラストが低いのも一因と思われる
- GT 写真の recall が 0.56 と低いのは、マスクの皿・テーブル境界が粗い（細かいギザギザ・飛び地がある）ため。生成モデルはマスクどおりに輪郭を描くので GT より高く出る
- 重ね描き：`result_f3/UECFOODPIX_flux3/compare_test10/boundary_<stem>.jpg`（マスク境界を赤で重ねた 512² 画像、GT | FLUX.2 | FLUX.3 Base | 蒸留版）、数値は `boundary.json`

### cond を序盤のステップだけ入れる（test 10 枚、`eval_uecfoodpix_flux3_condsteps.py`）

`Flux3RegionalPipeline(cond_steps=k)` を追加した：cond（輪郭）トークンは最初の k ステップだけ入れ、それ以降は cond なし（領域アテンションマスク・regional caption はそのまま）。None（既定）は全ステップで、従来と同じ。
生成は `gen_uecfoodpix_flux3_test.py --cond_steps k` → `result_f3/UECFOODPIX_flux3/cond_steps/{base20_cs<k>,distill16_cs<k>}`、パネル `cond_steps/panel_<stem>.jpg`（上段 Base 20/10/5/2/0、下段 蒸留版 16/8/4）。

| 条件 | recall (3px) | food | plate\|table | recall (8px) | precision |
|---|---|---|---|---|---|
| Base cond 20/20 | 0.623 | 0.760 | 0.534 | 0.721 | 0.489 |
| Base cond 18/20 | 0.614 | 0.744 | 0.529 | 0.717 | 0.482 |
| Base cond 16/20 | 0.591 | 0.713 | 0.513 | 0.704 | 0.473 |
| Base cond 14/20 | 0.567 | 0.676 | 0.496 | 0.691 | 0.463 |
| Base cond 12/20 | 0.523 | 0.613 | 0.466 | 0.662 | 0.445 |
| Base cond 10/20 | 0.479 | 0.556 | 0.433 | 0.636 | 0.423 |
| Base cond 5/20 | 0.312 | 0.327 | 0.310 | 0.520 | 0.336 |
| Base cond 2/20 | 0.138 | 0.140 | 0.141 | 0.302 | 0.215 |
| Base cond 0/20 | 0.093 | 0.096 | 0.091 | 0.205 | 0.160 |
| 蒸留版 cond 16/16 | 0.450 | 0.519 | 0.409 | 0.643 | 0.281 |
| 蒸留版 cond 8/16 | 0.382 | 0.428 | 0.357 | 0.599 | 0.249 |
| 蒸留版 cond 4/16 | 0.304 | 0.331 | 0.291 | 0.532 | 0.201 |

- **配置は最初の 2〜5 ステップ（20 中）で決まる**：cond 10/20・5/20 は 20/20 と見た目の構図がほぼ同じ（皿・料理の位置と形が一致）。2/20 で崩れ始め、0/20 は配置が全く別物になる（領域アテンションマスクだけでは配置を保てない）。0/20 の recall 0.09 は画像の破綻ではなく配置違い
- **recall (3px) は見た目の配置よりずっと敏感**：10/20 と 5/20 は見た目では配置を守っているのに recall は 0.62 → 0.48 → 0.31 と大きく下がる。後半ステップの cond は、マスクの輪郭ぴったりにエッジを立てる（輪郭をなぞる）働きをしている
- **cond を減らしても綺麗にはならない**：Base で cond を後半から外しても、画質・質感はほぼ変わらず、輪郭が少し柔らかくなる程度。蒸留版が綺麗に見えるのは、境界を守らないからではなくモデル自体の差と思われる（蒸留版 16/16 と Base 10/20 は recall がほぼ同じでも、蒸留版の方が彩度・質感がよい）
- 蒸留版の境界への従い方は「Base で cond を後半 1/2 外した」くらいに相当する（recall 3px・8px ともほぼ同じ）
- 20/20〜10/20 の間（18/16/14/12 を追加）：recall は 2 ステップごとに 0.01〜0.04 ずつなだらかに下がり、どこかで急に落ちる点はない（10 枚中ほぼ全部で単調減少）。見た目は 20/20〜10/20 でほとんど区別できない（ソースの縁・皿の縁が少し柔らかくなる程度）。この範囲は「輪郭をどれだけぴったりなぞるか」だけを連続的に調整するつまみになっている
- 次の候補：エッジ一致ではなく領域単位で配置を測る指標（食事画像のセグメンテーションモデルで mIoU、クラスごとの位置・面積のずれ）。蒸留版の境界を上げるなら、蒸留版での cond の効かせ方（cond2image_attention_weight、蒸留版で LoRA を学習）を試す

> 注: ここに出てくる段階確認スクリプトは整理のため削除済み。`git show 691c47a:<ファイル名>` で取り出せる。
