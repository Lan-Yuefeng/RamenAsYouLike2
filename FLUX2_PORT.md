# Seg2Any の FLUX.2 [klein] 4B への移植メモ

Seg2Any（セグメンテーションマスクから画像を生成する手法）の基盤モデルを、**FLUX.1-dev（12B）から FLUX.2 [klein] 4B に置き換えた**作業の記録です。
「何を変えたか」「なぜそうしたか」「どう確かめたか」「結果どうだったか」を、後から読んでも追えるように書いています。

最終更新：2026-10-07（val 55 枚での評価と CFG の既定変更まで）

---

## 0. 3 行でまとめると

1. FLUX.2 [klein] 4B 用に Seg2Any の **transformer・推論パイプライン・学習スクリプトを新しいファイルとして作り直した**。FLUX.1 版のファイルには一切手を入れていないので、FLUX.1 版はこれまでどおり動く。
2. 移植が正しいことは、純正モデルとの出力一致や学習と推論の入力一致など、**合計 19 項目のテストで確認した**。
3. UEC-Ramen555 で LoRA を学習し直した（3000 ステップ）。当初は境界が FLUX.1 より甘かったが、原因は CFG の組み方だった。直した後は、**val 55 枚で境界の一致度が FLUX.1 と統計的に同等**、元写真への近さ（CLIP-I）は FLUX.1 より高い（§11.5）。GPU メモリは約半分で済む。

---

## 1. 現在の状況

| ステップ | 内容 | 状態 |
|---|---|---|
| 1 | 新しい venv を作り、素の klein-4B で画像を生成する | 完了 |
| 2 | transformer に Seg2Any の仕組み（cond トークン、領域マスク、2 種類の LoRA）を移植する | 完了 |
| 3 | 推論パイプラインを移植する（テキストエンコーダ T5/CLIP → Qwen3、新しい VAE、4 軸の位置 ID） | 完了 |
| 4 | 学習スクリプトを移植する | 完了 |
| 5 | UEC-Ramen555 で LoRA を学習し直し、FLUX.1 版と比べる | **完了**（学習 3000 ステップ：10/6 23:05 に `checkpoint-3000` を保存。比較・評価は §11。10/7 に CFG の既定を変更：§11.4〜11.5） |

日付は、特に書かない限り 2026-10-06。

---

## 2. クイックスタート

### 使う venv とモデル
- venv：`~/Seg2Any/.venv-flux2`（FLUX.1 用の `.venv` とは別。混ぜないこと）
- モデル：ssd3 にキャッシュしてある。**どのコマンドも次の 2 つの環境変数を付けて実行する**（space0 から読むとロードがとても遅い）。
  ```bash
  export HF_HOME=/export/ssd3/lan-y/huggingface
  export HF_HUB_OFFLINE=1
  ```

| モデル | 何に使うか |
|---|---|
| `black-forest-labs/FLUX.2-klein-base-4B` | 蒸留なしの Base 版。**LoRA の学習**と、丁寧な推論（CFG あり・50 ステップ、1 枚約 12.6 秒）に使う |
| `black-forest-labs/FLUX.2-klein-4B` | 蒸留版。**速い推論**（CFG なし・4 ステップ、1 枚約 0.85 秒）に使う。Base 版で学習した LoRA がそのまま載る |

学習済み LoRA：`ckpt/ramen_flux2/checkpoint-3000/{default,cond}/`（250 ステップごとのチェックポイントもある）

### 推論（マスク → 画像）
入力形式は FLUX.1 版の `infer.py` と同じで、フォルダに `*.png`（色分けマスク）と `*.json`（caption と seed）を置く。
```bash
cd ~/Seg2Any
# 蒸留版で速く生成（既定。ステップ数は自動で 4）
.venv-flux2/bin/python infer_flux2.py --seg_mask_path examples_Ramen \
    --lora_ckpt_path ckpt/ramen_flux2/checkpoint-3000 --output_path result/flux2_ramen
# Base 版で丁寧に生成（ステップ数は自動で 50、CFG 4.0）
.venv-flux2/bin/python infer_flux2.py ... --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B
# Base 版でも 20 ステップでほぼ同じ品質になる（5.0 秒/枚。§11.3）
.venv-flux2/bin/python infer_flux2.py ... --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B --num_inference_steps 20
# 2026-10-07 から、CFG のネガティブ側には cond を入れないのが既定（境界に忠実。§11.4〜11.5）。昔の挙動に戻すとき
.venv-flux2/bin/python infer_flux2.py ... --negative_with_cond
# LoRA なし（領域マスクだけで制御）。学習前の cond は意味をなさないので --no_cond を付ける
.venv-flux2/bin/python infer_flux2.py --seg_mask_path examples_Ramen --no_cond
```

### 比較画像を作る
```bash
.venv-flux2/bin/python flux2_step5_compare.py --lora_ckpt_path ckpt/ramen_flux2/checkpoint-3000
# → result_f2/flux2_checkpoint-3000/panel.jpg（マスク | 元写真 | FLUX.1 | FLUX.2 Base | FLUX.2 蒸留版）
```

### 学習
```bash
bash train_ramen_flux2.sh                                   # 8 GPU、config/seg2any_ramen_flux2.yaml の設定で
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 NUM_PROCESSES=7 bash train_ramen_flux2.sh   # GPU を絞るときは両方指定する
bash train_ramen_flux2.sh --resume_from_checkpoint=latest   # 途中から再開
bash train_ramen_flux2.sh --trainer.gradient_checkpointing=True   # 48 GB 級の GPU や、他人と相乗りしそうなとき
```
- 出力先は `ckpt/ramen_flux2/checkpoint-N/{default,cond}/`（`ckpt` は space0 へのシンボリックリンク）
- 学習中の validation 画像は tensorboard のログにしか出ない。1 枚の一覧画像として取り出すには次のようにする。
  ```bash
  .venv-flux2/bin/python flux2_val_panel.py --step 1500     # → result_f2/flux2_val/step1500.jpg
  ```

### テスト（実装を触ったら回す）
```bash
.venv-flux2/bin/python flux2_step2_test.py   # transformer（10 項目）
.venv-flux2/bin/python flux2_step3_test.py   # パイプライン（5 項目）
.venv-flux2/bin/python flux2_step4_test.py --ckpt <checkpoint>   # LoRA の保存/読み込み、学習と推論の一致（4 項目）
```

---

## 3. ファイル一覧

### 新しく作ったもの
| ファイル | 役割 |
|---|---|
| `src/models/transformer_flux2.py` | **Seg2Any 版の FLUX.2 transformer**（`Flux2RegionalTransformer2DModel`） |
| `src/pipelines/pipeline_flux2.py` | **Seg2Any 版の FLUX.2 推論パイプライン**（`Flux2RegionalPipeline`） |
| `infer_flux2.py` | 推論スクリプト（`infer.py` の FLUX.2 版） |
| `train_flux2.py` | 学習スクリプト（`train.py` の FLUX.2 版） |
| `config/seg2any_ramen_flux2.yaml` | ラーメン学習の設定（`seg2any_ramen.yaml` の FLUX.2 版） |
| `train_ramen_flux2.sh` | 学習の起動スクリプト（`train_ramen.sh` の FLUX.2 版） |
| `flux2_smoke.py` | 素の klein で 1 枚生成する動作確認 |
| `flux2_step2_test.py` / `flux2_step3_test.py` / `flux2_step4_test.py` | 各ステップの検証 |
| `flux2_step3_compare.py` | LoRA なしで、テキスト位置の付け方 2 通りを見比べる一覧を作る |
| `flux2_step5_compare.py` | 学習後の比較一覧（FLUX.1 / FLUX.2 Base / FLUX.2 蒸留版）を作る |
| `flux2_steps_sweep.py` | Base 版でステップ数を変えて、生成結果と生成時間を記録する |
| `flux2_cfg_sweep.py` | CFG の強さを変えて生成し、境界の一致度（Canny エッジとの recall）を測る |
| `flux2_eval_val55.py` | val 55 枚の各条件を、境界・CLIP-T・CLIP-I・KID/FID で評価する |
| `result_f2/run_val55_gen.sh` | `flux2_eval_val55.py` 用の生成（4 条件） |
| `utils/boundary_metric.py` | 境界の一致度の計算（上の 2 つのスクリプトで共通に使う） |
| `flux2_val_panel.py` | tensorboard から validation 画像を取り出す |
| `.venv-flux2/` | FLUX.2 用の venv |

### 出力物
| 場所 | 中身 |
|---|---|
| `ckpt/ramen_flux2/checkpoint-{250,...,3000}/` | 学習した LoRA（250 ステップごと） |
| `result_f2/flux2_checkpoint-3000/` | 最終比較（`panel.jpg`、各画像、`timing.json`） |
| `result_f2/flux2_steps_sweep/` | Base 版の推論ステップ数比較（50/40/30/20/10、画像・時間・README。§11.3） |
| `result_f2/flux2_cfg_sweep/` | CFG の強さと境界への忠実さの比較（画像・境界スコア・README。§11.4） |
| `result_f2/val55/` | **val 55 枚での最終評価**（FLUX.1 / FLUX.2 旧既定・新既定 / 蒸留版。境界・CLIP・KID。README。§11.5） |
| `examples_Ramen_val55/` | val 55 枚すべてのマスクと caption（学習時と同じ文言） |
| `result_f2/flux2_val/step{250,1000,1500,2500,3000}.jpg` | 学習途中の validation 画像 |
| `result_f2/flux2_step3_compare/panel.jpg` | LoRA なしでの比較 |
| `ckpt/ramen_flux2_train*.log` | 学習ログ（中断ごとに分けて保存） |

### 変更していないもの（そのまま使っている）
- FLUX.1 版一式：`src/models/transformer_flux.py`、`src/models/attention.py`、`src/pipelines/pipeline_flux.py`、`utils/lora_controller.py`、`train.py`、`infer.py`
- データまわり：`dataset/ramen_dataset.py`、`dataset/ramen_captions.py`、`dataset/collate_fn.py`、`dataset/group_sampler.py`、`data/ramen/{train,val}.txt`、`examples_Ramen/`
- FLUX.2 版は、FLUX.1 版の関数の一部（`prepare_attention_mask`、`filter_cond_token`、`load_seg_map_and_prompt`）を **import して流用**している。FLUX.1 版を書き換えると FLUX.2 版にも影響する点に注意。

---

## 4. 前提知識：Seg2Any はどう動いているか

移植の中身を理解するのに必要な範囲で、Seg2Any の仕組みをまとめる。

### 4.1 1 本の系列にすべてを並べる
FLUX は MM-DiT で、テキストと画像のトークンを 1 本の系列につなげてアテンションを取る。Seg2Any はそこに **cond トークン**（マスクの輪郭画像を VAE で潜在に変換したもの）を足し、次の順に並べる。

```
[ global caption | regional caption 1 | ... | regional caption n | (pad) | noisy 画像 | cond (輪郭) | (pad) ]
```

- **global caption**：画像全体の説明（例：「豚骨スープのラーメンの俯瞰写真…」）
- **regional caption**：領域ごとの説明（例：「チャーシュー 1 枚、脂の縁…」）
- **cond トークン**：輪郭線を含む 16×16 パッチだけを残す（真っ黒なパッチは捨てる＝`filter_cond_token`）

### 4.2 アテンションマスクで「どの領域がどの説明を見るか」を決める
`prepare_attention_mask` が系列全体の (S × S) マスクを作る。
- 各 regional caption は、対応するマスク領域の画像トークンとだけアテンションを取る
- global caption は画像全体と取る
- cond トークンは画像トークンと互いに見える

`attention_mask_method="hard"` の場合、一部のブロック（`hard_attn_block_range`）だけで、前景の画像トークン同士も領域ごとに隔離する、より強いマスクを使う。

### 4.3 LoRA を 2 種類使い分ける
同じ重み（Linear）を、通すトークンの種類によって別の LoRA で使う。
- **default LoRA**：テキストと noisy 画像のトークン用
- **cond LoRA**：cond トークン用

切り替えは `select_lora` という context manager で、使わない側の LoRA の scaling を一時的に 0 にして行う。cond トークンは「ノイズなし（timestep = 0）」として変調する。

---

## 5. FLUX.1 と FLUX.2 の違いと、その対応（全体像）

移植で手を入れた理由は、ほぼすべてこの表の違いから来ている。

| 項目 | FLUX.1-dev | FLUX.2 klein 4B | 移植での対応 |
|---|---|---|---|
| 規模 | 12B（double 19 層 + single 38 層） | 4B（double 5 層 + single 20 層、hidden 3072） | `hard_attn_block_range` を同じ比率に換算（[19,37] → [5,14]） |
| テキストエンコーダ | T5-XXL（系列）+ CLIP（pooled） | **Qwen3**（hidden states の 9・18・27 層目を連結して 7680 次元）。pooled なし | テキスト処理を作り直し（§8） |
| caption の入れ方 | そのまま | **チャットテンプレートで包む**（12 トークン増える） | 学習・推論とも包む |
| テキストの位置 ID | すべて 0（位置情報なし） | `(0,0,0,l)`。l はトークン番号 | **caption ごとに l を 0 から振り直す**（§8.3） |
| 画像の位置 ID（RoPE） | 3 軸 | **4 軸 (T, H, W, L)** | 4 軸 ID を使う |
| VAE | 16ch、`(x - shift) * scale` | **32ch**、2×2 にパッチ化して 128ch、BatchNorm の統計で正規化 | klein 純正のエンコード処理を使う |
| AdaLN の変調 | 各ブロックが自前の Linear を持つ | **モデル全体で 3 つの Linear を共有** | cond 用の変調をモデルの先頭で作って各ブロックに渡す（§7） |
| single ブロック | QKV、MLP、出力が別々の Linear | **QKV と MLP の入力が 1 つの Linear に融合**、出力も融合 | 融合 Linear ごと LoRA を切り替える |
| FF | GELU | SwiGLU（名前も `ff.linear_in/out`） | LoRA 対象の名前を変更 |
| guidance | dev は guidance 埋め込みを持つ（3.5） | klein は持たない。Base 版は true CFG、蒸留版は CFG なし | パイプラインで切り替え |
| 必要な diffusers | 0.32 | **0.37 以降** | 別の venv（§6） |

---

## 6. Step 1：環境構築

### やったこと
- 新しい venv `.venv-flux2` を作った（既存の `.venv` はそのまま）
- `flux2_smoke.py` で、素の klein-4B で画像を生成できることを確認した

### venv の中身

| パッケージ | `.venv`（FLUX.1 用） | `.venv-flux2`（FLUX.2 用） | 理由 |
|---|---|---|---|
| Python | 3.10.20 | 3.10 | 同じ |
| torch | 2.6.0+cu124 | **2.8.0+cu128** | gp42 の GPU（Blackwell、sm_120）は cu124 版では動かない。cu128 版なら Ampere / Ada / Blackwell のどれでも動く（gp40 の A6000 でも確認済み） |
| diffusers | 0.32.2 | **0.41.0** | Flux2 は 0.37 以降で入った |
| transformers | 4.55.2 | **5.18.0** | Qwen3 のため |

作り方（すべて `uv pip`）：
```bash
cd ~/Seg2Any
uv venv .venv-flux2 --python 3.10 --prompt seg2any-flux2
VIRTUAL_ENV=$PWD/.venv-flux2 uv pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=$PWD/.venv-flux2 uv pip install "diffusers>=0.37" transformers accelerate peft safetensors \
    sentencepiece protobuf pillow numpy opencv-python-headless omegaconf
# 後から足したもの（infer.py / dataset / utils が import する）
VIRTUAL_ENV=$PWD/.venv-flux2 uv pip install matplotlib pandas joblib tensorboard pyarrow pycocotools \
    "git+https://github.com/cocodataset/panopticapi.git"
```

### 結果（gp42、1024²、4 ステップ）
- モデルの読み込み 43 秒、生成は 2 回目以降 **1.0 秒**、GPU メモリのピーク **17.3 GiB**（FLUX.1 では 31.6 GB）
- klein-4B と klein-base-4B は、transformer の構成がまったく同じ（重みだけが違う）

---

## 7. Step 2：transformer の移植（`src/models/transformer_flux2.py`）

### 7.1 方針
- diffusers の `Flux2Transformer2DModel` を**継承**し、ブロックとアテンション処理だけを Seg2Any 用に差し替えた。
- **新しいパラメータは 1 つも足していない**。重みのキー名も純正と同じなので、klein の重みが `from_pretrained` でそのまま読める。
- cond トークンは、画像トークンと**同じ重み**を通し、LoRA だけを 'cond' に切り替える（FLUX.1 版と同じ考え方）。
- 系列の並びとマスクの形式を FLUX.1 版とそろえた。そのおかげで、マスクを作る関数を FLUX.1 版からそのまま流用できる。

### 7.2 クラスの構成

```
Flux2RegionalTransformer2DModel          … モデル全体（Flux2Transformer2DModel を継承）
 ├─ Flux2RegionalTransformerBlock × 5   … double-stream（txt と img が別の重み）
 │    └─ Flux2RegionalAttnProcessor     … txt / img / cond を 1 回のアテンションで
 └─ Flux2RegionalSingleTransformerBlock × 20 … single-stream（txt と img が同じ重み）
      └─ Flux2RegionalParallelSelfAttnProcessor … 融合 Linear を img と cond で LoRA を切り替えて通す
```

### 7.3 forward の流れ（モデル全体）
1. **timestep の埋め込みと変調**：画像用（default LoRA）とテキスト用を作る。cond があれば、timestep = 0 の埋め込みから cond 用の変調を cond LoRA で作る。
   - FLUX.1 では変調の Linear が各ブロックの中にあったので、ブロックの中で切り替えていた。FLUX.2 では変調がモデル全体で共有されているので、**先頭でまとめて作って各ブロックに渡す**ように変えた。
2. **入力の射影**：`x_embedder` を、画像は default、cond は cond の LoRA で通す。
3. **位置埋め込み（RoPE）**：txt と img は全サンプル共通。cond はサンプルごとに残るトークンが違うので、サンプルごとに計算する。txt も、caption の構成がサンプルごとに違えばサンプルごとに計算する（§8.3）。
4. **ブロックを順に通す**：ブロックの通し番号（double 0〜4、single 5〜24）が `hard_attn_block_range` に入っていれば hard マスク、そうでなければ通常のマスクを使う。
5. **出力**：画像トークンだけを取り出して返す。

### 7.4 アテンション処理の中身（double-stream の場合）
1. 画像の Q/K/V を default LoRA で作り、QK-norm をかける
2. テキストの Q/K/V を作る（テキスト専用の重みなので、LoRA の切り替えは不要）
3. [テキスト, 画像] をつないで RoPE をかける
4. cond の Q/K/V を、**画像と同じ重み**を cond LoRA で通して作り、cond 用の RoPE をかけて後ろにつなぐ
5. マスク付きで 1 回だけアテンションを取る
6. 出力を [テキスト, 画像, cond] に切り分け、それぞれの出力射影に通す

single-stream も流れは同じ。ただし QKV と MLP の入力が 1 つの Linear（`to_qkv_mlp_proj`）に融合しているので、その Linear ごと LoRA を切り替える。

### 7.5 細かい変更点
- **`select_lora` の修正版を同じファイルに置いた**：元の `utils/lora_controller.py` は、抜けるときに scaling を**必ず 1 に戻す**。そのため LoRA の強さ（`scale`）を変えたときや `lora_alpha ≠ rank` のときに値が壊れる。修正版は元の値に戻す。今の設定（alpha = rank）ではどちらでも結果は同じ。FLUX.1 側のファイルは触っていない。
- **`hard_attn_block_range` の既定値**：FLUX.1 の [19, 37] は「double 全 19 層の直後から、single の前半 19/38 層」を意味する。同じ比率で、klein では **[5, 14]**（`default_hard_attn_block_range()`）。
- **使わない機能を外した**：参照画像編集用の KV キャッシュと、テンソル並列/コンテキスト並列の設定（cond の長さがサンプルごとに違うので使えない）。
- テンソルのレイアウトが FLUX.1（[B, H, S, D]）と FLUX.2（[B, S, H, D]）で違うので、SDPA の直前だけ転置する（`_sdpa`）。

### 7.6 LoRA の対象モジュールの対応表

| 役割 | FLUX.1 のモジュール | FLUX.2 のモジュール |
|---|---|---|
| 入力の射影 | `x_embedder` | `x_embedder` |
| 変調（double） | `transformer_blocks.N.norm1.linear` | `double_stream_modulation_img.linear`（モデルに 1 つ） |
| アテンション（double） | `transformer_blocks.N.attn.to_q/k/v`、`to_out.0` | 同じ名前 |
| FF（double） | `transformer_blocks.N.ff.net.0.proj`、`ff.net.2` | `transformer_blocks.N.ff.linear_in`、`ff.linear_out` |
| 変調（single） | `single_transformer_blocks.N.norm.linear` | `single_stream_modulation.linear`（モデルに 1 つ） |
| QKV と MLP の入力（single） | `proj_mlp` と `attn.to_q/k/v`（別々） | `attn.to_qkv_mlp_proj`（融合） |
| 出力（single） | `proj_out` | `attn.to_out` |

### 7.7 検証（`flux2_step2_test.py`、10 項目すべて PASS）
学習前なので「出力が良いか」は判断できない。代わりに、**正しい実装なら必ず成り立つ性質**を確かめた。

| # | 確かめたこと | 結果 | 何がわかるか |
|---|---|---|---|
| – | 重みのキーと値が純正と同一 | 169 keys 一致 | 純正の重みがそのまま読める |
| 1 | cond もマスクも無ければ、純正と出力が完全一致 | 差 0 | 差し替えたブロックが純正と同じ計算をしている |
| 2 | cond を完全に隔離するマスクを付ければ、cond なしと一致 | 差 0 | cond の結合と切り出しの位置が正しい |
| 3 | 隔離しなければ cond が出力に効く | 差 1.5 | cond が実際に使われている |
| 4 | hard マスクは、指定した範囲のブロックにだけ効く | 差 0 / 0 | ブロック範囲の判定が正しい |
| 5a | cond LoRA に乱数を入れても、cond なしの出力は変わらない | 差 0 | cond LoRA が画像側に漏れていない |
| 5b | default LoRA に乱数を入れると出力が変わる | 差 0.4 | default LoRA が効いている |
| 6 | 勾配チェックポイントありで backward が通り、両方の LoRA に勾配が流れる | OK | 学習できる |
| 7 | サンプルごとに違うテキスト位置 ID を入れても、1 サンプルずつ計算した結果と一致 | 差 0 | §8.3 の振り直しがバッチでも正しい |

蒸留版と Base 版の両方で PASS。

---

## 8. Step 3：推論パイプラインの移植（`src/pipelines/pipeline_flux2.py`）

### 8.1 方針
- klein 純正の `Flux2KleinPipeline` を継承した。latent の準備、VAE、スケジューラ、CFG の判定は純正のものをそのまま使う。
- 領域マスクの作成と cond の間引きは、**FLUX.1 版の関数をそのまま呼ぶ**（系列の並びが同じなので使える）。
- 作り直したのはテキストの処理と cond のエンコード。

### 8.2 テキストの処理
1. 各 caption（global と各 regional）を **Qwen3 のチャットテンプレートで包む**。klein は事前学習で常にこの形で入力されているので、それに合わせた。
2. **1 本ずつ Qwen3 に通し**、パディングを取り除いてから連結する。Qwen3 は causal（前のトークンしか見ない）なので、右にパディングしても実トークンの値は変わらない。
3. 長すぎる caption は、テンプレートで包む**前**に caption 部分だけを切り詰める。テンプレートの末尾を残すため。
   - そのため `regional_max_sequence_length=50` は「**caption 部分**の上限」という意味になった（テンプレートの 12 トークンは別）。

### 8.3 テキストの位置 ID：caption ごとに 0 から振り直す
FLUX.2 はテキストのトークン番号を位置情報（L 軸）として使う。global と regional を連結するとき、番号の付け方には 2 通りある。

| 付け方 | 内容 | 問題点 |
|---|---|---|
| 通し番号 | 系列全体で 0, 1, 2, … | 同じ caption でも、並び順や global の長さで番号が変わる。合計が 512 を超えると、事前学習で見たことのない位置になる |
| **振り直し（採用）** | caption ごとに 0 から | 特になし。複数の caption が同じ番号を持つが、どの領域がどの caption を見るかはマスクで決まるので困らない |

- テキスト同士のアテンションは、もともと caption の中に閉じている（マスクで遮断）。RoPE は位置の差しか見ないので、**テキスト同士については、どちらの付け方でも結果は同じ**。
- 違いが出るのは**テキストと画像の間**。画像トークンの L は常に 0 なので、テキスト側の番号がそのまま効く。
- 振り直しなら、どの caption も「プロンプトの先頭」に見え、事前学習の範囲（0〜511）に必ず収まる。各 caption を Qwen3 に別々に通している点とも辻褄が合う。
- LoRA なしで比べた結果（§8.5）では、10 例の範囲ではほとんど差がなかった（最長 284 トークンで、512 を超えなかったため）。

### 8.4 その他の違い

| 項目 | FLUX.1 版 | FLUX.2 版 |
|---|---|---|
| cond のエンコード | `latent_dist.sample()`（毎回ランダム） | `latent_dist.mode()`（決定的。klein 純正と同じ） |
| スケジューラの shift | 線形の `calculate_shift` | klein の `compute_empirical_mu` |
| CFG | dev の guidance 埋め込み（3.5） | 蒸留版は CFG なし。Base 版は `guidance_scale > 1` で true CFG（既定 4.0）。ネガティブ側も global と各 region に空文字を入れ、マスクも作り直す |
| 既定のステップ数 | 28 | 50（`infer_flux2.py` は蒸留版なら 4、Base 版なら 50 に自動で切り替え） |

### 8.5 検証

**`flux2_step3_test.py`（5 項目すべて PASS）**

| # | 確かめたこと | 結果 |
|---|---|---|
| 1 | パディングを除いた Qwen3 の埋め込みが、klein 純正の先頭部分と一致 | fp32 で相対誤差 1e-6 |
| 2 | caption ごとの長さと位置 ID が正しい（振り直し） | 例：global 82 + 7 領域、合計 284 トークン |
| 2b | 通し番号モードも正しい | 最大 283 |
| 3 | cond の間引き：残るトークンが「輪郭を含むパッチ」と一致し、位置 ID もその座標を指す | 1024 中 434 を残す |
| 4 | cond あり・hard マスクで 1 枚生成できる | 0.8 秒、17.1 GiB |

> **注意（bf16 での比較）**：bf16 で純正と比べると相対 1% ほどずれるが、系列長が違うと演算カーネルが変わるためで、実装の誤りではない（fp32 では一致）。Qwen3 の hidden states は先頭トークンで |x| ≈ 16000 にもなるので、絶対誤差で見ると大きく見える。

**LoRA なしでの目視確認**（`result_f2/flux2_step3_compare/panel.jpg`）
- **学習前でも、領域マスクだけで配置がかなり効く**。海苔は全例でマスクの位置に出て、卵・レンゲ・チャーシュー・ナルトもおおむねマスクどおり。
- 苦手なのは、メンマ（麺のような短冊として散らばる）と丼の縁の正確な形。これは cond と LoRA の学習で直す部分。

---

## 9. Step 4：学習スクリプトの移植（`train_flux2.py`）

### 9.1 方針
- `train.py` をコピーして必要な所だけ書き換えた。構造・引数・ログ・チェックポイントの形式は同じ。
- **学習と推論で同じ関数を使う**ことを最優先にした。テキストの処理（`encode_all_prompt`）と VAE のエンコードを、推論パイプラインと同じものにしている。ずれると、学習した LoRA が推論でうまく効かなくなる。
- dataset、caption の生成、train/val の分割は変更なしで使える。

### 9.2 `train.py` からの主な変更

| 箇所 | FLUX.1（`train.py`） | FLUX.2（`train_flux2.py`） |
|---|---|---|
| VAE エンコード | `sample()` → shift/scale → 16ch | `mode()` → 2×2 パッチ化 → BatchNorm 正規化 → 128ch（推論と同じ） |
| latent のパック | ここで 2×2 にパッチ化 | VAE 側で済んでいるので並べ替えだけ。逆変換の `unpack_latents` を新しく作った |
| 位置 ID | 3 軸 | 4 軸。cond は h, w を cond_scale_factor 倍にする |
| テキスト | T5 + CLIP pooled | 推論と同じ `encode_all_prompt`（pooled は無い） |
| guidance | dev の guidance 埋め込み | なし |
| validation | FLUX.1 のパイプライン | FLUX.2 のパイプライン（Base 版なので CFG 4.0、50 ステップ） |
| LoRA の保存 | `FluxRegionalPipeline.save_lora_weights` | `Flux2RegionalPipeline.save_lora_weights`（形式は同じ） |
| cond を間引かない設定 | cond の長さが 0 のままでマスクが壊れる（FLUX.1 版の潜在バグ。間引きを有効にしているので表に出ていなかった） | 正しく長さを入れるよう修正 |

損失の計算とタイムステップのサンプリングは `train.py` のまま。

### 9.3 設定の違い（`seg2any_ramen.yaml` → `seg2any_ramen_flux2.yaml`）

| 設定 | FLUX.1 | FLUX.2 | 理由 |
|---|---|---|---|
| ベースモデル | FLUX.1-dev | **FLUX.2-klein-base-4B** | 学習には蒸留していない Base 版を使う |
| 出力先 | `ckpt/ramen` | `ckpt/ramen_flux2` | 分けておく |
| `hard_attn_block_range` | [19, 37] | null（→ [5, 14]） | 同じ比率に自動で換算 |
| default LoRA の対象 | ブロック内の全 Linear | ブロック内の全 Linear **＋ 変調の 3 つ** | FLUX.1 ではブロック内の全 Linear に変調も含まれていた。FLUX.2 では変調がブロックの外にあるので、足して同じ範囲にした |
| cond LoRA の対象 | FLUX.1 の名前 | §7.6 の対応表どおり | 名前が変わったため |
| テキストの位置 ID | – | 振り直し | §8.3 |
| validation の CFG / ステップ数 | 3.5 / 32 | 4.0 / 50 | Base 版の既定値 |
| 勾配チェックポイント | あり | **なし**（gp42 向け。§10.2） | 速度のため。48 GB 級の GPU や相乗りが心配なときはコマンドラインでオンにする |
| 一度にモデルを読み込むランク数 | 2 | 4 | モデルが小さいので増やした |

解像度 512、rank 16、学習率 1e-4、batch 2/GPU、3000 ステップ、`attention_mask_method: base`、250 ステップごとの保存と validation は FLUX.1 と同じ。

LoRA のパラメータ数は **合計 45M**（全体 3.92B の 1.15%）。

### 9.4 検証

**`flux2_step4_test.py`（4 項目すべて PASS）**

| # | 確かめたこと | 結果 |
|---|---|---|
| 1 | 学習で保存した LoRA を、推論側の読み込み方法で読むと全テンソルが一致 | 352 テンソル、差 0 |
| 2 | LoRA を読むと出力が変わり、cond を外すとさらに変わる | OK |
| 3 | cond トークンと位置 ID が、学習と推論で一致 | 差 0 |
| 4 | テキストの入力（埋め込み・長さ・位置 ID）が、学習（バッチ 2）と推論（バッチ 1）で一致 | 一致 |

> テスト 4 は最初 FAIL したが、原因はテスト側のバグ（バッチ 1 でも長さのリストが入れ子になっているのを見落としていた）。修正後に PASS。

**短い学習での確認**
- 2 GPU × 20 ステップ（保存と validation あり）→ 完走
- チェックポイントからの再開 → OK
- 8 GPU × 60 ステップ → 1.21 秒/ステップ、GPU あたり 18.7 GB

**想定どおりの挙動**：cond LoRA のうち、最後の single ブロックの出力側（`single_transformer_blocks.19.attn.to_out`）だけは学習されずゼロのまま残る。最後のブロックの cond 出力は捨てられて損失に届かないため。FLUX.1 版でも同じ。

---

## 10. Step 5（前半）：本学習

### 10.1 経過

学習は、他のユーザーのジョブとの GPU の取り合いで何度か中断し、チェックポイントから再開しながら進めた。**最終的に 3000 ステップまで完了**（`checkpoint-3000`、2026-10-06 23:05）。

| ステップ | マシン / GPU | 全体バッチ | 勾配チェックポイント | 速度 | 出来事 |
|---|---|---|---|---|---|
| 0 → 750 | gp42 × 8 | 16 | なし | 約 0.87 秒/ステップ | 750 の直後に GPU 0 で OOM（他ユーザーのジョブが 66 GB で載ってきた） |
| 750 → 1250 | gp42 × 7（GPU 1〜7） | **14** | なし | 約 0.83 秒 | 1250 の直後に、同じユーザーの次のジョブがまたこちらの rank 0 の GPU に載り OOM |
| 1250 → 2000 | **gp40**（A6000 49 GB）× 8 | 16 | あり | 約 2.5 秒 | ユーザーの指示で 2000 で停止 |
| 2000 → 3000 | gp42 × 8 | 16 | あり | 1.1〜2.0 秒 | 途中から GPU 0 に他ユーザーが相乗り（合計 85 GB で両方載った）。その間は遅くなったが**完走** |

- **FLUX.1 のレシピとの違いは、750〜1250 ステップの全体バッチが 14 だった点だけ**。それ以外は同じ。
- ログ：`ckpt/ramen_flux2_train_crash750.log`、`ckpt/ramen_flux2_train_crash1250.log`、`ckpt/ramen_flux2_train_stop2000.log`、`ckpt/ramen_flux2_train.log`（2000→3000）
- プロセス数を変えて再開すると、一部のランクの乱数状態ファイルが無いことがある。accelerate は警告を出して読み飛ばすだけなので問題ない。

### 10.2 GPU の使い方の検討（gp42、96 GB × 8）
最初の設定では GPU を 18.7 GB しか使っていなかったので、短い実行で比べた。

| batch/GPU | 勾配チェックポイント | VRAM/GPU | 秒/ステップ | 画像/秒（8 GPU） |
|---|---|---|---|---|
| 2 | あり | 18.7 GB | 1.21 | 13.2 |
| **2** | **なし** | **42 GB** | **0.96** | **16.7（最速）** |
| 4 | なし | 87 GB | 2.13 | 15.0 |
| 8 | あり | 45 GB | 4.20 | 15.2 |
| 16 | あり | 52 GB | 9.39 | 13.6 |
| 8 | なし | メモリ不足 | – | – |

- **VRAM を埋めても速くならない**。GPU はすでに計算律速で、バッチを大きくしても 1 ステップが伸びるだけになる。
- 効くのは勾配チェックポイントを切ること（再計算が不要になる）で、+26%。
- 元画像は約 600×450 なので、1024 で学習しても拡大するだけで情報は増えない。解像度は 512 のまま。
- **ただし共用サーバでは注意**：勾配チェックポイントなしだと、validation 中の rank 0 は約 62 GB を使う。他人の 66 GB のジョブが来ると OOM になるので、**相乗りが起きそうなときはオン（18.7 GB）の方が安全**。実際、後半はオンにしたことで相乗りされても落ちなかった。

### 10.3 学習途中の validation 画像
`result_f2/flux2_val/step{250,1000,1500,2500,3000}.jpg`（各行：元写真 | マスクと caption | 生成画像。Base 版、CFG 4、50 ステップ。2000 ステップ目は validation の直前に止めたので無い）

| ステップ | 様子 |
|---|---|
| 250 | すでに丼の輪郭・位置、レンゲ・海苔・ナルトの位置、スープの色（辛味・味噌・豚骨）がマスクと caption どおり。ただし卵が 2 個に分かれるなど、細部は甘い |
| 1000 | 卵がマスクどおり 1 個に。メンマもマスクの位置に出るようになり、チャーシュー 2 枚もそれぞれの位置に出る |
| 1500 | 卵の形（丸ごと／半割）が caption どおりになり、チャーシューの形もマスクに近づいた |
| 2500・3000 | 1500 とほぼ同じ水準で安定。左下の小さなメンマなど、細かい領域も出るようになった |

全期間を通じて残っている傾向：**マスクにも caption にも無いネギが描かれる**。UEC-Ramen555 にはネギのクラスが無いのに、写真の大半にネギが写っているため（データセットの偏り）。

---

## 11. Step 5（後半）：FLUX.1 版との比較

### 11.1 比較の条件
- 入力：`examples_Ramen` の 10 例（すべて学習に使っていない val の 55 枚から選んだもの）。seed は各 json のもの。
- 並べたもの（`result_f2/flux2_checkpoint-3000/panel.jpg`）：

| 列 | 内容 | 生成条件 |
|---|---|---|
| 1 | マスク | – |
| 2 | 元写真 | UEC-Ramen555 の実写 |
| 3 | FLUX.1-dev + LoRA（checkpoint-3000） | 既存の出力 `result/ramen_3000` をそのまま使った（今回は生成し直していない） |
| 4 | FLUX.2 klein-base-4B + LoRA | CFG 4.0、50 ステップ、hard マスク |
| 5 | FLUX.2 klein-4B（蒸留版）+ **同じ LoRA** | CFG なし、4 ステップ、hard マスク |

- 生成時間（gp42、512²、1 GPU。最初の 1 枚を除いた平均）：**Base 版 12.6 秒/枚、蒸留版 0.85 秒/枚**（`timing.json`）。FLUX.1 の時間は今回計測していない。

### 11.2 結果
**3 つとも、具材の配置はマスクにかなり忠実**。海苔・卵・チャーシュー・ナルト・メンマ・レンゲの位置と大きさは、どのモデルもおおむねマスクどおり。

| 観点 | FLUX.1 | FLUX.2 Base（50 ステップ） | FLUX.2 蒸留版（4 ステップ） |
|---|---|---|---|
| マスクへの追従 | 良い | 良い（FLUX.1 と同程度） | 良い（Base とほぼ同じ配置） |
| 写真らしさ | 色が濃く、ややくっきりしすぎる場面がある | 丼の縁の模様や器の形など、データセットの写真の雰囲気に近い | Base よりややあっさり。照明や質感はやや単調 |
| 気になる点 | 例によってはネギが大きな領域を占める | マスクに無いネギが出る（データセットの偏り） | **もやしのような白い細い具が多くの例で混ざる**（マスクにも caption にも無い）。蒸留版にそのまま載せたことによるずれと思われる |

- **Base 版で学習した LoRA は、蒸留版（4 ステップ）でもそのまま効いた**。蒸留版用に学習し直さなくても、配置の制御はほぼ保たれる。速度は Base 版の約 15 倍。品質を少し譲って速さを取りたい用途（デモなど）に向いている。
- スープの色や器の見た目は、モデルごとにかなり違う。どれが caption に忠実かは、目視だけでは判断しきれない。定量評価（§14）で確かめたい。

### 11.3 Base 版の推論ステップ数を変えた比較（50 / 40 / 30 / 20 / 10）
結果は `result_f2/flux2_steps_sweep/` にまとめた（作成は `flux2_steps_sweep.py`）。

| 置き場所 | 中身 |
|---|---|
| `steps{50,40,30,20,10}/ramen_XXXX.jpg` | 各ステップ数の生成画像（10 例ずつ） |
| `panel.jpg` | 一覧（マスク / 元写真 / 50 / 40 / 30 / 20 / 10 / 参考：蒸留版 4 ステップ） |
| `zoom_50_20_10.jpg` | 2 例を 512² で並べた拡大（50 / 20 / 10） |
| `timing.csv` / `timing.json` | 画像ごとの生成時間と集計 |
| `README.md` | 条件と結果のまとめ |

条件は §11.1 と同じ（同じ LoRA、同じ seed、512²、CFG 4.0、hard マスク、gp42 の GPU 1 枚）。時間はテキストエンコードから VAE デコードまでを含む 1 枚あたりの時間で、最初にウォームアップを 1 枚生成して計測から除いた。

| ステップ数 | 平均（秒/枚） | 50 ステップ比 | 見た目 |
|---|---|---|---|
| 50 | 11.71 | 1.00× | 基準 |
| 40 | 9.38 | 0.80× | 50 と見分けがつかない |
| 30 | 7.25 | 0.62× | 50 と見分けがつかない |
| **20** | **5.01** | **0.43×** | **50 とほとんど見分けがつかない** |
| 10 | 2.60 | 0.22× | 配置は同じ。チャーシューの焼き目などの質感が薄れ、コントラストもやや下がる |
| （参考）蒸留版 4 | 0.85 | 0.07× | 配置は保つが、もやしのような白い具が混ざる |

- **構図・配置・色は、ステップ数を減らしてもほぼ変わらない**（同じ seed なら、具材の位置と形は同じ）。
- 生成時間はステップ数にほぼ比例する。1 ステップ約 0.23 秒（CFG ありなので transformer を 2 回通す）、ステップ数に関係しない固定の時間は約 0.3 秒。
- **蒸留版の白い具は、Base 版では 10 ステップでも出ない**。つまりステップ数が少ないせいではなく、蒸留版のモデルそのものに由来する。
- **目安**：品質を保つなら **Base 版 20 ステップ（5.0 秒）**、速さ優先なら Base 版 10 ステップ（2.6 秒）、さらに速くしたいなら蒸留版 4 ステップ（0.85 秒、白い具が混ざる）。
- 同じ 50 ステップでも、§11.1 の計測（12.6 秒）と今回（11.7 秒）では少し差がある。別の実行で、計測の仕方（ウォームアップの有無）も違うため。


### 11.4 境界への忠実さと CFG（`result_f2/flux2_cfg_sweep/`）
「FLUX.1 版より境界に忠実でない」という指摘を受け、CFG の強さを変えて調べた（作成は `flux2_cfg_sweep.py`、詳細は同じディレクトリの `README.md`）。条件は 10 例、20 ステップ、hard マスク、gp13（RTX 6000 Ada）。

**境界の指標**：マスクの境界画素のうち、生成画像の Canny エッジが 3px 以内にある割合（recall）。指標が効くことは、元写真とそのマスクで 0.59、別の写真のマスクと組み合わせると 0.25 になることで確かめた。

| 条件 | recall ↑ | 秒/枚 |
|---|---|---|
| FLUX.1-dev + LoRA（既存出力） | **0.843** | – |
| FLUX.2 cfg1.0（CFG なし） | 0.484 | 3.5 |
| FLUX.2 cfg2.0 | 0.530 | 7.2 |
| FLUX.2 cfg4.0（**これまでの設定**） | 0.646 | 7.4 |
| FLUX.2 cfg2.0、ネガティブ側から cond を外す | 0.688 | 6.3 |
| **FLUX.2 cfg4.0、ネガティブ側から cond を外す** | **0.838** | 6.3 |

- **指摘どおり、これまでの設定は FLUX.1 より境界の一致度が低かった**（0.646 対 0.843）。
- 主な原因は **CFG の組み方**だった。ネガティブ側にも同じ cond（輪郭）を入れていたので、「ポジティブ − ネガティブ」の差に輪郭の効果が入らない。その結果、CFG でテキストだけが 4 倍に強調され、輪郭は等倍のままだった。
- **ネガティブ側から cond を外すと、FLUX.1 と同じ水準（0.838）に戻る**。見た目でも、丼の外周や具の輪郭がマスクに沿い、輪郭線をそのまま描くような破綻は無い（`zoom_boundary_overlay.jpg`）。ネガティブ側の計算が軽くなる分、少し速くもなる。
- §5 で挙げた構造の違い（cond 用の変調がモデル全体で 1 つしかない、など）の影響は、この結果を見る限り小さい。
- この結果を受けて、val 55 枚で確かめたうえで（§11.5）、**2026-10-07 にパイプラインの既定を `negative_use_cond=False` に変更した**。昔の挙動に戻すには `negative_use_cond=True`、または `infer_flux2.py --negative_with_cond`。
- **注意：§11.1〜11.3 の画像と時間、学習中の validation 画像は、変更前（ネガティブ側にも cond あり）の設定で作ったもの**。今のコードで作り直すと、境界がよりマスクに沿った画像になる。
- 注意：この指標は「境目がマスクの位置にあるか」しか見ていない。画質や caption への忠実さは別に確かめる必要がある。サンプルも 10 例と少ない。


### 11.5 val 55 枚での最終評価（`result_f2/val55/`）
§11.4 の結果が 10 例だけでなく val 全体でも成り立つか、画質を落としていないかを確かめた（生成は `result_f2/run_val55_gen.sh`、評価は `flux2_eval_val55.py`、詳細は `result_f2/val55/README.md`）。

- 入力：学習に使っていない val 55 枚すべて（`examples_Ramen_val55`。caption は学習時と同じ文言）。512²、gp13（RTX 6000 Ada）
- **FLUX.1 も同じ入力で生成し直した**（`infer.py` の既定：32 ステップ、guidance 3.5、cond_scale_factor 1）。§11.1 のように以前の出力を流用していないので、条件がそろっている。
- FLUX.2 Base は 20 ステップ・CFG 4.0（§11.3 で 50 ステップとほぼ同じ品質だったため）

| 条件 | 境界 recall ↑ | CLIP-T ↑ | CLIP-I ↑ | KID×1000 ↓ | 秒/枚 |
|---|---|---|---|---|---|
| FLUX.1 | **0.808** | 28.0 | 85.3 | 18.8 | 17.8（ロード込み） |
| FLUX.2 Base、旧既定（ネガティブ側に cond あり） | 0.563 | 27.9 | 88.9 | 17.4 | 7.2 |
| **FLUX.2 Base、新既定（ネガティブ側に cond なし）** | **0.772** | 28.1 | **89.1** | 17.9 | 6.3 |
| FLUX.2 蒸留版 4 ステップ | 0.592 | 28.4 | 85.5 | 44.4 | 3.8（ほぼロード時間） |

- 指標：境界 recall はマスク境界と画像エッジの一致度（§11.4）。CLIP-T は global caption と画像の CLIP 類似度、CLIP-I は元写真と画像の CLIP 類似度、KID は val の元写真 55 枚との分布の距離（写真らしさ）。
- **新既定は旧既定より境界が大きく改善**した（+0.209、95% 信頼区間 [+0.186, +0.232]、p = 6e-24）。
- **新既定と FLUX.1 の差は −0.035**（95% 信頼区間 [−0.073, +0.002]、p = 0.07）で、**統計的に有意な差はない**。55 枚中 23 枚では FLUX.1 を上回る。
- **caption への忠実さ（CLIP-T）と写真らしさ（KID）は、新既定でも変わらない**（どの組み合わせでも有意差なし）。
- **元写真への近さ（CLIP-I）は、FLUX.2 の方が FLUX.1 より高い**（+3.8、p < 1e-5）。FLUX.1 はスープの色が濃くなりがちな例が目に付く。
- 蒸留版は境界も写真らしさも劣る。
- → **既定を `negative_use_cond=False` に変更した**（§11.4）。変更後のコードで評価時と同じ画像が再現されること（画素の差 0）も確かめた。

---

## 12. 判断したことのまとめ

| 判断 | 理由 |
|---|---|
| FLUX.1 版を書き換えず、新しいファイルとして作る | FLUX.1 版は動いているので壊さない。比較もできる |
| 学習は Base 版、推論は Base 版と蒸留版の両方で試す | 4 ステップに蒸留したモデルに直接 LoRA を学習させると崩れやすい。結果的に、Base 版で学習した LoRA は蒸留版でも効いた |
| caption をチャットテンプレートで包む | klein の事前学習と同じ入力の形にする |
| テキストの位置 ID は caption ごとに 0 から振り直す | 事前学習の範囲に収まり、並び順にも左右されない（§8.3） |
| default LoRA に変調の Linear も含める | FLUX.1 の「ブロック内の全 Linear」と同じ範囲にする |
| cond の VAE エンコードを `mode()` にし、学習もそろえる | 決定的で、klein 純正と同じ |
| 学習のハイパーパラメータは FLUX.1 と同じにする | 公平に比べるため |
| 解像度は 512 のまま | 元画像が約 600 px なので、上げても情報が増えない |
| gp42 では勾配チェックポイントなし、共用で相乗りが心配なときはあり | 速度と安全性のトレードオフ（§10.2） |
| CFG のネガティブ側には cond（輪郭）を入れない（2026-10-07 から既定） | 入れると輪郭の効果が CFG の差分から消え、境界が甘くなる。外すと val 55 枚で FLUX.1 並みになり、画質は落ちない（§11.4〜11.5） |
| Base 版の推論は 20 ステップを目安にする | 50 ステップとほぼ同じ品質で、時間は 43%（§11.3） |

---

## 13. ハマりどころ・注意

- **venv を混ぜない**：FLUX.1 は `.venv`、FLUX.2 は `.venv-flux2`。diffusers のバージョンが違う。
- **`HF_HOME` を ssd3 に**：space0 から読むとロードがとても遅い。
- **他人と GPU を共有している**：途中から他人のジョブが同じ GPU に載ってくることがある。長い学習は、勾配チェックポイントをオンにしてメモリに余裕を残すのが安全。GPU を絞るときは `CUDA_VISIBLE_DEVICES` と `NUM_PROCESSES` を両方指定する。
- **`nvidia-smi` の使用率表示は当てにならないことがある**：gp42 で「使用率 100%・メモリ 1 MiB・プロセスなし」と出ていたが、行列積で測ると 8 枚とも普通に空いていた（約 290 TFLOPS）。迷ったら実測する。
- **`pgrep -f` は自分自身にも一致する**：`pgrep -f train_flux2.py` をシェルの中で使うと、そのコマンド行自体に一致して「動いている」と誤判定することがある。
- **bf16 で数値を比べるときは相対誤差で**：Qwen3 の出力は値が大きいので、絶対誤差だと大きく見える。
- **マスクの色は完全一致が必要**：FLUX.1 版と同じく、マスクのリサイズは NEAREST で行う。色がずれた領域は黙って無視される。

---

## 14. これからやること

- [x] UEC-Ramen555 で LoRA を 3000 ステップ学習する
- [x] FLUX.1 / FLUX.2 Base / FLUX.2 蒸留版を同じ seed で並べて比べる（§11）
- [x] Base 版で学習した LoRA が蒸留版でも効くか確かめる → 効く（§11.2）
- [x] Base 版の推論ステップ数を 50 / 40 / 30 / 20 / 10 で比べる → 20 ステップでほぼ同じ品質、時間は 43%（§11.3）
- [x] 境界への忠実さが FLUX.1 より低い原因を調べる → CFG のネガティブ側に cond を入れていたため。外すと FLUX.1 並み（§11.4）
- [x] `negative_use_cond=False` を既定にするか決める → val 55 枚で確かめて既定に変更（§11.5）
- [x] **定量評価**：val 55 枚で、境界・CLIP-T・CLIP-I・KID を FLUX.1 と比べる（§11.5）
- [x] FLUX.1 の推論時間を同じ GPU で測る（§11.5。17.8 秒/枚、ロード込み）
- [ ] セグメンテーションモデルを使った、より厳密なマスク追従の評価（領域ごとの mIoU など）。今の境界指標は「境目の位置」しか見ていない
- [ ] 蒸留版で混ざる「もやしのような白い具」の対策（蒸留版向けに短く追加学習する、ネガティブプロンプトは蒸留版では効かないので別の方法を考える、など）
- [ ] マスクに無いネギへの対策（caption で明示する、など）
- [ ] 不要になった動作確認用のチェックポイント（`ckpt/ramen_flux2_smoke*`）を消す

> 注: ここに出てくる段階確認スクリプトは整理のため削除済み。`git show 691c47a:<ファイル名>` で取り出せる。
