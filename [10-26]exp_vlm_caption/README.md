# VLM による caption 生成と拡張 (2026-10-26〜)

## 目的
[10-25]exp_ablation で、意味は regional caption に頼り切っていることがわかった。ところが今の caption は
`dataset/ramen_captions.py` の手書きテンプレート（Claude が書いたもの）で、論文の手法として再現できない。
そこで caption を公開 VLM で作れるようにする。最終的には、推論時にユーザーが短い言葉（「塩のスープ」）を入れると、
その意味が画像に出るようにしたい。入力の範囲はデータセットにあるもの（スープ 5 種、10 クラス）に限る。

## 構成（案）
1. **学習用 caption を VLM で付ける**: Qwen3-VL-8B-Instruct に正解画像、領域を強調した画像、クラス名、スープ種を
   見せて、領域ごとの caption と全体の caption を書かせる（`caption_vlm.py`）。正解画像を使うのは学習時だけ。
2. **Seg2Any を VLM caption で学習し直す**: 推論時の caption と分布をそろえるため。
3. **推論時の拡張器**: 入力はマスクと領域ごとの短いプロンプト、出力は 1 と同じ書き方の caption。
   1 の (短いプロンプト → VLM caption) の組で fine-tune する（教師→生徒の蒸留）。
   余力があれば、生成結果を報酬にする強化学習へ進む。
4. **評価**: スープ種の分類精度（「塩」と入れたら塩に見えるか）、mIoU、画質。比較対象は手書きテンプレート。

## 進捗
- [x] 1. train の 10 枚（スープ 5 種 × 2、`trial10.txt`）で caption を試した → `captions_trial/`、比較パネル `compare_trial/`
  - v0（領域を強調表示した画像を渡す）は、背景と器の caption に中身（麺・卵）まで書いた → `*_v0/`
  - v1（領域外を灰色で塗って切り出す＋器と背景に個別の指示）で直った
  - 良い点: 器の色や柄、注釈にない具（ネギ、もやし、魚）、スープの見た目まで画像ごとに書かれる
  - 悪い点: 背景がほぼ見えない画像では「木のテーブル」と推測で書く（0004 など）
- [x] 全 555 枚に付けた → `captions_all/`（`run_all.sh`、GPU 8 枚で約 15 分）。`check_captions.py` で抜け 0・50 トークン超 0
  - 0096 の bowl だけ同じ文字列の繰り返しで崩れた（120 トークン）→ `--repetition_penalty 1.1` でこの 1 枚だけ作り直し（崩れた版は `redo/`）
- [x] v2: スープ領域だけ作り直した（`--only_class soup`）。v1 は塩 36/67・豚骨 66/148 のスープ caption に種類名が無く、
  「salt broth」という言葉と見た目が結びつかない恐れがあった → 「The shio (salt) broth ...」で書き出させて 555/555 に。v1 は `captions_all_v1/`
- [x] 2. 再学習: `run_train_bg.sh`（gp41 8GPU、2026-10-26 12:25〜17:35、3000 step。相乗りで遅くなり 5h10m）
  - 12:09 に v1 caption で始めたものは 20 分で止めた（`aborted/`）
  - config は `seg2any_ramen_flux3_vlmcap.yaml`（手書き版との違いは caption_dir と output_dir だけ）→ space0/lan-y/seg2any_ckpt/ramen_flux3_vlmcap
  - `dataset/ramen_dataset.py` に `caption_dir` を足した（None なら手書きテンプレートのまま）
- [x] 3a. 短いプロンプト: `make_short_prompts.py` で caption の文面だけから Qwen3-VL (テキストのみ) に 3 段階で書かせた → `short_prompts/`
  - name（1〜3 語、例 "chashu pork"）/ short（2〜4 語、特徴 1 つ）/ detail（6 語まで）。スープは種類名を必ず残す（抜けは 3/1665）
  - 学習中の GPU 2 枚に相乗りして約 1 時間半
- [x] 3b. 拡張器: `expander.py`（Qwen3-4B-Instruct-2507 + LoRA r16、train 500 枚 × 3 段階 = 1500 組、2 epoch、自前の SFT ループ）
  - `run_expander.sh`（tmux `expander`、GPU 5 に相乗り）→ space0/lan-y/seg2any_ckpt/expander_qwen3_4b。学習後に val10 で zero-shot / fine-tune × 3 段階を生成 → `expanded/`
- [x] 4. val10 で生成して比べた（下の「結果」）（`make_inputs.py` で入力を作り、`../[10-25]exp_ablation/gen_ablation.py` で生成）
  - 条件: 手書き + 旧 LoRA（exp_ablation の full_hard）/ VLM caption（上限）/ 短いプロンプトそのまま / zero-shot 拡張 / fine-tune 拡張

## スープ判定器（評価用）
`judge_soup.py`: VLM に 5 択で答えさせる。本物の写真 val55 での正解率（判定器自体の上限の目安）:
- Qwen3-VL-8B: **0.545**。豚骨が 1/15 しか当たらない（塩・醤油と判定）
- 原因はデータ側: val の豚骨の多くは茶色〜金色の豚骨醤油系で、白濁は少数。人が見ても見分けにくい
- InternVL3-8B-hf（別系統、ssd2 にキャッシュ）: **0.545**。豚骨は 0/15（塩・醤油・味噌に散る）
- 豚骨を除いた 40 枚では Qwen 0.725 / InternVL 0.75。2 つの判定の一致は 32/55、両方正解は 23/55
- 生成画像の評価では、両判定器の正解率を本物写真の値（上限の目安）と並べて出す。豚骨は判定できないものとして別扱い
- 学習中の GPU に相乗りしても大丈夫（学習は検証中も 23.3GB、判定器は約 17GB）

## 結果 (val10、2026-10-26)
生成は `run_gen_val10.sh`（FLUX.3 Base、hard、50 step、CFG 3.5、seed は exp_ablation と同じ）、評価は `run_eval_val10.sh`。
拡張はすべて short 段階（2〜4 語）の入力から。`results/panel.jpg`（左から mask / real / 各条件）、`results/summary.csv` ほか。

| 条件 | LoRA | regional caption | mIoU | bIoU d3 | egg | cut_egg | menma | spoon | スープ判定 Qwen / Intern |
|---|---|---|---|---|---|---|---|---|---|
| hand_full | 手書き | 手書きテンプレート | **0.941** | 0.442 | 0.95 | 0.96 | 0.85 | 0.94 | 5 / 6 |
| hand_short_raw | 手書き | 短いプロンプトそのまま | 0.883 | 0.421 | 0.95 | 0.96 | 0.50 | 0.75 | 3 / 2 |
| vlm_oracle | VLM | 正解画像からの VLM caption（上限） | 0.932 | **0.444** | 0.95 | 0.95 | 0.86 | 0.95 | 4 / 4 |
| vlm_short_raw | VLM | 短いプロンプトそのまま | 0.814 | 0.348 | 0.42 | 0.75 | 0.62 | 0.81 | 5 / 7 |
| vlm_short_zs | VLM | zero-shot 拡張 | 0.737 | 0.339 | 0.00 | 0.59 | 0.40 | 0.78 | 4 / 4 |
| vlm_short_ft | VLM | fine-tune 拡張 | 0.896 | 0.419 | 0.95 | 0.96 | 0.86 | 0.95 | 5 / 5 |
| real_photo | – | – | 0.861 | – | | | | | 5 / 4 |

スープ判定は 10 枚中の正解数（うち豚骨 3 枚）。

- **fine-tune 拡張は効いている**: 短いプロンプトそのまま (0.814) → 0.896。卵 0.42→0.95、メンマ 0.62→0.86 で、上限の oracle (0.932) にかなり近づく。
  zero-shot は短いプロンプトを整えるだけで伸ばさず、むしろ悪い (0.737)
- naruto は ft で 0.51 だが、val10 に 1 枚しかない（0007）ので誤差の範囲
- **VLM caption は画像ごとの見た目（器の色、背景、具）を伝えられる**: oracle は水色の器・赤いテーブル・ニンジンなど本物の写真に近い。手書き版はどれも同じ器になる
- **手書き LoRA は短いプロンプトに強い**: hand_short_raw 0.883 > vlm_short_raw 0.814。手書きテンプレートは毎回同じ言い回しなので、物の名前と見た目が強く結びついているためと思われる
- **スープ判定は 10 枚では差が見えない**: 本物の写真でも 5/10・4/10 しか当たらず、条件間の差は 1〜2 枚。判定するなら val55 以上が必要
- **fine-tune 拡張の全体 caption が決まり文句に固まる**: 辛味では入力にないキャベツとネギの山が毎回出て、生成画像にも山が現れる（panel 4 行目）

### 蒸留版 (flux-3-image-klein、16 step、CFG なし) — `run_gen_val10_distill.sh` → `results_distill/`
LoRA は Base 版で学習したものをそのまま載せた。評価は同じ（スープ判定はまだ）。

| 条件 | mIoU | bIoU d3 | egg | cut_egg | menma | naruto |
|---|---|---|---|---|---|---|
| hand_full | 0.907 | 0.352 | 0.93 | 0.96 | 0.75 | 0.92 |
| hand_short_raw | 0.815 | 0.291 | 0.63 | 0.79 | 0.48 | 0.90 |
| vlm_oracle | **0.922** | **0.370** | 0.91 | 0.97 | 0.81 | 0.94 |
| vlm_short_raw | 0.744 | 0.310 | 0.00 | 0.57 | 0.48 | 0.92 |
| vlm_short_zs | 0.719 | 0.296 | 0.00 | 0.55 | 0.23 | 0.92 |
| vlm_short_ft | 0.876 | 0.345 | 0.85 | 0.97 | 0.80 | 0.54 |

- 蒸留版では **VLM caption（oracle）が手書きを上回る**（mIoU 0.922 vs 0.907、bIoU 0.370 vs 0.352）
- fine-tune 拡張の効きは Base 版より大きい: 短いまま 0.744 → 0.876 (+0.13)。手書き LoRA に短いまま (0.815) より上
- naruto は Base 版と同じく 0007 の 1 枚で 0.54。ft の caption が「pale yellow outer layer」と書いたためかもしれない

## 結果 (val55、2026-10-27) — Base 版と蒸留版
- 拡張器 v2（`--regions_only`）: 拡張器は領域 caption だけを書き、全体 caption は拡張後の領域 caption を LoRA なしで要約して作る
  （`run_expander_v2.sh` → space0/lan-y/seg2any_ckpt/expander_qwen3_4b_regonly）。v1 の全体 caption が決まり文句に固まる問題への対策
  - 全体 caption は画像ごとに変わるようになった。ただしスープの領域 caption 自体に「shredded cabbage」などの事前知識が残ることはある
- 生成は `gen_one.sh`（Base 50 step CFG 3.5 → 蒸留版 16 step）、`chain_gen55.sh`。評価は `run_eval_val55.sh` → `results_val55/`, `results_distill_val55/`
- 拡張の入力はすべて short 段階（2〜4 語）

| 条件 | mIoU Base | mIoU 蒸留 | bIoU Base | bIoU 蒸留 | egg Base | naruto Base | スープ Qwen Base | スープ Intern Base | スープ Qwen 蒸留 | スープ Intern 蒸留 |
|---|---|---|---|---|---|---|---|---|---|---|
| 本物の写真 | 0.885 | | | | | | 30 | 30 | 30 | 30 |
| hand_full（手書き） | **0.935** | **0.905** | 0.426 | 0.316 | 0.93 | 0.93 | 29 | 33 | 26 | 33 |
| hand_short_raw | 0.794 | 0.799 | 0.360 | 0.279 | 0.46 | 0.65 | 24 | 21 | 23 | 25 |
| vlm_oracle（上限の目安） | 0.910 | **0.905** | 0.420 | **0.325** | 0.93 | 0.78 | 33 | 33 | 33 | 32 |
| vlm_short_raw | 0.729 | 0.733 | 0.339 | 0.278 | 0.06 | 0.69 | 34 | **39** | 29 | 32 |
| vlm_short_zs | 0.762 | 0.731 | 0.355 | 0.272 | 0.00 | 0.91 | 32 | 28 | 28 | 28 |
| vlm_short_ft（v1） | 0.912 | 0.863 | 0.425 | 0.309 | 0.92 | 0.74 | 34 | 37 | 29 | 32 |
| **vlm_short_ftv2** | 0.929 | 0.896 | **0.433** | 0.320 | 0.94 | 0.94 | **36** | 32 | **33** | 32 |

スープ判定は 55 枚中の正解数（豚骨 15・味噌 14・醤油 11・辛味 8・塩 7）。

- **短いプロンプト + 拡張器 v2 で、手書きテンプレートとほぼ同じ配置精度**: mIoU 0.929 vs 0.935（Base）、0.896 vs 0.905（蒸留）。
  画像ごとの対応のある比較で差は有意でない（Base p=0.056、蒸留 p=0.38、Wilcoxon。勝ち数は 36/55・34/55 で ftv2 が多い）
- **v2 は v1 より良い**: 蒸留版で 0.863 → 0.896、naruto 0.74 → 0.94（Base）。全体 caption の決まり文句をやめた効果
- **スープ判定（「塩と入れたら塩に見えるか」）**: 拡張器 v2 は本物の写真（30/55）と同等以上。手書き LoRA に短いプロンプトを入れると落ちる
  - ftv2 vs hand_short_raw（対応のある McNemar）: Base Qwen +12/-0 p<0.001、Intern +13/-2 p=0.007、蒸留 Qwen +13/-3 p=0.021、Intern +11/-4 p=0.12
  - ftv2 vs hand_full: Qwen は +9/-2（p=0.065 で両モデルとも）、Intern は差なし
  - vlm_short_raw は配置が崩れる（卵 0.06）のに判定は高め。「creamy tonkotsu」だけだと典型的な白濁スープになり、判定器が当てやすいため。豚骨も 6/15 当たる。スープ判定だけでは良し悪しを決められない
- 本物の写真の判定が 30/55 なので、判定器の分解能はこのあたりで頭打ち。豚骨はどの条件でもほぼ当たらない（ラベルと見た目のずれ）
