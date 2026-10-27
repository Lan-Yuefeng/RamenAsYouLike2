# Seg2Any の部品を外す実験（exp_ablation）

2026-10-25。Seg2Any のどの部品が効いていて、どこに伸びしろがあるかをデータで確かめる（提案手法を考えるための「既存手法の分析」）。

## 条件
- FLUX.3 Image Klein Base + ramen LoRA（`/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000`。学習時のマスクは `base`）
- 512²、50 ステップ、CFG 3.5（ネガティブ側に cond なし）、val55 の先頭 10 枚、seed は各 json のもの。gp41（A40）
- **推論時に外しただけで、学習し直してはいない**。LoRA は全部品ありで学習しているので、外した条件は学習時と分布がずれる（「その部品に頼っている度合い」を測っている）
- 評価は `../[10-21]exp_layout_cfg/` の `eval.py`（境界 recall、mIoU、CLIP）と `boundary_iou.py`（ラーメン用 SegFormer で分割して比べる）

## ファイル
| ファイル | 役割 |
|---|---|
| `make_inputs.py` | regional caption を具材名だけ／空にした入力を作る（`inputs_val10_classname`、`inputs_val10_noregional`） |
| `gen_ablation.py` | 部品を外して生成（`--mask_method`、`--hard_range`、`--no_cond`、`--no_lora`、`--inputs`） |
| `run_ablation.sh` | 8 条件を GPU 8 枚で同時に生成（1 条件 10 枚で約 8 分） |
| `results/` | 生成画像と `summary.csv`・`per_class_iou.csv`・`boundary_iou.csv`・`paired.csv`・`panel.jpg` |

## 結果（10 枚）
| 条件 | 外したもの | mIoU | Boundary IoU (d=3) | 境界 recall | CLIP-T | CLIP-I |
|---|---|---|---|---|---|---|
| **full_hard**（推論の既定） | なし | **0.941** | **0.442** | 0.631 | 27.09 | 89.07 |
| mask_base | AIA（属性の隔離マスク） | 0.940 | 0.440 | 0.663 | 26.99 | **89.92** |
| hard_all_blocks | （AIA を全ブロックに広げる） | 0.628 | 0.138 | 0.275 | 26.45 | 83.60 |
| no_cond | 輪郭 cond | 0.370 | 0.059 | 0.184 | 27.19 | 88.13 |
| mask_base_no_cond | AIA と輪郭 cond（SAA だけ残る） | 0.371 | 0.063 | 0.244 | 27.22 | 89.56 |
| no_lora | 学習（LoRA と cond） | 0.208 | 0.048 | 0.258 | 31.09 | 82.12 |
| cap_classname | regional caption の説明文（具材名だけにする） | 0.820 | 0.382 | 0.713 | 26.53 | 87.60 |
| cap_empty | regional caption（空にする） | 0.669 | 0.329 | 0.738 | 26.28 | 85.39 |
| 実写（参考） | – | 0.861 | 0.328 | 0.615 | 26.70 | 100 |

クラスごとの IoU（抜粋。メンマ / 肉 / スプーン）：full_hard 0.85 / 0.91 / 0.94、cap_classname **0.12** / 0.73 / 0.70、cap_empty **0.00** / 0.46 / 0.30

## わかったこと
1. **形（どこに何かがあるか）は、ほぼ輪郭 cond だけが担っている。** cond を外すと mIoU が 0.94 → 0.37、Boundary IoU が 0.44 → 0.06 に落ちる。領域アテンションマスク（SAA）だけでは、丼の大きさもスプーンの位置も守られない（パネル参照）
2. **AIA（属性の隔離マスク）は、このデータでは効いていない。** 外しても mIoU・Boundary IoU は同じで、CLIP-I はむしろ +0.85。そもそも学習は AIA なし（base）なので、推論の既定 hard は学習とずれている。全ブロックに広げると壊れる（二重に写る）
3. **「何があるか（具材の種類）」は regional caption の説明文に頼っている。** 輪郭 cond はクラスを区別しない（どの領域も同じ線）ので、caption を具材名だけにするとメンマの IoU が 0.85 → 0.12 になる（肉のような別の具になる）。caption を空にすると、形は残るが中身がでたらめになる（mIoU 0.67）
4. 境界 recall はまた逆の動きをする（caption を空にすると 0.738 で最高）。recall はエッジの量に左右されるので、主指標には使えない

## 提案手法への示唆
- Seg2Any は「形 = クラスを区別しない輪郭 cond」「意味 = 領域ごとの caption とアテンションマスク」と役割を分けている。**意味の側が弱点**で、短い caption や見慣れない具材ではすぐ崩れる。ここに伸びしろがある
- AIA が要らないなら、hard マスクの計算や「どのブロックに掛けるか」の調整は削れる（単純化と速度）
- 次の候補：輪郭 cond にクラスの意味を持たせる（領域ごとの caption 埋め込みを cond トークンに足すなど。open-vocabulary のまま意味を cond 側でも伝える）→ 学習が必要なので、まず ramen で 1 回学習して、この表の cap_classname / cap_empty の落ち込みが減るかで確かめる
