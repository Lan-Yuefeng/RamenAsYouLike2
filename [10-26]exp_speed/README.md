# FLUX3 蒸留版の高速化 (exp_speed)

## 学習なしの ① LoRA 融合 と ③ txt / cond の K/V キャッシュ (2026-10-27, gp36 RTX 4090)

- 実装: `cache_flux3.py` (src/ は触らずモンキーパッチ)。実行: `run_cache.py`、検算: `check_cache.py`、パネル: `make_panel_cache.py` → `panel_cache_val10.jpg`
- ① `fuse_default_lora`: default LoRA を base の重みに足し込む。default 側の呼び出しは peft を無効化、cond 側は default の scaling を -s にして差し引く
- ③ `CachedForward`: joint (single) blocks で txt / cond 行 → img 列を遮断し、txt / cond の modulation を t=0・飛び先 0 に固定。1 ステップ目に各 single block の txt / cond の K/V を貯め、2 ステップ目以降は img だけ計算
- 設定: 蒸留版 + ramen LoRA (checkpoint-3000)、4 ステップ、512²、val10

| 条件 | 1 枚 (s) | transformer 合計 (s) | 1 回目 (ms) | 2 回目以降 (ms) |
|---|---|---|---|---|
| base | 1.22 | 0.95 | 247 | 234 |
| 一方向化のみ (キャッシュなし) | 1.16 | 0.91 | 232 | 227 |
| ③ キャッシュ | 0.80 | 0.54 | 234 | 103 |
| ① 融合 | 0.98 | 0.72 | 184 | 177 |
| ① + ③ | **0.69** | 0.43 | 196 | 76 |

検算 (`check_cache.py`、同じ入力で transformer 1 回の出力を比較):
- 自前の一方向化 (キャッシュなし) と probe 方式 (元 forward + マスク書換え + フック) は完全一致 (相対誤差 0)
- キャッシュあり / なし: 融合後は全ステップ完全一致、未融合は 1e-2 程度 (bf16 の行列形状による丸めの差)。画像で 38.7 dB
- ① 融合の前後: 1 回あたり相対誤差 1〜5e-2、画像で 37 dB。見た目は区別できない
- probe の jpg (gp40 A6000) とは 17.7 dB。base も A6000 と 4090 で 25.7 dB しか一致しないので GPU 差。一方向化したモデルは分布外で差が増幅される

見た目: ① は base と同じ。①+③ は probe の oneway_both_fixtemb と同じ崩れ方 (器が大きく寄る、具が増える、配置が mask からずれる、器が 2 つになる例あり)。
→ 速度は 1.22 → 0.69 秒 (4090) で 1 秒以下を達成するが、③ は品質を自己蒸留で取り戻すのが前提。① は今すぐデモに入れてよい (-0.25 秒)

## 学習なしの ② cond だけ K/V キャッシュ (OminiControl2 方式, 2026-10-27, gp36 RTX 4090)

- `CachedForward(cache_txt=False)`: cond 行 → img 列を遮断 (cond はもともと txt を見ないので、これで cond は自分だけに依る)、cond の modulation を t=0・飛び先 0 に固定、cond の K/V を 1 ステップ目に貯めて使い回す。txt と img は毎ステップ計算
- 実行: `run_cache2.py` → `cache2_timing_4090.json`、パネル: `make_panel_cache2.py` → `panel_cache2_val10.jpg` (列 = mask / base / ① / ①+② / ①+③)
- 検算: `check_cache.py` で probe 方式と完全一致、キャッシュあり / なしも完全一致 (画像で 148 dB)

| 条件 | 1 枚 (s) | transformer 合計 (s) | 1 回目 (ms) | 2 回目以降 (ms) |
|---|---|---|---|---|
| base | 1.25 | 0.99 | 254 | 244 |
| ② キャッシュ | 0.96 | 0.71 | 235 | 157 |
| ① 融合 | 0.98 | 0.72 | 186 | 177 |
| ① + ② | **0.80** | 0.55 | 182 | 121 |

見た目: ② は base と PSNR 13 dB。具の量は base に近い (③ のようにごちゃつかない) が、器の輪郭・形が崩れる (器が潰れる、ガラス器や網目模様になる、配置がずれる)。probe の oneway_cond と同じで、cond が画像を見られないと輪郭の効きが落ちる。
→ ② も学習なしでは使えない。①+② で 0.80 s、①+③ で 0.69 s
