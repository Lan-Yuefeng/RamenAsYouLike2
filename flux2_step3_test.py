"""Step 3 の検証: Flux2RegionalPipeline の部品が klein 純正と整合しているか。

    HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 .venv-flux2/bin/python flux2_step3_test.py

  1. パディングを除いた Qwen3 埋め込みが、klein 純正 (512 にパディング) の先頭部分と一致する
  2. txt_seq_lens / text_ids が [global, regional...] の構成どおりで、L が区間ごとに 0 から振り直される
  3. cond トークンの間引き数が、輪郭のある 16x16 パッチ数と一致し、位置 id がその座標を指す
  4. 1 枚生成して形とメモリを確認する
"""
import sys
import time

import torch

from infer import load_seg_map_and_prompt
from src.models.transformer_flux2 import Flux2RegionalTransformer2DModel
from src.pipelines.pipeline_flux2 import Flux2RegionalPipeline

MODEL = "black-forest-labs/FLUX.2-klein-4B"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


transformer = Flux2RegionalTransformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux2RegionalPipeline.from_pretrained(MODEL, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")
device = pipe._execution_device

batch = load_seg_map_and_prompt("examples_Ramen/ramen_0007.png", "examples_Ramen/ramen_0007.json", 1)
regional = batch["regional_captions"]
global_caption = batch["global_caption"]

# ---- 1. 埋め込み ----
# bf16 だと系列長 (512 パディング vs 短い系列) でカーネルが変わり相対 1% 程度ずれる。
# Qwen3 の hidden states は先頭トークンで |x|~16000 に達するので、絶対誤差でなく fp32 での相対誤差で比べる
pipe.text_encoder.to(torch.float32)
with torch.no_grad():
    texts = [global_caption] + regional[:2]
    mine = pipe._get_qwen3_prompt_embeds_nopad(texts, 10**6, device)  # 出力は transformer の dtype (bf16) に丸め
    worst = 0.0
    for text, emb in zip(texts, mine):
        ref = pipe._get_qwen3_prompt_embeds(pipe.text_encoder, pipe.tokenizer, [text], dtype=torch.float32,
                                            device=device, max_sequence_length=512)[0, : emb.shape[0]]
        worst = max(worst, ((ref - emb.float()).abs().max() / ref.abs().max()).item())
pipe.text_encoder.to(torch.bfloat16)
check("1. 埋め込み = klein 純正の先頭部分 (fp32 encoder)", worst < 1e-2, f"max 相対誤差 {worst:.2e} (bf16 丸めのみ)")

# ---- 2. txt_seq_lens / text_ids ----
embeds, txt_seq_lens, text_ids = pipe.encode_all_prompt([global_caption], [regional], device=device)
lens = txt_seq_lens[0]
l = text_ids[0, :, 3].tolist()
starts = [sum(lens[:i]) for i in range(len(lens))]
ok = embeds.shape[1] == sum(lens) and all(l[s] == 0 for s in starts) and all(
    l[s + j] == j for s, n in zip(starts, lens) for j in range(n)
) and (text_ids[..., :3] == 0).all().item()
check("2. txt_seq_lens と L の振り直し", ok,
      f"lens={lens} (global + {len(regional)} regions, template 込み), S={embeds.shape[1]}")
_, _, ids_cont = pipe.encode_all_prompt([global_caption], [regional], device=device, regional_text_position="continuous")
check("2b. continuous は通し番号", ids_cont[0, :, 3].tolist() == list(range(sum(lens))), f"max L={max(ids_cont[0, :, 3].tolist())}")

# ---- 3. cond トークン ----
cond = (batch["cond_pixel_values"][None] + 1) / 2
proc = pipe.image_processor.preprocess(cond, height=512, width=512).to(device, pipe.vae.dtype)
n_valid = Flux2RegionalPipeline.get_valid_cond_token_num if hasattr(Flux2RegionalPipeline, "get_valid_cond_token_num") else None
latents = pipe._encode_vae_image(proc, generator=None)
ids = pipe._prepare_latent_ids(latents)[0].to(device)
hs = pipe._pack_latents(latents)
hs_f, ids_f, cond_lens, pad_lens = pipe.filter_cond_token(proc, hs, ids, vae_scale_factor=16)
patch_has_edge = torch.nn.functional.pixel_unshuffle((proc[:, :1] > -1).float(), 16).sum(1).flatten() > 0
expected_hw = torch.nonzero(patch_has_edge.view(32, 32))
ok = cond_lens[0] == int(patch_has_edge.sum()) and torch.equal(ids_f[0, :, 1:3].long().cpu(), expected_hw.cpu())
check("3. cond の間引きと位置 id", ok, f"{cond_lens[0]} / {hs.shape[1]} tokens 残す, latent {tuple(latents.shape)}")

# ---- 4. 生成 ----
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
image = pipe(global_prompt=global_caption, regional_prompts=regional, regional_labels=batch["label"], cond=cond,
             height=512, width=512, num_inference_steps=4,
             generator=torch.Generator("cuda").manual_seed(batch["seed"])).images[0]
dt = time.time() - t0
check("4. 生成 (cond あり, hard mask)", image.size == (512, 512),
      f"{dt:.1f}s (初回), peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
