"""Step 3 の検証: Flux3RegionalPipeline が FLUX.3 純正 (ModularPipeline) と整合しているか。

    HF_HOME=/export/ssd3/lan-y/huggingface .venv-flux3/bin/python flux3_step3_test.py

  1. caption ごとの Qwen3-VL 埋め込みが、純正の Flux3TextEncoderStep と一致する (長さが同じ caption をまとめて通しても)
  2. txt_seq_lens が 80 の倍数で、text_ids の L が区間ごとに 0 から振り直される
  3. cond トークンの間引き数が輪郭のある 16x16 パッチ数と一致し、位置 id が (0, 0.5h, 0.5w, -1) を指す
  4. regional caption を完全に隔離すれば、純正パイプラインと同じ画像になる (蒸留版 16 ステップ / Base 版 CFG)
  5. ラーメンの例で 1 枚生成して、形・時間・メモリを確認する (LoRA なし、cond なし、hard mask)
"""
import sys
import time

import numpy as np
import torch

from diffusers import ModularPipeline
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
from infer import load_seg_map_and_prompt
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline

BASE = "black-forest-labs/flux-3-image-klein-base"
DISTILLED = "black-forest-labs/flux-3-image-klein"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def psnr(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return 10 * np.log10(255.0**2 / max(((a - b) ** 2).mean(), 1e-12))


def load(repo):
    transformer = Flux3RegionalTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.bfloat16)
    return Flux3RegionalPipeline.from_pretrained(repo, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")


pipe = load(DISTILLED)
device = pipe._execution_device

batch = load_seg_map_and_prompt("examples_Ramen/ramen_0007.png", "examples_Ramen/ramen_0007.json", 1)
regional = batch["regional_captions"]
global_caption = batch["global_caption"]

# ---- 1. 埋め込み ----
with torch.no_grad():
    texts = [global_caption] + regional
    mine = pipe._get_qwen3_vl_prompt_embeds(texts, 10**6, device)
    worst, lens = 0.0, []
    for text, emb in zip(texts, mine):
        ref = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(pipe.text_encoder, pipe.tokenizer, text, device=device)[0]
        lens.append(emb.shape[0])
        if ref.shape != emb.shape:
            worst = float("inf")
            break
        worst = max(worst, ((ref.float() - emb.float()).abs().max() / ref.float().abs().max()).item())
check("1. 埋め込み = 純正 Flux3TextEncoderStep", worst < 1e-2, f"max 相対誤差 {worst:.2e}, 長さ {sorted(set(lens))}")

# ---- 2. txt_seq_lens / text_ids ----
embeds, txt_seq_lens, text_ids = pipe.encode_all_prompt([global_caption], [regional], device=device)
lens = txt_seq_lens[0]
l = text_ids[0, :, 3].long().tolist()
starts = [sum(lens[:i]) for i in range(len(lens))]
ok = (embeds.shape[1] == sum(lens) and all(n % 80 == 0 for n in lens)
      and all(l[s + j] == j for s, n in zip(starts, lens) for j in range(n))
      and (text_ids[..., :3] == 0).all().item())
check("2. caption ごとに 80 の倍数、L の振り直し", ok, f"lens={lens} (global + {len(regional)} regions), S={embeds.shape[1]}")

# ---- 3. cond トークン ----
cond = (batch["cond_pixel_values"][None] + 1) / 2
proc = pipe.image_processor.preprocess(cond, height=512, width=512).to(device, pipe.vae.dtype)
latents = pipe.encode_vae_image(proc)
from src.pipelines.pipeline_flux3 import prepare_image_ids  # noqa: E402

ids = prepare_image_ids(32, 32, l_coord=-1).to(device)
hs_f, ids_f, cond_lens, pad_lens = pipe.filter_cond_token(proc, pipe.pack_latents(latents), ids, vae_scale_factor=16)
patch_has_edge = torch.nn.functional.pixel_unshuffle((proc[:, :1] > -1).float(), 16).sum(1).flatten() > 0
expected_hw = torch.nonzero(patch_has_edge.view(32, 32)).float() * 0.5
ok = (cond_lens[0] == int(patch_has_edge.sum()) and torch.equal(ids_f[0, :, 1:3].cpu(), expected_hw.cpu())
      and (ids_f[0, :, 3] == -1).all().item())
check("3. cond の間引きと位置 id", ok, f"{cond_lens[0]} / {32 * 32} tokens 残す, latent {tuple(latents.shape)}")


# ---- 4. 純正との一致 ----
def compare_official(pipe, repo, steps, guidance):
    official = ModularPipeline.from_pretrained(repo)
    # 重複して読まないよう、こちらの部品をそのまま渡す (transformer は純正クラスと重みが同一)
    official.update_components(text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, vae=pipe.vae)
    official.load_components(names=["transformer"], dtype=torch.bfloat16)
    official.to("cuda")
    if guidance is not None:
        official.guider = official.guider.new(guidance_scale=guidance)
    seed, size = 1234, 512
    ref = official(prompt=global_caption, height=size, width=size, num_inference_steps=steps,
                   generator=torch.Generator("cuda").manual_seed(seed), output="images")[0]
    del official
    torch.cuda.empty_cache()

    # regional caption 1 つを、どの画素にも対応させず (label 全 0)、base マスクで完全に隔離する
    empty_label = torch.zeros(1, size // 16, size // 16)
    mine = pipe(global_prompt=global_caption, regional_prompts=["isolated dummy region"], regional_labels=empty_label,
                attention_mask_method="base", height=size, width=size, num_inference_steps=steps,
                guidance_scale=guidance or 3.5,
                generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    return psnr(ref, mine), ref, mine


p, ref_img, my_img = compare_official(pipe, DISTILLED, 16, None)
check("4a. 隔離した regional caption なら純正と同じ画像 (蒸留版 16 ステップ)", p > 30, f"PSNR {p:.1f} dB")
ref_img.save("result_f3/flux3_step3_official_distilled.png")
my_img.save("result_f3/flux3_step3_mine_distilled.png")

# ---- 5. ラーメンの例 (蒸留版) ----
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
image = pipe(global_prompt=global_caption, regional_prompts=regional, regional_labels=batch["label"],
             height=batch["image_height"], width=batch["image_width"],
             generator=torch.Generator("cuda").manual_seed(batch["seed"])).images[0]
dt = time.time() - t0
image.save("result_f3/flux3_step3_ramen_distilled.png")
check("5a. ラーメンの例 (蒸留版, cond なし, hard mask)", image.size == (batch["image_width"], batch["image_height"]),
      f"{image.size}, {dt:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")

del pipe
torch.cuda.empty_cache()
pipe = load(BASE)

p, ref_img, my_img = compare_official(pipe, BASE, 50, 3.5)
check("4b. 隔離した regional caption なら純正と同じ画像 (Base 版 50 ステップ, CFG 3.5)", p > 30, f"PSNR {p:.1f} dB")
ref_img.save("result_f3/flux3_step3_official_base.png")
my_img.save("result_f3/flux3_step3_mine_base.png")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
image = pipe(global_prompt=global_caption, regional_prompts=regional, regional_labels=batch["label"],
             height=batch["image_height"], width=batch["image_width"],
             generator=torch.Generator("cuda").manual_seed(batch["seed"])).images[0]
dt = time.time() - t0
image.save("result_f3/flux3_step3_ramen_base.png")
check("5b. ラーメンの例 (Base 版 50 ステップ, CFG 3.5)", image.size == (batch["image_width"], batch["image_height"]),
      f"{dt:.1f}s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
