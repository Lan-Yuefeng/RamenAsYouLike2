"""Step 2 の検証: Flux2RegionalTransformer2DModel が正しく動くか。

    HF_HOME=/export/ssd3/lan-y/huggingface HF_HUB_OFFLINE=1 .venv-flux2/bin/python flux2_step2_test.py

学習済み LoRA がまだ無いので、出力の「意味」ではなく以下の性質で正しさを確かめる。
  1. cond なし・マスクなしなら diffusers 純正 Flux2Transformer2DModel と出力が一致する
  2. cond トークンを完全に隔離するマスクを付けると、cond なしの出力と一致する
  3. 隔離しないマスクなら cond が出力に影響する
  4. hard_attention_mask は hard_attn_block_range のブロックにだけ効く
  5. 'cond' LoRA は noisy img 側に漏れない / 'default' LoRA は効く
  6. 勾配チェックポイント有効で backward が通り、両 LoRA に勾配が流れる
"""
import argparse
import sys
import time

import torch
from peft import LoraConfig

from diffusers import Flux2Transformer2DModel
from src.models.transformer_flux2 import Flux2RegionalTransformer2DModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
args = parser.parse_args()

device = "cuda"
torch.manual_seed(0)
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# ---- 入力 (512x512 相当: latent 32x32 = 1024 トークン) ----
B, H, W, T, C = 2, 32, 32, 64, 128
NUM_COND = 300


def make_inputs(dtype):
    g = torch.Generator().manual_seed(0)
    hs = torch.randn(B, H * W, C, generator=g).to(device, dtype)
    enc = torch.randn(B, T, 7680, generator=g).to(device, dtype)
    cond = torch.randn(B, NUM_COND, C, generator=g).to(device, dtype)
    t = torch.full((B,), 0.7, device=device, dtype=dtype)
    img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(H), torch.arange(W), torch.arange(1)).to(device)
    txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(T)).to(device)
    # cond は img の一部の位置 (バッチごとに異なる) に置く
    cond_ids = torch.stack([img_ids[torch.randperm(H * W, generator=g)[:NUM_COND]] for _ in range(B)])
    return hs, enc, cond, t, img_ids, txt_ids, cond_ids


def make_mask(isolate_cond, heads, dtype):
    S = T + H * W + NUM_COND
    allow = torch.ones(S, S, dtype=torch.bool, device=device)
    if isolate_cond:
        allow[: T + H * W, T + H * W :] = False
        allow[T + H * W :, : T + H * W] = False
    mask = torch.log(allow.float()).to(dtype)
    return mask[None, None].expand(B, heads, -1, -1)


# ================= 1-4: fp32 で厳密に比較 =================
dtype = torch.float32
t0 = time.time()
model = Flux2RegionalTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype).to(device)
print(f"load regional: {time.time() - t0:.1f}s, blocks: {type(model.transformer_blocks[0]).__name__}, "
      f"{type(model.single_transformer_blocks[0]).__name__}, default hard range {model.default_hard_attn_block_range()}")
ref = Flux2Transformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype).to(device)

sd_a, sd_b = model.state_dict(), ref.state_dict()
check("state_dict のキーが純正と同一", sd_a.keys() == sd_b.keys(), f"({len(sd_a)} keys)")
check("重みが純正と同一", all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a))

heads = model.config.num_attention_heads
hs, enc, cond, t, img_ids, txt_ids, cond_ids = make_inputs(dtype)

with torch.no_grad():
    out_ref = ref(hidden_states=hs, encoder_hidden_states=enc, timestep=t, img_ids=img_ids, txt_ids=txt_ids,
                  return_dict=False)[0]
    del ref
    torch.cuda.empty_cache()

    def run(**kw):
        return model(hidden_states=hs, encoder_hidden_states=enc, timestep=t, img_ids=img_ids, txt_ids=txt_ids,
                     return_dict=False, **kw)[0]

    out_nocond = run()
    d = maxdiff(out_nocond, out_ref)
    check("1. cond なし = 純正 Flux2", d < 1e-3, f"max|diff|={d:.2e} (出力 std {out_ref.std().item():.2f})")

    iso = make_mask(True, heads, dtype)
    full = make_mask(False, heads, dtype)

    out_iso = run(cond_hidden_states=cond, cond_ids=cond_ids, joint_attention_kwargs={"attention_mask": iso})
    d = maxdiff(out_iso, out_nocond)
    check("2. cond を隔離するマスク = cond なし", d < 1e-3, f"max|diff|={d:.2e}")

    out_full = run(cond_hidden_states=cond, cond_ids=cond_ids, joint_attention_kwargs={"attention_mask": full})
    d = maxdiff(out_full, out_nocond)
    check("3. 隔離しなければ cond が効く", d > 1e-2, f"max|diff|={d:.2e}")

    n_blocks = model.config.num_layers + model.config.num_single_layers
    out_hard_all = run(cond_hidden_states=cond, cond_ids=cond_ids, hard_attn_block_range=[0, n_blocks - 1],
                       joint_attention_kwargs={"attention_mask": full, "hard_attention_mask": iso})
    out_hard_none = run(cond_hidden_states=cond, cond_ids=cond_ids, hard_attn_block_range=[n_blocks, n_blocks],
                        joint_attention_kwargs={"attention_mask": full, "hard_attention_mask": iso})
    d1, d2 = maxdiff(out_hard_all, out_iso), maxdiff(out_hard_none, out_full)
    check("4. hard マスクは範囲内のブロックにだけ効く", d1 < 1e-3 and d2 < 1e-3, f"全範囲 {d1:.2e} / 範囲外 {d2:.2e}")

    # 7. txt_ids がバッチごとに違う場合 (regional caption ごとに L を 0 から振り直す) は、
    #    サンプル i の出力が「全バッチ txt_ids_i」で回したときのサンプル i と一致する
    ids_a = txt_ids.clone()
    ids_b = txt_ids.clone()
    ids_b[20:, 3] -= 20  # 20 トークン目から L を振り直した構成
    out_mix = model(hidden_states=hs, encoder_hidden_states=enc, timestep=t, img_ids=img_ids,
                    txt_ids=torch.stack([ids_a, ids_b]), return_dict=False)[0]
    out_b = model(hidden_states=hs, encoder_hidden_states=enc, timestep=t, img_ids=img_ids,
                  txt_ids=ids_b, return_dict=False)[0]
    d1, d2 = maxdiff(out_mix[0], out_nocond[0]), maxdiff(out_mix[1], out_b[1])
    d3 = maxdiff(out_b[1], out_nocond[1])
    check("7. バッチごとの txt_ids", d1 < 1e-3 and d2 < 1e-3 and d3 > 1e-3,
          f"sample0 {d1:.2e} / sample1 {d2:.2e} / 振り直しの影響 {d3:.2e}")

# ================= 5-6: LoRA (bf16) =================
del model
torch.cuda.empty_cache()
dtype = torch.bfloat16
model = Flux2RegionalTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype).to(device)
model.requires_grad_(False)
hs, enc, cond, t, img_ids, txt_ids, cond_ids = make_inputs(dtype)
full = make_mask(False, heads, dtype)

# FLUX.1 版 config の対応物:
#   default: DiT ブロック内の全 Linear (all-linear-in-dit-blocks)
#   cond   : x_embedder / modulation / img 側 attn + ff / single の融合 Linear
default_targets = [n for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear) and n.startswith(("transformer_blocks", "single_transformer_blocks"))]
cond_targets = (r"(x_embedder|double_stream_modulation_img\.linear|single_stream_modulation\.linear"
                r"|transformer_blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)"
                r"|transformer_blocks\.\d+\.ff\.(linear_in|linear_out)"
                r"|single_transformer_blocks\.\d+\.attn\.(to_qkv_mlp_proj|to_out))")
model.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=default_targets), adapter_name="default")
model.add_adapter(LoraConfig(r=16, lora_alpha=16, target_modules=cond_targets), adapter_name="cond")
model.set_adapter(["cond", "default"])
n_def = sum(p.numel() for n, p in model.named_parameters() if ".default." in n)
n_cond = sum(p.numel() for n, p in model.named_parameters() if ".cond." in n)
print(f"LoRA params: default {n_def / 1e6:.1f}M, cond {n_cond / 1e6:.1f}M")


def run(**kw):
    return model(hidden_states=hs, encoder_hidden_states=enc, timestep=t, img_ids=img_ids, txt_ids=txt_ids,
                 return_dict=False, **kw)[0]


def randomize(adapter):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if f"lora_B.{adapter}." in n:
                p.normal_(0, 0.02)


with torch.no_grad():
    base = run()
    randomize("cond")
    d = maxdiff(run(), base)
    check("5a. 'cond' LoRA は cond なし出力に影響しない", d == 0, f"max|diff|={d:.2e}")
    randomize("default")
    d = maxdiff(run(), base)
    check("5b. 'default' LoRA は効く", d > 1e-2, f"max|diff|={d:.2e}")

model.train()
model.enable_gradient_checkpointing()
for n, p in model.named_parameters():
    if "lora_" in n:
        p.requires_grad_(True)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
out = run(cond_hidden_states=cond, cond_ids=cond_ids, joint_attention_kwargs={"attention_mask": full})
out.float().pow(2).mean().backward()
torch.cuda.synchronize()
g_def = sum(p.grad.abs().sum().item() for n, p in model.named_parameters() if ".default." in n and p.grad is not None)
g_cond = sum(p.grad.abs().sum().item() for n, p in model.named_parameters() if ".cond." in n and p.grad is not None)
check("6. backward (grad ckpt) で両 LoRA に勾配", g_def > 0 and g_cond > 0, f"|g| default {g_def:.2e}, cond {g_cond:.2e}")
print(f"train step (B={B}, 512², cond {NUM_COND} tok): {time.time() - t0:.2f}s, "
      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
