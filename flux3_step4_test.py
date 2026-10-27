"""Step 4 の検証: train_flux3.py が保存した LoRA を推論側で正しく読めるか、学習と推論で入力が一致するか。

    HF_HOME=/export/ssd3/lan-y/huggingface .venv-flux3/bin/python flux3_step4_test.py \
        --ckpt /export/ssd3/lan-y/seg2any_ckpt/ramen_flux3_smoke/checkpoint-20

  1. 保存した default / cond LoRA を Flux3RegionalPipeline.load_lora_weights で読むと、全テンソルが保存値と一致する
  2. LoRA を読むと出力が変わり、set_adapters で cond を外すと出力が変わる
  3. 学習側 (encode_images + pack + ids + filter) と推論側 (pipeline 内部と同じ手順) の cond トークン・位置 id が一致する
  4. 学習側と推論側のテキスト入力 (embeds / txt_seq_lens / text_ids) が一致する
  5. 学習と同じ読み込み方 (torch_dtype=bf16 で読んで .to(device)) で QK norm が fp32 のまま残る
  6. Base 版で学習した LoRA が蒸留版 (flow map) にそのまま載り、生成できる
"""
import argparse
import os
import sys

import torch
from safetensors.torch import load_file

from dataset.collate_fn import collate_fn
from dataset.ramen_dataset import RamenDataset
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline, prepare_image_ids
from train_flux3 import encode_images

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="black-forest-labs/flux-3-image-klein-base")
parser.add_argument("--distilled", default="black-forest-labs/flux-3-image-klein")
parser.add_argument("--ckpt", default="/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3_smoke/checkpoint-20")
args = parser.parse_args()

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def load_pipe(repo):
    transformer = Flux3RegionalTransformer2DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = Flux3RegionalPipeline.from_pretrained(repo, transformer=transformer, torch_dtype=torch.bfloat16).to("cuda")
    pipe.load_lora_weights(os.path.join(args.ckpt, "default"), adapter_name="default")
    pipe.load_lora_weights(os.path.join(args.ckpt, "cond"), adapter_name="cond")
    pipe.set_adapters(["cond", "default"])
    return pipe


pipe = load_pipe(args.model)
device = pipe._execution_device

# ---- 5. QK norm の dtype ----
norms = [m.weight.dtype for n, m in pipe.transformer.named_modules() if n.endswith(("norm_q", "norm_k"))]
check("5. QK norm が fp32 のまま (bf16 で読んで .to(device))", all(d == torch.float32 for d in norms),
      f"({len(norms)} 個, {set(norms)})")

# ---- 1. LoRA の読み込み ----
params = dict(pipe.transformer.named_parameters())
worst, n, missing = 0.0, 0, []
for adapter in ["default", "cond"]:
    saved = load_file(os.path.join(args.ckpt, adapter, "pytorch_lora_weights.safetensors"))
    for k, v in saved.items():
        name = k.removeprefix("transformer.").replace(".lora_A.weight", f".lora_A.{adapter}.weight").replace(
            ".lora_B.weight", f".lora_B.{adapter}.weight")
        if name not in params:
            missing.append(name)
            continue
        worst = max(worst, (params[name].float().cpu() - v.float()).abs().max().item())
        n += 1
check("1. 保存した LoRA が全テンソル一致で読める", worst == 0 and not missing,
      f"({n} tensors, max|diff|={worst:.1e}, 読めないキー {len(missing)})")

# ---- 3/4 の準備: val の 2 サンプル ----
ds = RamenDataset("/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555", "./data/ramen/val.txt", resolution=512)
batch = collate_fn([ds[0], ds[1]])

# ---- 3. cond トークン ----
with torch.no_grad():
    # 学習側 (train_flux3.py と同じ。VAE は fp32)
    vae32 = pipe.vae.to(torch.float32)
    lat = encode_images(batch["cond_pixel_values"].to(device), vae32, torch.bfloat16)
    tr_hs = Flux3RegionalPipeline.pack_latents(lat)
    tr_ids = prepare_image_ids(*lat.shape[-2:], l_coord=-1, pos_scale=1).to(device)
    tr_hs, tr_ids, tr_lens, tr_pads = Flux3RegionalPipeline.filter_cond_token(
        batch["cond_pixel_values"].to(device), tr_hs, tr_ids, vae_scale_factor=16)
    # 推論側 (Flux3RegionalPipeline.__call__ と同じ。infer_flux3.py は (cond+1)/2 を渡す)
    cond = pipe.image_processor.preprocess((batch["cond_pixel_values"] + 1) / 2, height=512, width=512)
    cond = cond.to(device, pipe.vae.dtype)
    lat2 = pipe.encode_vae_image(cond)
    in_ids = prepare_image_ids(*lat2.shape[-2:], l_coord=-1, pos_scale=1).to(device)
    in_hs, in_ids, in_lens, in_pads = pipe.filter_cond_token(cond, pipe.pack_latents(lat2).to(torch.bfloat16), in_ids, 16)
    pipe.vae.to(torch.bfloat16)
d = (tr_hs.float() - in_hs.float()).abs().max().item()
check("3. cond トークン: 学習 = 推論", tr_lens == in_lens and torch.equal(tr_ids, in_ids) and d < 1e-2,
      f"lens {tr_lens} / {in_lens}, max|diff|={d:.1e}")

# ---- 4. テキスト ----
with torch.no_grad():
    tr = pipe.encode_all_prompt(batch["global_caption"], batch["regional_captions"], 512, 50, 1, device, "restart")
    inf = [pipe.encode_all_prompt([g], [r], 512, 50, 1, device, "restart")
           for g, r in zip(batch["global_caption"], batch["regional_captions"])]
ok = True
for i, (e, lens, ids) in enumerate(inf):
    L = e.shape[1]
    # 学習はバッチ内で長さを揃えるため末尾に pad 区間が付く。実トークン部分を比べる
    lens = lens[0]
    ok &= tr[1][i][: len(lens)] == lens
    ok &= torch.equal(tr[2][i, :L], ids[0])
    ok &= (tr[0][i, :L].float() - e[0].float()).abs().max().item() / e.float().abs().max().item() < 1e-2
check("4. テキスト入力: 学習 (B=2) = 推論 (B=1)", bool(ok), f"txt_seq_lens {tr[1]}")

# ---- 2. LoRA の効き ----
lab, rp, g = batch["label"][0], batch["regional_captions"][0], batch["global_caption"][0]
kw = dict(global_prompt=g, regional_prompts=rp, regional_labels=lab, cond=(batch["cond_pixel_values"][:1] + 1) / 2,
          height=512, width=512, num_inference_steps=2, output_type="latent", attention_mask_method="base")
with torch.no_grad():
    out_lora = pipe(**kw, generator=torch.Generator("cuda").manual_seed(0)).images
    pipe.set_adapters(["default"])
    out_nocond = pipe(**kw, generator=torch.Generator("cuda").manual_seed(0)).images
    pipe.disable_lora()
    out_base = pipe(**kw, generator=torch.Generator("cuda").manual_seed(0)).images
d1 = (out_lora.float() - out_base.float()).abs().max().item()
d2 = (out_lora.float() - out_nocond.float()).abs().max().item()
check("2. LoRA が効く (default+cond vs なし / cond を外す)", d1 > 0 and d2 > 0, f"{d1:.2e} / {d2:.2e}")

# ---- 6. 蒸留版に載せる ----
del pipe
torch.cuda.empty_cache()
pipe = load_pipe(args.distilled)
kw.update(num_inference_steps=None, output_type="pil")
with torch.no_grad():
    image = pipe(**kw, generator=torch.Generator("cuda").manual_seed(0)).images[0]
image.save("result_f3/flux3_step4_distilled_smoke.png")
check("6. Base 版で学習した LoRA が蒸留版に載る", pipe.is_distilled and image.size == (512, 512),
      "-> result_f3/flux3_step4_distilled_smoke.png")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
