"""semcond の実装テスト (学習前に回す)。

    cd ~/Seg2Any/"[10-25]exp_semcond" && HF_HOME=/export/ssd3/lan-y/huggingface ../.venv-flux3/bin/python test_semcond.py

1. caption 部分のトークン範囲が正しい (デコードすると caption に戻る)
2. 意味ベクトル：各トークンの面積比の和が 1、輪郭上のトークンは複数領域を混ぜる、領域の内側の画素だけのトークンは 1 領域
3. ゼロ初期化なら出力は semcond なしと完全に一致する (学習開始時は元の Seg2Any と同じ)
4. W を乱数にすると出力が変わる (フックが効いている)。勾配チェックポイントありの backward で W に勾配が流れる
5. W の保存 / 読み込みで値が一致する
6. 推論用の semcond_for_pipeline_call が、パイプライン内部の cond トークン数と一致する (1 枚生成して確認)
"""
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from infer import load_seg_map_and_prompt  # noqa: E402
from semcond import (cond_semantic, enable_semcond, load_sem_proj, region_pooled_embeds, save_sem_proj,  # noqa: E402
                     semcond_for_pipeline_call, _template_prefix_suffix)
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel  # noqa: E402
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline, prepare_image_ids  # noqa: E402

MODEL = "black-forest-labs/flux-3-image-klein-base"
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
EX = os.path.join(os.path.dirname(HERE), "examples/examples_Ramen_val55/ramen_0061")
results = []


def check(name, ok, info=""):
    results.append(ok)
    print(("PASS" if ok else "FAIL"), name, info, flush=True)


tr = Flux3RegionalTransformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained(MODEL, transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda")
dev = "cuda"

b = load_seg_map_and_prompt(seg_map_path=EX + ".png", seg_anno_path=EX + ".json", cond_scale_factor=1)
H, W = b["image_height"], b["image_width"]
rp = b["regional_captions"]

# ---- 1
pe, lens, tids = pipe.encode_all_prompt([b["global_caption"]], [rp], device=dev)
pooled = region_pooled_embeds(pipe, pe, lens, [rp])
pre, suf = _template_prefix_suffix(pipe)
tok = pipe.tokenizer.tokenizer
ok = True
for r, p in enumerate(rp):
    ids = tok(pipe._wrap_chat_template(p, 50)).input_ids
    span = tok.decode(ids[pre:len(ids) - suf])
    ok &= span == p
check("1 caption のトークン範囲", ok and pooled[0].shape == (len(rp), pe.shape[-1]), f"pre={pre} suf={suf}, pooled {tuple(pooled[0].shape)}")

# ---- 2
cond01 = (b["cond_pixel_values"][None] + 1) / 2.0
sem = semcond_for_pipeline_call(pipe, b["global_caption"], rp, b["label"], cond01, H, W)
c = pipe.image_processor.preprocess(cond01, height=H, width=W).to(dev)
th, tw = H // 16, W // 16
ids = prepare_image_ids(th, tw, l_coord=-1).to(dev)
_, cids, clens, _ = pipe.filter_cond_token(c, torch.zeros(1, th * tw, 1, device=dev), ids, vae_scale_factor=16)
frac = torch.nn.functional.adaptive_avg_pool2d(b["label"].float()[None].to(dev), (th, tw))[0]
rows, cols = (cids[0, :clens[0], 1] / 0.5).round().long(), (cids[0, :clens[0], 2] / 0.5).round().long()
a = frac[:, rows, cols].T
n_mixed = int(((a > 0).sum(-1) >= 2).sum())
recon = (a / a.sum(-1, keepdim=True)) @ pooled[0]
check("2 意味ベクトル", torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
      and torch.allclose(recon, sem[0, :clens[0]], atol=1e-3) and n_mixed > 0,
      f"cond {clens[0]} トークン中 {n_mixed} が複数領域を混ぜる")

# ---- 3 / 4：transformer を直接呼ぶ
torch.manual_seed(0)
lat = torch.randn(1, th * tw, 128, device=dev, dtype=torch.bfloat16)
cond_lat = pipe.pack_latents(pipe.encode_vae_image(c.to(pipe.vae.dtype))).to(torch.bfloat16)
cond_lat, cids2, clens2, plens2 = pipe.filter_cond_token(c, cond_lat, ids, vae_scale_factor=16)
am, ham = pipe.prepare_attention_mask(
    attention_mask_method="base", regional_labels=[b["label"].to(dev)], txt_seq_lens=lens, cond_seq_lens=clens2,
    pad_seq_lens=plens2, height=th, width=tw, num_attention_heads=tr.config.num_attention_heads,
    dtype=pe.dtype, device=dev)


def run(grad=False):
    with torch.set_grad_enabled(grad):
        return tr(hidden_states=lat, encoder_hidden_states=pe, timestep=torch.tensor([0.7], device=dev, dtype=torch.bfloat16),
                  img_ids=prepare_image_ids(th, tw).to(dev), txt_ids=tids, cond_hidden_states=cond_lat, cond_ids=cids2,
                  joint_attention_kwargs={"attention_mask": am, "hard_attention_mask": ham}, return_dict=False)[0]


ref = run()
enable_semcond(tr)
tr._cond_sem = sem
out0 = run()
check("3 ゼロ初期化なら元と一致", torch.equal(ref, out0), f"max diff {(ref - out0).abs().max().item():.2e}")

torch.nn.init.normal_(tr.cond_sem_proj.weight, std=1e-3)
out1 = run()
d = (out1.float() - ref.float()).abs().mean().item()
tr.enable_gradient_checkpointing()
tr.train()
loss = run(grad=True).float().pow(2).mean()
loss.backward()
g = tr.cond_sem_proj.weight.grad
check("4 フックが効く / 勾配が流れる (勾配チェックポイントあり)", d > 0 and g is not None and g.abs().sum() > 0,
      f"出力差 {d:.3e}, |grad| {g.abs().sum().item() if g is not None else None:.3e}")
tr.disable_gradient_checkpointing()
tr.eval()
tr.cond_sem_proj.weight.grad = None

# ---- 5
with tempfile.TemporaryDirectory() as td:
    w_before = tr.cond_sem_proj.weight.detach().clone()
    save_sem_proj(tr, td)
    torch.nn.init.zeros_(tr.cond_sem_proj.weight)
    load_sem_proj(tr, td)
    check("5 保存 / 読み込み", torch.equal(w_before, tr.cond_sem_proj.weight.detach()))

# ---- 6：パイプラインで 1 枚 (内部の cond トークン数がフックの assert で確かめられる)
tr._cond_sem = sem
try:
    img = pipe(global_prompt=b["global_caption"], regional_prompts=rp, regional_labels=b["label"], cond=cond01,
               height=H, width=W, num_inference_steps=4, guidance_scale=3.5,
               generator=torch.Generator(dev).manual_seed(0)).images[0]
    check("6 推論用の意味ベクトルがパイプライン内部の cond と一致", True, f"{img.size}")
except AssertionError as e:
    check("6 推論用の意味ベクトルがパイプライン内部の cond と一致", False, str(e))

print(f"\n{sum(results)}/{len(results)} PASS")
