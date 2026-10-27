"""自己蒸留用に、教師 (今の蒸留版 + ramen LoRA、全結合 attention) の 4 ステップの軌跡を train 500 枚 × K seed 分保存する。
デモと同じ推論設定 (4 ステップ、512²、hard マスク) で pipe() を回し、transformer への入力と出力をそのまま記録するので、
学習時の生徒の入力は推論時と完全に同じになる。
1 枚 1 ファイル: <out>/<image_name>.pt
  static: encoder_hidden_states, txt_ids, cond_hidden_states, cond_ids, img_ids, 2 つのマスク (許可を bool で packbits)
  traj  : seed ごとに 4 ステップ分の (hidden_states, timestep, target_timestep, 教師の出力)
使い方: python precompute_teacher.py <shard> <num_shards>  (GPU 1 枚に 1 shard)"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/Seg2Any"))
from omegaconf import OmegaConf
from dataset.ramen_dataset import RamenDataset
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline

SHARD, NSHARD = int(sys.argv[1]), int(sys.argv[2])
OUT = "/export/ssd2/lan-y/seg2any_data/ramen_selfdistill/teacher_steps4"
BASE = "black-forest-labs/flux-3-image-klein"
LORA = "/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000"
STEPS, SEEDS = 4, (0, 1, 2, 3)
os.makedirs(OUT, exist_ok=True)

cfg = OmegaConf.load(os.path.expanduser("~/Seg2Any/config/seg2any_ramen_flux3.yaml"))
params = dict(cfg.data.train.params)
params["split_file"] = os.path.expanduser("~/Seg2Any/data/ramen/train.txt")
ds = RamenDataset(**params)

tr = Flux3RegionalTransformer2DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = Flux3RegionalPipeline.from_pretrained(BASE, transformer=tr, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(LORA, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(LORA, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
pipe = pipe.to("cuda"); pipe.set_progress_bar_config(disable=True)
tr = pipe.transformer
orig_forward = tr.forward

rec = []
def grab(*a, **k):
    out = orig_forward(*a, **k)
    rec.append((k, out[0]))
    return out
tr.forward = grab


def pack_mask(m):
    """加算マスク (0 / -inf) を許可の bool にして packbits。全 head 共通なので 1 枚だけ。"""
    if m is None:
        return None
    m = m[0, 0].float()
    assert bool(((m == 0) | torch.isneginf(m)).all()), "0 / -inf 以外の値がある"
    return {"shape": tuple(m.shape), "bits": np.packbits((m == 0).cpu().numpy())}


cpu = lambda x: x.detach().to("cpu") if torch.is_tensor(x) else x
for idx in range(SHARD, len(ds), NSHARD):
    item = ds.data[idx]
    path = os.path.join(OUT, item["image"].replace(".png", ".pt"))
    if os.path.exists(path):
        continue
    b = ds[idx]
    rec_all, static = [], None
    for seed in SEEDS:
        rec.clear()
        pipe(global_prompt=b["global_caption"], regional_prompts=b["regional_captions"], regional_labels=b["label"],
             cond=(b["cond_pixel_values"][None] + 1) / 2.0, attention_mask_method="hard", is_filter_cond_token=True,
             height=512, width=512, cond_scale_factor=1, num_inference_steps=STEPS,
             generator=[torch.Generator("cuda").manual_seed(seed)],
             max_sequence_length=512, regional_max_sequence_length=50)
        k0 = rec[0][0]
        if static is None:
            j = k0["joint_attention_kwargs"]
            static = {key: cpu(k0[key]) for key in ("encoder_hidden_states", "txt_ids", "cond_hidden_states",
                                                     "cond_ids", "img_ids", "hard_attn_block_range")}
            static["attention_mask"] = pack_mask(j.get("attention_mask"))
            static["hard_attention_mask"] = pack_mask(j.get("hard_attention_mask"))
        rec_all.append({"seed": seed, "steps": [
            {"hidden_states": cpu(k["hidden_states"]), "timestep": cpu(k["timestep"]),
             "target_timestep": cpu(k["target_timestep"]), "out": cpu(o)} for k, o in rec]})
    torch.save({"static": static, "traj": rec_all, "image": item["image"]}, path + ".tmp")
    os.replace(path + ".tmp", path)
    print("saved", idx, item["image"], flush=True)
