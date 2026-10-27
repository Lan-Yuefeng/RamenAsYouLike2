"""③ (txt / cond を一方向化して K/V を使い回す) の自己蒸留。
教師 = 今の蒸留版 + ramen LoRA (全結合 attention)。軌跡と出力は precompute_teacher.py で保存済み。
生徒 = 同じ重みから始めて、CachedForward(use_cache=False, cache_txt=True) で一方向化 + txt / cond の modulation 固定。
推論時のキャッシュあり計算と完全に一致する (check_cache.py) ので、学習はキャッシュなしで全ステップを独立に扱う。
損失は教師の状態 x_t での出力 (flow map の予測) の MSE。default / cond の両 LoRA を学習する。
起動: torchrun --nproc_per_node N train_selfdistill.py --out ...  (run_train.sh 参照)"""
import argparse, glob, json, os, random, sys, time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/Seg2Any"))
sys.path.insert(0, os.path.expanduser("~/Seg2Any/[10-26]exp_speed"))
from peft.utils import get_peft_model_state_dict
from src.models.transformer_flux3 import Flux3RegionalTransformer2DModel
from src.pipelines.pipeline_flux3 import Flux3RegionalPipeline
from cache_flux3 import CachedForward

p = argparse.ArgumentParser()
p.add_argument("--data", default="/export/ssd2/lan-y/seg2any_data/ramen_selfdistill/teacher_steps4")
p.add_argument("--out", default="/export/space0/lan-y/seg2any_ckpt/ramen_flux3_selfdistill")
p.add_argument("--init_lora", default="/export/ssd3/lan-y/seg2any_ckpt/ramen_flux3/checkpoint-3000")
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--max_steps", type=int, default=2000)
p.add_argument("--ckpt_every", type=int, default=500)
p.add_argument("--log_every", type=int, default=10)
p.add_argument("--max_grad_norm", type=float, default=1.0)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dev = torch.device("cuda")
main = rank == 0

BASE = "black-forest-labs/flux-3-image-klein"
tr = Flux3RegionalTransformer2DModel.from_pretrained(BASE, subfolder="transformer", torch_dtype=torch.bfloat16)
# LoRA の読み込みだけに pipeline を使う (テキストエンコーダと VAE は読まない)
pipe = Flux3RegionalPipeline.from_pretrained(BASE, transformer=tr, text_encoder=None, tokenizer=None, vae=None,
                                             torch_dtype=torch.bfloat16)
pipe.load_lora_weights(os.path.join(args.init_lora, "default"), adapter_name="default")
pipe.load_lora_weights(os.path.join(args.init_lora, "cond"), adapter_name="cond")
pipe.set_adapters(["cond", "default"])
tr = pipe.transformer.to(dev)
tr.requires_grad_(False)
params = []
for n, prm in tr.named_parameters():
    if "lora_" in n:
        prm.data = prm.data.float()  # 学習する LoRA は fp32 (train_flux3.py の cast_training_params と同じ)
        prm.requires_grad_(True)
        params.append(prm)
if main:
    print(f"trainable {sum(x.numel() for x in params) / 1e6:.1f}M params, world {world}", flush=True)
opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
student = CachedForward(tr, use_cache=False, cache_txt=True, ckpt=True)


def unpack_mask(m):
    if m is None:
        return None
    n = m["shape"][0] * m["shape"][1]
    allowed = torch.from_numpy(np.unpackbits(m["bits"])[:n].reshape(m["shape"]).astype(bool))
    add = torch.zeros(m["shape"], dtype=torch.bfloat16)
    add[~allowed] = float("-inf")
    return add[None, None]


class Points(torch.utils.data.Dataset):
    """(ファイル, seed, ステップ) の組を 1 点とする。"""

    def __init__(self, files):
        self.files = files
        self.n_seed, self.n_step = 4, 4

    def __len__(self):
        return len(self.files) * self.n_seed * self.n_step

    def __getitem__(self, i):
        f, r = divmod(i, self.n_seed * self.n_step)
        s, k = divmod(r, self.n_step)
        d = torch.load(self.files[f], weights_only=False)
        st, step = d["static"], d["traj"][s]["steps"][k]
        return {"static": {**st, "attention_mask": unpack_mask(st["attention_mask"]),
                           "hard_attention_mask": unpack_mask(st["hard_attention_mask"])},
                "step": step, "k": k}


files = sorted(glob.glob(os.path.join(args.data, "*.pt")))
ds = Points(files)
sampler = torch.utils.data.DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
dl = torch.utils.data.DataLoader(ds, batch_size=None, sampler=sampler, num_workers=3, prefetch_factor=2,
                                 persistent_workers=True)
if main:
    print(f"{len(files)} files, {len(ds)} points", flush=True)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)


def save(step):
    if not main:
        return
    d = os.path.join(args.out, f"checkpoint-{step}")
    for name in ("default", "cond"):
        sd = {k: v.to(torch.bfloat16) for k, v in get_peft_model_state_dict(tr, adapter_name=name).items()}
        Flux3RegionalPipeline.save_lora_weights(os.path.join(d, name), transformer_lora_layers=sd)
    print("saved", d, flush=True)


step, epoch = 0, 0
log_loss, log_k, t0 = [], [], time.time()
while step < args.max_steps:
    sampler.set_epoch(epoch)
    for b in dl:
        st, sp = b["static"], b["step"]
        to = lambda x: x.to(dev) if torch.is_tensor(x) else x
        kw = dict(hidden_states=to(sp["hidden_states"]), encoder_hidden_states=to(st["encoder_hidden_states"]),
                  timestep=to(sp["timestep"]), img_ids=to(st["img_ids"]), txt_ids=to(st["txt_ids"]),
                  cond_hidden_states=to(st["cond_hidden_states"]), cond_ids=to(st["cond_ids"]),
                  target_timestep=to(sp["target_timestep"]), hard_attn_block_range=st["hard_attn_block_range"],
                  joint_attention_kwargs={"attention_mask": to(st["attention_mask"]),
                                          "hard_attention_mask": to(st["hard_attention_mask"])})
        student.reset()
        pred = student(**kw)[0]  # 推論と同じく autocast なし (base は bf16、LoRA は fp32 で計算して bf16 に戻る)
        loss = F.mse_loss(pred.float(), sp["out"].to(dev).float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # 手で all-reduce (生徒は nn.Module ではないので DDP を使わない)
        grads = [x.grad if x.grad is not None else torch.zeros_like(x) for x in params]
        flat = torch.cat([g.flatten() for g in grads])
        dist.all_reduce(flat)
        flat /= world
        off = 0
        for x, g in zip(params, grads):
            x.grad = flat[off:off + g.numel()].view_as(g); off += g.numel()
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        step += 1

        lt = torch.tensor([loss.item(), float(b["k"])], device=dev)
        allv = [torch.zeros_like(lt) for _ in range(world)]
        dist.all_gather(allv, lt)
        log_loss += [v[0].item() for v in allv]; log_k += [int(v[1].item()) for v in allv]
        if main and step % args.log_every == 0:
            per_k = {k: float(np.mean([l for l, kk in zip(log_loss, log_k) if kk == k]))
                     for k in sorted(set(log_k))}
            print(f"step {step} loss {np.mean(log_loss):.5f} by_step {json.dumps({k: round(v, 5) for k, v in per_k.items()})} "
                  f"{(time.time() - t0) / args.log_every:.2f}s/it mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G",
                  flush=True)
            log_loss, log_k, t0 = [], [], time.time()
        if step % args.ckpt_every == 0 or step == args.max_steps:
            save(step)
        if step >= args.max_steps:
            break
    epoch += 1
dist.destroy_process_group()
