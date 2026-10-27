"""mIoU 評価用のラーメン用セグメンテーションモデルを作る。

SegFormer-B4 (ADE20K 事前学習) を UEC-Ramen555 の train 500 枚 (data/ramen/train.txt) で fine-tune する。
画像とマスクは生成と同じく 512² に変形 (縦横比は潰す)。11 クラス (0 = 背景も実クラス)。
val 55 枚の実写での mIoU を最後に出す (= この評価器で測れる上限の目安)。

    cd ~/Seg2Any/"[10-21]exp_layout_cfg" && HF_HOME=/export/ssd3/lan-y/huggingface \
        ../.venv-flux3/bin/python seg_train.py
出力: /export/ssd3/lan-y/seg2any_ckpt/ramen_segformer_b4/ (save_pretrained 形式)
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from seg_common import NUM_CLASSES, RAMEN_ROOT, SEG_ROOT, confusion, miou_from_conf, normalize

p = argparse.ArgumentParser()
p.add_argument("--base", default="nvidia/segformer-b4-finetuned-ade-512-512")
p.add_argument("--out", default="/export/ssd3/lan-y/seg2any_ckpt/ramen_segformer_b4")
p.add_argument("--iters", type=int, default=6000)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=6e-5)
args = p.parse_args()
torch.manual_seed(0), random.seed(0), np.random.seed(0)


def load_split(name):
    names = [l.strip() for l in open(os.path.join(SEG_ROOT, f"data/ramen/{name}.txt")) if l.strip()]
    ims, ms = [], []
    for n in names:
        ims.append(np.asarray(Image.open(os.path.join(RAMEN_ROOT, "Ramen_images", n)).convert("RGB").resize((512, 512), Image.BICUBIC)))
        ms.append(np.asarray(Image.open(os.path.join(RAMEN_ROOT, "Ramen_masks", n)).resize((512, 512), Image.NEAREST)))
    return np.stack(ims), np.stack(ms).astype(np.int64)


train_x, train_y = load_split("train")
val_x, val_y = load_split("val")
assert train_y.max() < NUM_CLASSES
print("train", train_x.shape, "val", val_x.shape, flush=True)

dev = "cuda"
model = SegformerForSemanticSegmentation.from_pretrained(
    args.base, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(dev)
opt = torch.optim.AdamW([{"params": model.segformer.parameters(), "lr": args.lr},
                         {"params": model.decode_head.parameters(), "lr": args.lr * 10}], weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (1 - i / args.iters) ** 1.0)


def augment(x, y):
    """x: B,3,512,512 (0..1), y: B,512,512。スケール・クロップ・左右反転・色の揺らし (生成画像の色の違いに強くする)"""
    B = x.shape[0]
    xs, ys = [], []
    for b in range(B):
        s = random.uniform(0.75, 1.5)
        sz = int(512 * s)
        xi = F.interpolate(x[b:b + 1], size=(sz, sz), mode="bilinear", align_corners=False)
        yi = F.interpolate(y[b:b + 1, None].float(), size=(sz, sz), mode="nearest")[:, 0].long()
        if sz >= 512:
            i, j = random.randint(0, sz - 512), random.randint(0, sz - 512)
            xi, yi = xi[..., i:i + 512, j:j + 512], yi[..., i:i + 512, j:j + 512]
        else:
            pad = 512 - sz
            xi = F.pad(xi, (0, pad, 0, pad))
            yi = F.pad(yi, (0, pad, 0, pad), value=255)
        if random.random() < 0.5:
            xi, yi = xi.flip(-1), yi.flip(-1)
        xi = xi * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08)  # 明るさ・コントラスト
        gray = xi.mean(1, keepdim=True)
        xi = gray + (xi - gray) * random.uniform(0.7, 1.3)  # 彩度
        xs.append(xi.clamp(0, 1)), ys.append(yi)
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def evaluate(x_np, y_np):
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    for i in range(0, len(x_np), 8):
        x = torch.from_numpy(x_np[i:i + 8]).permute(0, 3, 1, 2).float().div(255).to(dev)
        logits = model(pixel_values=normalize(x)).logits
        pred = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False).argmax(1).cpu().numpy()
        conf += confusion(pred, y_np[i:i + 8])
    model.train()
    return miou_from_conf(conf)


model.train()
for it in range(1, args.iters + 1):
    idx = np.random.randint(0, len(train_x), args.bs)
    x = torch.from_numpy(train_x[idx]).permute(0, 3, 1, 2).float().div(255).to(dev)
    y = torch.from_numpy(train_y[idx]).to(dev)
    x, y = augment(x, y)
    logits = model(pixel_values=normalize(x)).logits
    logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
    loss = F.cross_entropy(logits, y, ignore_index=255)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if it % 200 == 0:
        print(f"iter {it} loss {loss.item():.4f}", flush=True)
    if it % 2000 == 0 or it == args.iters:
        miou, per = evaluate(val_x, val_y)
        print(f"iter {it} val(real) mIoU {miou:.4f}", " ".join(f"{v:.3f}" for v in per), flush=True)

model.save_pretrained(args.out)
print("saved", args.out)
