"""seg_train.py と eval.py で共通に使う、mIoU 評価まわりの定義。"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SEG_ROOT = os.path.dirname(HERE)  # ~/Seg2Any
sys.path.insert(0, SEG_ROOT)
from dataset.ramen_captions import CLASS_COLORS, CLASS_ID_TO_NAME  # noqa: E402

RAMEN_ROOT = "/export/space0/lan-y/UEC-Ramen555/UEC-Ramen555"
NUM_CLASSES = len(CLASS_ID_TO_NAME)  # 11
CLASS_NAMES = [CLASS_ID_TO_NAME[i] for i in range(NUM_CLASSES)]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(x):
    """x: B,3,H,W (0..1) -> ImageNet 正規化 (SegFormer の前処理と同じ)"""
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def color_mask_to_index(mask_rgb):
    """色分けマスク (examples_Ramen_val55/*.png) -> クラス番号。パレットに無い色は 255 (評価で無視)"""
    out = np.full(mask_rgb.shape[:2], 255, np.int64)
    for cid, name in CLASS_ID_TO_NAME.items():
        out[(mask_rgb == np.array(CLASS_COLORS[name])).all(-1)] = cid
    return out


def confusion(pred, gt):
    """pred, gt: (..., H, W) の整数配列。gt=255 は無視。行 = 正解、列 = 予測"""
    k = gt != 255
    return np.bincount(NUM_CLASSES * gt[k] + pred[k], minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def miou_from_conf(conf):
    """データセット全体の混同行列から mIoU。正解に出てこないクラスは平均から除く
    (val55 には箸が 1 枚も無い。誤検出だけで IoU 0 として平均に入るのを避ける)"""
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    iou = np.where(conf.sum(1) > 0, inter / np.maximum(union, 1), np.nan)
    return float(np.nanmean(iou)), iou
