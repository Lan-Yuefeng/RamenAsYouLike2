"""マスクの境界と生成画像のエッジの一致度 (scripts/flux2_cfg_sweep.py / scripts/flux2_eval_val55.py で共通に使う)。

recall    : マスクの境界画素のうち、画像の Canny エッジが tol px 以内にある割合 (境界の位置に実際の境目があるか)
precision : 画像のエッジのうち、マスク境界の tol px 以内にある割合 (麺などの模様のエッジも数えるので低めに出る)
"""
import cv2
import numpy as np
from PIL import Image


def mask_boundary(mask_rgb):
    """色分けマスク -> 境界画素 (隣と色が違う画素)。画像の外周は除く"""
    m = mask_rgb.astype(np.int32)
    ids = m[..., 0] * 65536 + m[..., 1] * 256 + m[..., 2]
    b = np.zeros(ids.shape, bool)
    b[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    b[1:, :] |= ids[1:, :] != ids[:-1, :]
    b[[0, -1], :] = False
    b[:, [0, -1]] = False
    return b


def image_edges(img_rgb):
    gray = cv2.GaussianBlur(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    med = np.median(gray)  # 画像ごとに閾値を決める (auto-Canny)
    lo, hi = int(max(0, 0.67 * med)), int(min(255, 1.33 * med))
    return cv2.Canny(gray, lo, hi) > 0


def boundary_score(img_rgb, mask_rgb, tol=3):
    b = mask_boundary(mask_rgb)
    e = image_edges(img_rgb)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    e_d = cv2.dilate(e.astype(np.uint8), k) > 0
    b_d = cv2.dilate(b.astype(np.uint8), k) > 0
    recall = (b & e_d).sum() / max(1, b.sum())
    precision = (e & b_d).sum() / max(1, e.sum())
    return float(recall), float(precision)


def load_rgb(path, size=512):
    return np.array(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC))


def load_mask(path):
    """色分けマスクは補間すると境界に中間色ができるので、リサイズせずに読む"""
    return np.array(Image.open(path).convert("RGB"))
