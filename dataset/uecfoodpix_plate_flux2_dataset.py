"""UECFOODPIXCOMPLETE + 皿マスク版の FLUX.2 学習用 Dataset。

UECFoodPixDataset (FLUX.1 版) との違い:
- マスクは dataset/make_uecfoodpix_plate_masks.py で作った 1ch PNG (0=テーブル等, 1..102=料理, 103=皿・器・お盆)
- caption は dataset/uecfoodpix_plate_captions.py: 背景 2 種に具体的な文言、global caption も付ける
- バケット分けは Qwen3 トークン長 (global + regional、各々チャットテンプレート込み) と cond トークン数。
  輪郭が変わるので cond 長も数え直す (cache_root/plate_*.parquet にキャッシュ)
"""
import os

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from dataset.uecfoodpix_dataset import UECFoodPixDataset
from dataset.uecfoodpix_plate_captions import global_caption, regional_caption
from src.pipelines import FluxRegionalPipeline
from utils.utils import mask2box
from utils.visualizer import Visualizer

MASK_ROOT = "/export/ssd3/lan-y/seg2any_data/uecfoodpix/mask_plate"


def _cond_token_num(segm_path, resolution, cond_transforms):
    segm = np.array(Image.open(segm_path))
    label = torch.from_numpy(np.stack([segm == i for i in np.unique(segm)], axis=0))[None].float()
    label = F.interpolate(label, size=resolution, mode="nearest-exact")[0].long()
    cond = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
    cond = Visualizer().draw_contours(cond, label.numpy(), thickness=1, colors=[(255, 255, 255)] * len(label))
    return FluxRegionalPipeline.get_valid_cond_token_num(cond_transforms(Image.fromarray(cond)))


class UECFoodPixPlateFlux2Dataset(UECFoodPixDataset):
    def __init__(
        self,
        dataset_root,
        split_file,
        image_dir,
        is_group_bucket=False,
        cache_root=None,
        resolution=512,
        cond_scale_factor: int = 1,
        mask_root: str = MASK_ROOT,
        tokenizer_path: str = "black-forest-labs/FLUX.2-klein-base-4B",
        global_max_sequence_length: int = 512,
        regional_max_sequence_length: int = 50,
        bucket_size: int = 50,
    ):
        super().__init__(
            dataset_root, split_file, image_dir, is_group_bucket=False, cache_root=cache_root,
            resolution=resolution, cond_scale_factor=cond_scale_factor,
        )
        self.segm_root = os.path.join(mask_root, image_dir)
        if not is_group_bucket:
            return
        if cache_root is None:
            raise ValueError("Bucket grouping is enabled. Please specify a cache_root directory.")
        os.makedirs(cache_root, exist_ok=True)

        tag = f"{os.path.basename(split_file).split('.')[0]}_{self.cond_resolution[0]}H_{self.cond_resolution[1]}W"
        cache_path = os.path.join(cache_root, f"plate_bucket_{tag}_g{global_max_sequence_length}_r{regional_max_sequence_length}.parquet")
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            assert list(df["seg"]) == [d["seg"] for d in self.data], f"{cache_path} is stale"
        else:
            df = self._count_tokens(tokenizer_path, global_max_sequence_length, regional_max_sequence_length)
            df.to_parquet(cache_path, index=False)
        self.flag = (df["cond_seq_len"].to_numpy() // bucket_size
                     + df["txt_seq_len"].to_numpy() // bucket_size * 1_000_000).astype(np.int64)

    def _count_tokens(self, tokenizer_path, global_max, regional_max):
        from transformers import Qwen2Tokenizer  # AutoTokenizer は subfolder だと config を探して落ちる

        tok = Qwen2Tokenizer.from_pretrained(tokenizer_path, subfolder="tokenizer")
        template = tok.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        template_len = len(tok(template).input_ids)
        txt = []
        for d in tqdm(self.data, desc="qwen3 txt tokens"):
            ids = np.unique(self.read_segm(os.path.join(self.segm_root, d["seg"]))).tolist()
            g = len(tok(global_caption(ids), add_special_tokens=False).input_ids)
            n = template_len + min(g, global_max - template_len)
            n += sum(template_len + min(len(tok(regional_caption(i), add_special_tokens=False).input_ids), regional_max) for i in ids)
            txt.append(n)
        cond = Parallel(n_jobs=16)(
            delayed(_cond_token_num)(os.path.join(self.segm_root, d["seg"]), self.resolution, self.cond_transforms)
            for d in tqdm(self.data, desc="cond tokens")
        )
        return pd.DataFrame({"seg": [d["seg"] for d in self.data], "cond_seq_len": cond, "txt_seq_len": txt})

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_root, item["image"])
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        segm_map = self.read_segm(os.path.join(self.segm_root, item["seg"]))

        ids = np.unique(segm_map).tolist()
        boxes, label, regional_captions = [], [], []
        for label_id in ids:
            mask = segm_map == label_id
            x0, y0, x1, y1 = mask2box(mask)
            boxes.append(np.array([x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h]))
            label.append(mask)
            regional_captions.append(regional_caption(label_id))

        label = torch.from_numpy(np.stack(label, axis=0))[None, ...]
        label = F.interpolate(label.float(), size=self.resolution, mode="nearest-exact")[0].long()

        cond_pixel_values = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
        cond_pixel_values = self.visualizer.draw_contours(
            cond_pixel_values, label.cpu().numpy(), thickness=1, colors=[(255, 255, 255)] * len(regional_captions)
        )
        cond_pixel_values = self.cond_transforms(Image.fromarray(cond_pixel_values))

        return {
            "label": label,
            "regional_captions": regional_captions,
            "global_caption": global_caption(ids),
            "class_ids": ids,
            "pixel_values": self.image_transforms(image),
            "cond_pixel_values": cond_pixel_values,
            "image_name": item["image"],
            "image_path": image_path,
            "boxes": boxes,
        }
