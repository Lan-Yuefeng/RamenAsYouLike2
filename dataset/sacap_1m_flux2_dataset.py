"""SACap-1M を FLUX.2 (klein) 版の学習で使うための Dataset。

FLUX.1 版の SACap_1M_Dataset との違い:
- 画像が手元にある行だけを使う (SA-1B を一部の tar しか落としていない状態で学習できるように)。
- バケット分けのテキスト長を T5 ではなく Qwen3 のトークン数で数える。FLUX.2 版は caption ごとに
  チャットテンプレートで包んで別々にエンコードするので、テンプレート長も caption ごとに足す
  (pipeline_flux2.encode_all_prompt と同じ数え方)。
- cond トークン数は公開されている 512H_512W-group_bucket.parquet (anno_train.parquet と同じ行順) を使う。
  FLUX.2 の VAE も 1 トークン = 16px なので数はそのまま使える。無ければ自前で数える。
- FLUX.1 版の _set_group_flag は ADE20K 版のコピーで self.data / self.segm_root を参照して動かないので使わない。

__getitem__ は FLUX.1 版をそのまま使う (出力形式は FLUX.1 / FLUX.2 で共通)。
"""
import json
import os
from typing import List, Union

import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
from joblib import Parallel, delayed
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from dataset.sacap_1m_dataset import SACap_1M_Dataset
from src.pipelines import FluxRegionalPipeline
from utils.visualizer import Visualizer


def _cond_token_num(image_path, segment_ids, resolution, cond_transforms):
    anno_path = image_path[: image_path.rfind(".")] + ".json"
    with open(anno_path, "r", encoding="utf-8") as f:
        anns = json.load(f)
    label = [mask_util.decode(seg["segmentation"]) == 1 for seg in anns["annotations"] if seg["id"] in segment_ids]
    if len(label) == 0:
        return 0
    label = torch.from_numpy(np.stack(label, axis=0))[None].float()
    label = F.interpolate(label, size=resolution, mode="nearest-exact")[0].long()
    cond = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
    cond = Visualizer().draw_contours(cond, label.numpy(), thickness=1, colors=[(255, 255, 255)] * len(label))
    return FluxRegionalPipeline.get_valid_cond_token_num(cond_transforms(Image.fromarray(cond)))


class SACap1MFlux2Dataset(SACap_1M_Dataset):
    def __init__(
        self,
        seg_caption_path,
        image_root,
        is_group_bucket=False,
        cache_root=None,
        resolution: Union[List, int] = 1024,
        cond_scale_factor: int = 2,
        tokenizer_path: str = "black-forest-labs/FLUX.2-klein-base-4B",
        global_max_sequence_length: int = 512,
        regional_max_sequence_length: int = 50,
        bucket_size: int = 50,
        max_samples: int = None,
    ):
        Dataset.__init__(self)
        self.image_root = image_root
        self.seg_caption_path = seg_caption_path
        self.cache_root = cache_root
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.cond_resolution = [self.resolution[0] // cond_scale_factor, self.resolution[1] // cond_scale_factor]

        images_info = pd.read_parquet(seg_caption_path)
        images_info["orig_idx"] = np.arange(len(images_info))
        if "image_group" not in images_info:
            images_info["image_group"] = ""  # SACap-eval は test/ 直下に平置き
        # 100 万行を os.path.exists で調べると NFS では遅すぎる (x ランク数) ので、ディレクトリを一覧して突き合わせる
        listing = {}
        for g in images_info["image_group"].unique():
            d = os.path.join(image_root, g)
            listing[g] = set(os.listdir(d)) if os.path.isdir(d) else set()
        exists = [
            n in listing[g] and (n[: n.rfind(".")] + ".json") in listing[g]
            for g, n in zip(images_info["image_group"], images_info["imagename"])
        ]
        self.num_total = len(images_info)
        images_info = images_info[np.array(exists, dtype=bool)].reset_index(drop=True)
        if max_samples is not None:
            images_info = images_info.iloc[:max_samples].reset_index(drop=True)
        self.images_info = images_info
        print(f"===== SACap1MFlux2Dataset: {len(self)} / {self.num_total} images available under {image_root} =====")
        if len(self) == 0:
            raise ValueError(f"no images found under {image_root}")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.cond_transforms = transforms.Compose(
            [
                transforms.Resize(self.cond_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.visualizer = Visualizer()

        if is_group_bucket:
            if cache_root is None:
                raise ValueError("Bucket grouping is enabled. Please specify a cache_root.")
            os.makedirs(cache_root, exist_ok=True)
            cond_len = self._cond_seq_lens()
            txt_len = self._qwen3_txt_seq_lens(tokenizer_path, global_max_sequence_length, regional_max_sequence_length)
            self.flag = (cond_len // bucket_size + txt_len // bucket_size * 1_000_000).astype(np.int64)
        else:
            self.flag = np.zeros(len(self), dtype=np.int64)

    def _cond_seq_lens(self):
        published = os.path.join(self.cache_root, f"{self.cond_resolution[0]}H_{self.cond_resolution[1]}W-group_bucket.parquet")
        if os.path.exists(published):
            df = pd.read_parquet(published, columns=["cond_seq_len"])
            if len(df) == self.num_total:
                return df["cond_seq_len"].to_numpy()[self.images_info["orig_idx"].to_numpy()]
            print(f"{published} has {len(df)} rows, expected {self.num_total}; counting cond tokens instead")
        lens = Parallel(n_jobs=16)(
            delayed(_cond_token_num)(
                os.path.join(self.image_root, r.image_group, r.imagename),
                {s["anno_id"] for s in r.segments_info},
                self.resolution,
                self.cond_transforms,
            )
            for r in tqdm(self.images_info.itertuples(), total=len(self), desc="cond tokens")
        )
        return np.array(lens, dtype=np.int64)

    def _qwen3_txt_seq_lens(self, tokenizer_path, global_max, regional_max):
        # imagename -> txt_seq_len のキャッシュ。画像を足していっても差分だけ数えればよい
        cache_path = os.path.join(self.cache_root, f"qwen3_txt_seq_len_g{global_max}_r{regional_max}.parquet")
        cache = pd.read_parquet(cache_path) if os.path.exists(cache_path) else pd.DataFrame({"imagename": [], "txt_seq_len": []})
        known = dict(zip(cache["imagename"], cache["txt_seq_len"]))
        todo = self.images_info[~self.images_info["imagename"].isin(known)]

        if len(todo) > 0:
            from transformers import Qwen2Tokenizer  # Flux2KleinPipeline と同じクラス (AutoTokenizer は subfolder だと config を探して落ちる)

            tok = Qwen2Tokenizer.from_pretrained(tokenizer_path, subfolder="tokenizer")
            template = tok.apply_chat_template(
                [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            template_len = len(tok(template).input_ids)
            new = {}
            for r in tqdm(todo.itertuples(), total=len(todo), desc="qwen3 txt tokens"):
                regional = [s["caption"] for s in r.segments_info]
                g = len(tok(r.caption, add_special_tokens=False).input_ids)
                rs = tok(regional, add_special_tokens=False).input_ids if regional else []
                new[r.imagename] = (
                    template_len + min(g, global_max - template_len)
                    + sum(template_len + min(len(x), regional_max) for x in rs)
                )
            known.update(new)
            pd.DataFrame({"imagename": list(known), "txt_seq_len": list(known.values())}).to_parquet(cache_path, index=False)
        return self.images_info["imagename"].map(known).to_numpy().astype(np.int64)
