import json
import os
from collections import defaultdict
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from joblib import Parallel, delayed
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import T5TokenizerFast

from dataset.ramen_captions import (
    CLASS_ID_TO_NAME,
    global_caption,
    load_soup_labels,
    regional_caption,
)
from src.pipelines import FluxRegionalPipeline
from utils.utils import mask2box, get_text_token_len
from utils.visualizer import Visualizer


class RamenDataset(Dataset):
    """UEC-Ramen555 (mask -> ramen photo).

    Ramen_masks/*.png are single-channel index maps using the same 0..10 ids as
    label_list.txt, which makes this dataset structurally identical to ADE20K --
    this class is a close adaptation of ADE20KDataset.

    Two deliberate differences from ADE20KDataset:
      * id 0 is kept. In ADE20K 0 means "unannotated", but here it is the table
        background, a real fully-annotated class we want to caption and control.
      * global_caption is a real sentence built from the soup label rather than
        None, so that soup type is controllable and so that inference prompts
        (examples_Ramen/*.json) are in-distribution.
    """

    def __init__(
        self,
        dataset_root,
        split_file,
        is_group_bucket=False,
        cache_root=None,
        resolution: Union[List, int] = 512,
        cond_scale_factor: int = 1,
        caption_dir=None,
    ):
        super(RamenDataset, self).__init__()
        self.dataset_root = dataset_root
        self.image_root = os.path.join(dataset_root, "Ramen_images")
        self.segm_root = os.path.join(dataset_root, "Ramen_masks")
        self.cache_root = cache_root
        # caption_dir: VLM caption の JSON ({"global", "regions": {class: str}}) の置き場。
        # None なら ramen_captions.py の手書きテンプレート ([10-26]exp_vlm_caption)
        self.caption_dir = caption_dir

        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.cond_resolution = [
            self.resolution[0] // cond_scale_factor,
            self.resolution[1] // cond_scale_factor,
        ]

        if is_group_bucket:
            if self.cache_root is None:
                raise ValueError("Bucket grouping is enabled. Please specify a cache_root directory.")
            os.makedirs(self.cache_root, exist_ok=True)

        self.soup_labels = load_soup_labels(dataset_root)

        with open(split_file) as f:
            names = [ln.strip() for ln in f if ln.strip()]

        self.data = []
        for name in names:
            self.data.append({"seg": name, "image": name, "soup": self.soup_labels[name]})

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
            self.tokenizer = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")
            self._set_group_flag()
        else:
            self.flag = np.full([len(self)], fill_value=0, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def _set_group_flag(self):
        print("=====RamenDataset set group flag=====")
        cache_path = os.path.join(
            self.cache_root, f"{self.cond_resolution[0]}H_{self.cond_resolution[1]}W-group_bucket.parquet"
        )
        if os.path.exists(cache_path):
            result_df = pd.read_parquet(cache_path)
        else:
            results = Parallel(n_jobs=24)(
                delayed(RamenDataset.get_token_num)(
                    id,
                    data=self.data,
                    segm_root=self.segm_root,
                    resolution=self.resolution,
                    tokenizer=self.tokenizer,
                    visualizer=self.visualizer,
                    cond_transforms=self.cond_transforms,
                )
                for id in tqdm(range(len(self)))
            )
            save_data = defaultdict(list)
            for res in results:
                for k in res:
                    save_data[k].append(res[k])
            result_df = pd.DataFrame(save_data)
            result_df.to_parquet(cache_path, index=False)

        assert len(result_df) == len(self)

        flags = []
        for _, row in result_df.iterrows():
            flags.append(row["cond_seq_len"] // 50 + row["txt_seq_len"] // 50 * 1e6)
        self.flag = np.array(flags, dtype=np.int64)

    @staticmethod
    def get_token_num(idx, data, segm_root, resolution, tokenizer, visualizer, cond_transforms):
        item = data[idx]
        segm_map = np.array(Image.open(os.path.join(segm_root, item["seg"])))
        soup_type = item["soup"]

        label = []
        txt_seq_len = 0
        for label_id in np.unique(segm_map).tolist():
            mask = segm_map == label_id
            label.append(mask)
            txt_seq_len += get_text_token_len(
                tokenizer, regional_caption(CLASS_ID_TO_NAME[label_id], soup_type)
            )

        if len(label) > 0:
            label = torch.from_numpy(np.stack(label, axis=0))[None, ...]
            label = F.interpolate(label.float(), size=resolution, mode="nearest-exact")[0].long()

            cond_pixel_values = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
            cond_pixel_values = visualizer.draw_contours(
                cond_pixel_values, label.cpu().numpy(), thickness=1, colors=[(255, 255, 255)] * len(label)
            )
            cond_pixel_values = cond_transforms(Image.fromarray(cond_pixel_values))
            valid_cond_token_num = FluxRegionalPipeline.get_valid_cond_token_num(cond_pixel_values)
        else:
            valid_cond_token_num = 0

        return {"cond_seq_len": valid_cond_token_num, "txt_seq_len": txt_seq_len}

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_root, item["image"])
        segm_path = os.path.join(self.segm_root, item["seg"])
        soup_type = item["soup"]

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        segm_map = np.array(Image.open(segm_path))

        boxes = []
        cat_names = []
        label = []
        regional_captions = []
        vlm = None
        if self.caption_dir is not None:
            vlm = json.load(open(os.path.join(self.caption_dir, item["image"].replace(".png", ".json"))))

        for label_id in np.unique(segm_map).tolist():
            class_name = CLASS_ID_TO_NAME[label_id]
            mask = segm_map == label_id

            x0, y0, x1, y1 = mask2box(mask)
            boxes.append(np.array([x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h]))
            label.append(mask)
            cat_names.append(class_name)
            regional_captions.append(
                vlm["regions"][class_name] if vlm else regional_caption(class_name, soup_type)
            )

        if len(regional_captions) == 0:  # try again
            return self.__getitem__(np.random.randint(len(self)))

        label = torch.from_numpy(np.stack(label, axis=0))[None, ...]
        label = F.interpolate(label.float(), size=self.resolution, mode="nearest-exact")[0].long()

        pixel_values = self.image_transforms(image)

        cond_pixel_values = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
        cond_pixel_values = self.visualizer.draw_contours(
            cond_pixel_values,
            label.cpu().numpy(),
            thickness=1,
            colors=[(255, 255, 255)] * len(regional_captions),
        )
        cond_pixel_values = self.cond_transforms(Image.fromarray(cond_pixel_values))

        return {
            "label": label,
            "regional_captions": regional_captions,
            "global_caption": vlm["global"] if vlm else global_caption(cat_names, soup_type),
            "pixel_values": pixel_values,
            "cond_pixel_values": cond_pixel_values,
            "image_name": item["image"],
            "image_path": image_path,
            "boxes": boxes,
        }
