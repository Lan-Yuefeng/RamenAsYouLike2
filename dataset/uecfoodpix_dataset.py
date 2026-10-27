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

from dataset.uecfoodpix_captions import CLASS_ID_TO_NAME, regional_caption
from src.pipelines import FluxRegionalPipeline
from utils.utils import mask2box, get_text_token_len
from utils.visualizer import Visualizer


class UECFoodPixDataset(Dataset):
    """UECFOODPIXCOMPLETE (mask -> food photo).

    Layout under `dataset_root`:
        data/category.txt                          id \t name, ids 1..102
        data/train9000.txt, data/test1000.txt      one image stem per line
        data/UECFoodPIXCOMPLETE/{train,test}/img/<stem>.jpg
        data/UECFoodPIXCOMPLETE/{train,test}/mask/<stem>.png

    Three differences from ADE20KDataset, which this otherwise follows closely:

      * Masks are RGBA PNGs, not single-channel index maps. The class id lives in
        the R channel only (G=B=0, A=255), so `__getitem__` slices `[..., 0]`.
      * Image and mask have different extensions (.jpg / .png) and live in sibling
        `img/` and `mask/` directories, so the split file lists bare stems.
      * id 0 is kept rather than skipped. In ADE20K 0 means "unannotated"; here the
        dataset is fully annotated and 0 is the real non-food background (plate,
        tray, table), which we want captioned and controllable.

    Captions are bare class names with no global caption, matching ADE20KDataset.
    """

    def __init__(
        self,
        dataset_root,
        split_file,
        image_dir,
        is_group_bucket=False,
        cache_root=None,
        resolution: Union[List, int] = 512,
        cond_scale_factor: int = 1,
    ):
        super(UECFoodPixDataset, self).__init__()
        self.dataset_root = dataset_root
        split_root = os.path.join(dataset_root, "data", "UECFoodPIXCOMPLETE", image_dir)
        self.image_root = os.path.join(split_root, "img")
        self.segm_root = os.path.join(split_root, "mask")
        self.cache_root = cache_root

        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.cond_resolution = [
            self.resolution[0] // cond_scale_factor,
            self.resolution[1] // cond_scale_factor,
        ]

        if is_group_bucket:
            if self.cache_root is None:
                raise ValueError("Bucket grouping is enabled. Please specify a cache_root directory.")
            os.makedirs(self.cache_root, exist_ok=True)

        if not os.path.isabs(split_file):
            split_file = os.path.join(dataset_root, split_file)
        with open(split_file) as f:
            stems = [ln.strip() for ln in f if ln.strip()]

        self.data = [{"seg": f"{s}.png", "image": f"{s}.jpg"} for s in stems]

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

    @staticmethod
    def read_segm(path):
        """Class-id map from a UECFOODPIXCOMPLETE RGBA mask (id is in the R channel)."""
        segm = np.array(Image.open(path))
        return segm[..., 0] if segm.ndim == 3 else segm

    def _set_group_flag(self):
        print("=====UECFoodPixDataset set group flag=====")
        cache_path = os.path.join(
            self.cache_root, f"{self.cond_resolution[0]}H_{self.cond_resolution[1]}W-group_bucket.parquet"
        )
        if os.path.exists(cache_path):
            result_df = pd.read_parquet(cache_path)
        else:
            results = Parallel(n_jobs=24)(
                delayed(UECFoodPixDataset.get_token_num)(
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
        segm_map = UECFoodPixDataset.read_segm(os.path.join(segm_root, item["seg"]))

        label = []
        txt_seq_len = 0
        for label_id in np.unique(segm_map).tolist():
            mask = segm_map == label_id
            label.append(mask)
            txt_seq_len += get_text_token_len(tokenizer, regional_caption(CLASS_ID_TO_NAME[label_id]))

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

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size
        segm_map = self.read_segm(segm_path)

        boxes = []
        cat_names = []
        label = []
        regional_captions = []

        for label_id in np.unique(segm_map).tolist():
            class_name = CLASS_ID_TO_NAME[label_id]
            mask = segm_map == label_id

            x0, y0, x1, y1 = mask2box(mask)
            boxes.append(np.array([x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h]))
            label.append(mask)
            cat_names.append(class_name)
            regional_captions.append(regional_caption(class_name))

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
            "global_caption": None,
            "pixel_values": pixel_values,
            "cond_pixel_values": cond_pixel_values,
            "image_name": item["image"],
            "image_path": image_path,
            "boxes": boxes,
        }
