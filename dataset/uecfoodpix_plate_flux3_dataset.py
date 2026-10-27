"""UECFOODPIXCOMPLETE + 皿マスク版の FLUX 3 Image Klein 学習用 Dataset。

FLUX.2 版 (UECFoodPixPlateFlux2Dataset) との違いはバケット分けのテキスト長の数え方だけ
(dataset/sacap_1m_flux3_dataset.py と同じ):
- テキストエンコーダは Qwen3-VL。tokenizer は repo の text_encoder/ にある AutoProcessor の .tokenizer
- caption ごとにチャットテンプレートで包み、80 (TEXT_PAD_MULTIPLE) の倍数に切り上げる
cond トークン数は FLUX.2 と同じ (VAE 1 トークン = 16px)。

マスクは研究室共有の UECFOODPIXCOMPLETEwithPLATE_104 (/export/data/dataset/UECFOOD) を ssd2 にコピーしたもの。
値は make_uecfoodpix_plate_masks.py の自作版と同じ (0=テーブル等, 1..102=料理, 103=皿)。
原本は 3ch 同値 PNG だが、_cond_token_num が 1ch 前提なので ssd2 のコピーは R だけの 1ch に変換してある
"""
import math
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from dataset.uecfoodpix_plate_captions import global_caption, regional_caption
from dataset.uecfoodpix_plate_flux2_dataset import UECFoodPixPlateFlux2Dataset, _cond_token_num
from src.pipelines.pipeline_flux3 import TEXT_PAD_MULTIPLE

MASK_ROOT = "/export/ssd2/lan-y/seg2any_data/uecfoodpix/mask_plate_lab"


class UECFoodPixPlateFlux3Dataset(UECFoodPixPlateFlux2Dataset):
    def __init__(self, *args, mask_root: str = MASK_ROOT,
                 tokenizer_path: str = "black-forest-labs/flux-3-image-klein-base",
                 text_pad_multiple: int = TEXT_PAD_MULTIPLE, **kwargs):
        self.text_pad_multiple = text_pad_multiple
        super().__init__(*args, mask_root=mask_root, tokenizer_path=tokenizer_path, **kwargs)

    def _count_tokens(self, tokenizer_path, global_max, regional_max):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(tokenizer_path, subfolder="text_encoder")
        tok = processor.tokenizer
        template_len = len(tok(processor.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True)).input_ids)
        pad = self.text_pad_multiple
        ceil = lambda n: math.ceil(n / pad) * pad
        txt = []
        for d in tqdm(self.data, desc="qwen3-vl txt tokens"):
            ids = np.unique(self.read_segm(os.path.join(self.segm_root, d["seg"]))).tolist()
            g = len(tok(global_caption(ids), add_special_tokens=False).input_ids)
            n = ceil(template_len + min(g, global_max - template_len))
            n += sum(ceil(template_len + min(len(tok(regional_caption(i), add_special_tokens=False).input_ids), regional_max)) for i in ids)
            txt.append(n)
        cond = Parallel(n_jobs=16)(
            delayed(_cond_token_num)(os.path.join(self.segm_root, d["seg"]), self.resolution, self.cond_transforms)
            for d in tqdm(self.data, desc="cond tokens")
        )
        return pd.DataFrame({"seg": [d["seg"] for d in self.data], "cond_seq_len": cond, "txt_seq_len": txt})
