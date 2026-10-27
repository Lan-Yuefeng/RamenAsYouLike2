"""SACap-1M を FLUX 3 Image Klein 版の学習で使うための Dataset。

FLUX.2 版 (SACap1MFlux2Dataset) との違いはバケット分けのテキスト長の数え方だけ:
- テキストエンコーダは Qwen3-VL。tokenizer は repo の text_encoder/ にある AutoProcessor の .tokenizer
- caption ごとにチャットテンプレートで包み、80 (TEXT_PAD_MULTIPLE) の倍数に切り上げる
  (pipeline_flux3.Flux3RegionalPipeline._get_qwen3_vl_prompt_embeds と同じ数え方)
cond トークン数は FLUX.2 と同じ (VAE 1 トークン = 16px) なので公開キャッシュをそのまま使う。
"""
import math
import os

import pandas as pd
from tqdm import tqdm

from dataset.sacap_1m_flux2_dataset import SACap1MFlux2Dataset
from src.pipelines.pipeline_flux3 import TEXT_PAD_MULTIPLE


class SACap1MFlux3Dataset(SACap1MFlux2Dataset):
    def __init__(self, *args, tokenizer_path: str = "black-forest-labs/flux-3-image-klein-base",
                 text_pad_multiple: int = TEXT_PAD_MULTIPLE, **kwargs):
        self.text_pad_multiple = text_pad_multiple
        super().__init__(*args, tokenizer_path=tokenizer_path, **kwargs)

    def _qwen3_txt_seq_lens(self, tokenizer_path, global_max, regional_max):
        pad = self.text_pad_multiple
        cache_path = os.path.join(self.cache_root, f"qwen3vl_txt_seq_len_g{global_max}_r{regional_max}_p{pad}.parquet")
        cache = pd.read_parquet(cache_path) if os.path.exists(cache_path) else pd.DataFrame({"imagename": [], "txt_seq_len": []})
        known = dict(zip(cache["imagename"], cache["txt_seq_len"]))
        todo = self.images_info[~self.images_info["imagename"].isin(known)]

        if len(todo) > 0:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(tokenizer_path, subfolder="text_encoder")
            tok = processor.tokenizer

            template_len = len(tok(processor.apply_chat_template(
                [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True)).input_ids)
            ceil = lambda n: math.ceil(n / pad) * pad
            new = {}
            for r in tqdm(todo.itertuples(), total=len(todo), desc="qwen3-vl txt tokens"):
                regional = [s["caption"] for s in r.segments_info]
                g = len(tok(r.caption, add_special_tokens=False).input_ids)
                rs = tok(regional, add_special_tokens=False).input_ids if regional else []
                new[r.imagename] = ceil(template_len + min(g, global_max - template_len)) + sum(
                    ceil(template_len + min(len(x), regional_max)) for x in rs
                )
            known.update(new)
            pd.DataFrame({"imagename": list(known), "txt_seq_len": list(known.values())}).to_parquet(cache_path, index=False)
        return self.images_info["imagename"].map(known).to_numpy().astype(int)
