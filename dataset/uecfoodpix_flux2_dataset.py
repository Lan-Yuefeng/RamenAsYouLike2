"""UECFOODPIXCOMPLETE を FLUX.2 (klein) 版の学習で使うための Dataset。

FLUX.1 版の UECFoodPixDataset との違いはバケット分けだけ:
- テキスト長を T5 ではなく Qwen3 のトークン数で数える。FLUX.2 版は caption ごとにチャットテンプレートで
  包んで別々にエンコードするので、テンプレート長も caption ごとに足す (pipeline_flux2.encode_all_prompt と同じ数え方)。
  global_caption は None なので global 分は 0。
- cond トークン数は FLUX.1 版が作った <cache_root>/512H_512W-group_bucket.parquet (split と同じ行順) を流用する。
  FLUX.2 の VAE も 1 トークン = 16px なので数はそのまま使える。

__getitem__ は FLUX.1 版をそのまま使う (出力形式は FLUX.1 / FLUX.2 で共通)。
"""
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.uecfoodpix_captions import CLASS_ID_TO_NAME, regional_caption
from dataset.uecfoodpix_dataset import UECFoodPixDataset


class UECFoodPixFlux2Dataset(UECFoodPixDataset):
    def __init__(
        self,
        dataset_root,
        split_file,
        image_dir,
        is_group_bucket=False,
        cache_root=None,
        resolution=512,
        cond_scale_factor: int = 1,
        tokenizer_path: str = "black-forest-labs/FLUX.2-klein-base-4B",
        regional_max_sequence_length: int = 50,
        bucket_size: int = 50,
    ):
        # 親の is_group_bucket=True は T5 tokenizer を読むので使わず、ここで数え直す
        super().__init__(
            dataset_root, split_file, image_dir, is_group_bucket=False, cache_root=cache_root,
            resolution=resolution, cond_scale_factor=cond_scale_factor,
        )
        if not is_group_bucket:
            return
        if cache_root is None:
            raise ValueError("Bucket grouping is enabled. Please specify a cache_root directory.")

        cond_cache = os.path.join(cache_root, f"{self.cond_resolution[0]}H_{self.cond_resolution[1]}W-group_bucket.parquet")
        if not os.path.exists(cond_cache):
            raise FileNotFoundError(f"{cond_cache} がない。FLUX.1 版 (train_uecfoodpix.sh) の初回実行で作られる")
        cond_len = pd.read_parquet(cond_cache, columns=["cond_seq_len"])["cond_seq_len"].to_numpy()
        assert len(cond_len) == len(self), f"{cond_cache}: {len(cond_len)} rows, dataset has {len(self)}"

        txt_len = self._qwen3_txt_seq_lens(cache_root, tokenizer_path, regional_max_sequence_length)
        self.flag = (cond_len // bucket_size + txt_len // bucket_size * 1_000_000).astype(np.int64)

    def _qwen3_txt_seq_lens(self, cache_root, tokenizer_path, regional_max):
        cache_path = os.path.join(cache_root, f"qwen3_txt_seq_len_r{regional_max}.parquet")
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            if len(df) == len(self) and list(df["seg"]) == [d["seg"] for d in self.data]:
                return df["txt_seq_len"].to_numpy().astype(np.int64)

        from transformers import Qwen2Tokenizer  # AutoTokenizer は subfolder だと config を探して落ちる

        tok = Qwen2Tokenizer.from_pretrained(tokenizer_path, subfolder="tokenizer")
        template = tok.apply_chat_template(
            [{"role": "user", "content": ""}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        template_len = len(tok(template).input_ids)
        caption_len = {
            i: template_len + min(len(tok(regional_caption(n), add_special_tokens=False).input_ids), regional_max)
            for i, n in CLASS_ID_TO_NAME.items()
        }
        lens = [
            sum(caption_len[c] for c in np.unique(self.read_segm(os.path.join(self.segm_root, d["seg"]))).tolist())
            for d in tqdm(self.data, desc="qwen3 txt tokens")
        ]
        pd.DataFrame({"seg": [d["seg"] for d in self.data], "txt_seq_len": lens}).to_parquet(cache_path, index=False)
        return np.array(lens, dtype=np.int64)
