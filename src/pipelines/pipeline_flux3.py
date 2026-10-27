# Seg2Any の領域制御パイプラインを FLUX 3 Image Klein に移植したもの。
# diffusers には FLUX.3 の ModularPipeline しか無いので、その各ブロック (modular_pipelines/flux3) の処理を
# 1 つの DiffusionPipeline にまとめ直し、FLUX.2 版 (pipeline_flux2.py) と同じ __call__ の形にした。
# FLUX.2 版からの主な変更:
#   - テキスト: Qwen3-VL の hidden states 8 層 (4, 8, ..., 32) を連結した 20480 次元
#   - 各 caption は公式と同じく「チャットテンプレートで包む → 80 の倍数に右パディング (パディングの position id は 1)」
#     してからエンコードする。FLUX.3 は transformer がパディングも見る前提で学習されているので、
#     パディングはその caption のトークンとして残し、マスク上も caption と同じ区間に入れる
#   - 画像 / cond の位置 id は (0, 0.5h, 0.5w, l)。noisy は l=0、cond は参照画像 1 枚目と同じ l=-1
#   - スケジューラは shift 6.93 固定。蒸留版 (flow map) はステップごとに飛び先の timestep を渡し、CFG なし
# 領域アテンションマスクの作り方と cond トークンの間引きは FLUX.1 版の関数をそのまま使う。

import math
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from diffusers.loaders import Flux3LoraLoaderMixin
from diffusers.modular_pipelines.flux3.image_processor import Flux3ImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

from .pipeline_flux import FluxRegionalPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

SCHEDULER_SHIFT = 6.93
IMAGE_ROPE_SCALE = 0.5
TEXT_PAD_MULTIPLE = 80
TEXT_ENCODER_OUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)


def prepare_image_ids(height: int, width: int, l_coord: int = 0, pos_scale: int = 1) -> torch.Tensor:
    """(height*width, 4) の (t, h, w, l)。h, w は FLUX.3 の image_rope_scale (0.5) 倍。
    pos_scale は cond を noisy より粗い格子で作るとき (cond_scale_factor) に座標を noisy 側に合わせる倍率。"""
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(height), torch.arange(width), torch.tensor([l_coord])
    ).to(torch.float32)
    ids[:, 1:3] *= IMAGE_ROPE_SCALE * pos_scale
    return ids


class Flux3RegionalPipeline(DiffusionPipeline, Flux3LoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    # FLUX.1 版と共通の staticmethod
    prepare_attention_mask = staticmethod(FluxRegionalPipeline.prepare_attention_mask)
    filter_cond_token = staticmethod(FluxRegionalPipeline.filter_cond_token)

    def __init__(self, transformer, vae, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.register_modules(
            transformer=transformer, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler
        )
        self.vae_scale_factor = 8
        self.image_processor = Flux3ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        transformer=None,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_text_encoder: bool = True,
        **kwargs,
    ):
        """FLUX.3 の repo は ModularPipeline 用 (modular_model_index.json) なので、部品を 1 つずつ読む。
        transformer=None なら transformer なし (学習時のテキストエンコード専用パイプライン用)。"""
        repo = pretrained_model_name_or_path
        vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=torch_dtype)
        text_encoder = tokenizer = None
        if load_text_encoder:
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(repo, subfolder="text_encoder", dtype=torch_dtype)
            tokenizer = AutoProcessor.from_pretrained(repo, subfolder="text_encoder")
        scheduler = FlowMatchEulerDiscreteScheduler(shift=SCHEDULER_SHIFT)
        return cls(transformer=transformer, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler)

    @property
    def is_distilled(self) -> bool:
        return bool(self.transformer.config.flow_map)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return not self.is_distilled and self._guidance_scale > 1

    # ------------------------------------------------------------------ text
    def _wrap_chat_template(self, prompt: str, max_caption_tokens: int) -> str:
        # 長すぎる caption はテンプレートで包む前に caption 部分だけ切る (テンプレートの末尾を残すため)
        tok = self.tokenizer.tokenizer
        ids = tok(prompt, add_special_tokens=False).input_ids
        if len(ids) > max_caption_tokens:
            logger.warning(f"caption truncated to {max_caption_tokens} tokens: {tok.decode(ids[max_caption_tokens:])!r}")
            prompt = tok.decode(ids[:max_caption_tokens])
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def _get_qwen3_vl_prompt_embeds(
        self,
        prompts: List[str],
        max_caption_tokens: int,
        device: torch.device,
        pad_multiple: int = TEXT_PAD_MULTIPLE,
    ) -> List[torch.Tensor]:
        """各 prompt を公式 (Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds) と同じ手順でエンコードし、
        (L_i, 20480) のリストで返す。L_i は「テンプレート込みの長さを pad_multiple の倍数に切り上げた長さ」。
        pad_multiple=1 ならパディングなし (FLUX.2 版と同じ扱い。比較実験用)。
        同じ L_i になる prompt はまとめて 1 バッチで通す (パディング長が同じなので結果は 1 本ずつと一致する)。"""
        tok = self.tokenizer.tokenizer
        texts = [self._wrap_chat_template(p, max_caption_tokens) for p in prompts]
        real_lens = [len(tok(t).input_ids) for t in texts]
        padded_lens = [math.ceil(n / pad_multiple) * pad_multiple for n in real_lens]

        te_device = self.text_encoder.device
        # 学習時のテキスト専用パイプラインは transformer=None で作るので、その場合はエンコーダの dtype
        dtype = self.transformer.dtype if self.transformer is not None else self.text_encoder.dtype
        out_list: List[Optional[torch.Tensor]] = [None] * len(texts)
        for length in sorted(set(padded_lens)):
            idx = [i for i, n in enumerate(padded_lens) if n == length]
            inputs = tok([texts[i] for i in idx], return_tensors="pt", padding="max_length", padding_side="right",
                         truncation=True, max_length=length)
            input_ids = inputs["input_ids"].to(te_device)
            attention_mask = inputs["attention_mask"].to(te_device)
            # 公式と同じく、パディングの position id は 1 (学習環境の transformers 4.57 の挙動) に固定する
            position_ids = torch.arange(length, device=te_device)[None].expand(len(idx), -1)
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            output = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                output_hidden_states=True, use_cache=False,
            )
            out = torch.stack([output.hidden_states[k] for k in TEXT_ENCODER_OUT_LAYERS], dim=1)  # B, 8, L, D
            b, c, l, d = out.shape
            out = out.permute(0, 2, 1, 3).reshape(b, l, c * d).to(device=device, dtype=dtype)
            for j, i in enumerate(idx):
                out_list[i] = out[j]
        return out_list

    def encode_all_prompt(
        self,
        global_prompt: Optional[List[str]],
        regional_prompts: List[List[str]],
        global_max_sequence_length: int = 512,
        regional_max_sequence_length: int = 50,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        regional_text_position: str = "restart",
        text_pad_multiple: int = TEXT_PAD_MULTIPLE,
    ):
        """returns
        prompt_embeds: (B*N, S, 20480)。並びは [global, regional_1, ..., regional_n, zero pad]
            各 caption は text_pad_multiple の倍数長 (Qwen 側のパディング込み)
        txt_seq_lens : list[list[int]]。[global 長, regional_1 長, ..., (バッチ揃えの pad 長)]。prepare_attention_mask 用
        text_ids     : (B*N, S, 4)。(0, 0, 0, l)
        regional_text_position:
            "restart"    -> l を global / 各 regional caption ごとに 0 から振り直す (既定)
            "continuous" -> l を系列全体の通し番号にする (比較実験用)
        global_max_sequence_length: global caption のテンプレート込みの上限 (caption 部分はそこからテンプレート長を引いた分)
        regional_max_sequence_length: regional caption の caption 部分の上限 (テンプレートは含まない)
        """
        device = device or self._execution_device
        template_len = len(self.tokenizer.tokenizer(self._wrap_chat_template("", 10**6)).input_ids)

        if isinstance(regional_prompts, list) and isinstance(regional_prompts[0], str):
            regional_prompts = [regional_prompts]
        batch_size = len(regional_prompts)

        global_embeds = None
        if global_prompt is not None:
            global_embeds = self._get_qwen3_vl_prompt_embeds(
                global_prompt, global_max_sequence_length - template_len, device, text_pad_multiple
            )

        flat = [p for ps in regional_prompts for p in ps]
        flat_embeds = self._get_qwen3_vl_prompt_embeds(flat, regional_max_sequence_length, device, text_pad_multiple)
        offsets = np.cumsum([0] + [len(ps) for ps in regional_prompts])

        embeds, txt_seq_lens, positions = [], [], []
        for i in range(batch_size):
            segs = ([global_embeds[i]] if global_embeds is not None else []) + list(
                flat_embeds[offsets[i]: offsets[i + 1]]
            )
            lens = [s.shape[0] for s in segs]
            if global_embeds is None:
                lens = [0] + lens  # prepare_attention_mask は先頭を global 長とみなす
            embeds.append(torch.cat(segs, dim=0))
            txt_seq_lens.append(lens)
            if regional_text_position == "restart":
                positions.append(torch.cat([torch.arange(n) for n in lens]))
            elif regional_text_position == "continuous":
                positions.append(torch.arange(sum(lens)))
            else:
                raise ValueError(f"regional_text_position must be 'restart' or 'continuous', got {regional_text_position}")

        # バッチ内で長さを揃える (zero 埋め込み。マスクで遮断されるので位置は 0 でよい)
        max_len = max(e.shape[0] for e in embeds)
        for i in range(batch_size):
            pad = max_len - embeds[i].shape[0]
            if pad > 0:
                embeds[i] = torch.cat([embeds[i], embeds[i].new_zeros(pad, embeds[i].shape[1])], dim=0)
                positions[i] = torch.cat([positions[i], torch.zeros(pad, dtype=positions[i].dtype)])
                txt_seq_lens[i].append(pad)

        prompt_embeds = torch.stack(embeds, dim=0).repeat_interleave(num_images_per_prompt, dim=0)
        l = torch.stack(positions, dim=0).repeat_interleave(num_images_per_prompt, dim=0)
        text_ids = torch.zeros(*l.shape, 4, dtype=torch.float32)
        text_ids[..., 3] = l.float()
        text_ids = text_ids.to(device)
        txt_seq_lens = [lens for lens in txt_seq_lens for _ in range(num_images_per_prompt)]
        return prompt_embeds, txt_seq_lens, text_ids

    # ------------------------------------------------------------------ latents
    @staticmethod
    def _patchify_latents(latents):
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify_latents(latents):
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    def _bn_stats(self, device, dtype):
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
        std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(device, dtype)
        return mean, std

    @torch.no_grad()
    def encode_vae_image(self, image: torch.Tensor) -> torch.Tensor:
        """[-1, 1] の画像 (B, 3, H, W) -> 正規化済みのパッチ化 latent (B, 128, H/16, W/16)。Flux3VaeEncoderStep と同じ。"""
        latents = self.vae.encode(image.to(self.vae.device, self.vae.dtype)).latent_dist.mode()
        latents = self._patchify_latents(latents)
        mean, std = self._bn_stats(latents.device, latents.dtype)
        return (latents - mean) / std

    @staticmethod
    def pack_latents(latents: torch.Tensor) -> torch.Tensor:
        return latents.flatten(2).transpose(1, 2)  # (B, C, h, w) -> (B, h*w, C)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, token_h: int, token_w: int, output_type: str = "pil"):
        b, _, c = latents.shape
        latents = latents.view(b, token_h, token_w, c).permute(0, 3, 1, 2)
        mean, std = self._bn_stats(latents.device, latents.dtype)
        latents = self._unpatchify_latents(latents * std + mean)
        if output_type == "latent":
            return latents
        image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type=output_type)

    # ------------------------------------------------------------------ call
    @torch.no_grad()
    def __call__(
        self,
        global_prompt: Optional[Union[str, List[str]]],
        regional_prompts: Union[List[str], List[List[str]]],
        regional_labels: Union[torch.Tensor, List[torch.Tensor]],
        negative_prompt: str = "",
        attention_mask_method: str = "hard",
        hard_attn_block_range: Optional[List[int]] = None,
        cond: Optional[torch.Tensor] = None,
        cond_scale_factor: int = 1,
        is_filter_cond_token: bool = True,
        cond2image_attention_weight: float = 1,
        regional_text_position: str = "restart",
        text_pad_multiple: int = TEXT_PAD_MULTIPLE,
        negative_use_cond: bool = False,
        cond_steps: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        regional_max_sequence_length: int = 50,
    ):
        """引数の意味は FLUX.2 版 Flux2RegionalPipeline.__call__ と同じ。違うのは以下。
        hard_attn_block_range: None なら transformer.default_hard_attn_block_range() (FLUX.3 klein は [5, 18])
        text_pad_multiple: caption ごとのパディング単位。既定 80 (FLUX.3 公式)。1 でパディングなし
        num_inference_steps: None なら蒸留版 16 / Base 版 50 (公式の既定)
        guidance_scale: Base 版のときだけ有効 (> 1 で true CFG)。既定 3.5 は公式の値。蒸留版では無視される
        negative_use_cond: CFG のネガティブ側にも cond トークン (輪郭) を入れるか。既定 False (FLUX.2 版での結論)
        cond_steps: cond トークンを入れるのは最初の cond_steps ステップだけ (それ以降は cond なし。領域アテンションマスクと
            regional caption はそのまま)。None なら全ステップ。配置は序盤で決まるので、輪郭への忠実さと画質の兼ね合いを見る用
        """
        device = self._execution_device
        self._guidance_scale = guidance_scale
        attention_kwargs = dict(joint_attention_kwargs or {})
        if num_inference_steps is None:
            num_inference_steps = 16 if self.is_distilled else 50

        if isinstance(global_prompt, str):
            global_prompt = [global_prompt]
        if isinstance(regional_prompts, list) and isinstance(regional_prompts[0], str):
            regional_prompts = [regional_prompts]
        if isinstance(regional_labels, torch.Tensor):
            regional_labels = [regional_labels]
        batch_size = len(regional_prompts)
        for ps, lab in zip(regional_prompts, regional_labels):
            if len(ps) != len(lab):
                raise ValueError("Each regional_prompt list length must match its region_label length.")

        # 画像サイズは (16 * cond_scale_factor) の倍数に丸める
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        multiple = self.vae_scale_factor * 2 * cond_scale_factor
        height, width = int(height) // multiple * multiple, int(width) // multiple * multiple
        cond_height, cond_width = height // cond_scale_factor, width // cond_scale_factor
        token_h, token_w = height // (self.vae_scale_factor * 2), width // (self.vae_scale_factor * 2)

        # 1. テキスト
        text_kwargs = dict(
            global_max_sequence_length=max_sequence_length, regional_max_sequence_length=regional_max_sequence_length,
            num_images_per_prompt=num_images_per_prompt, device=device,
            regional_text_position=regional_text_position, text_pad_multiple=text_pad_multiple,
        )
        prompt_embeds, txt_seq_lens, text_ids = self.encode_all_prompt(global_prompt, regional_prompts, **text_kwargs)
        if self.do_classifier_free_guidance:
            neg_embeds, neg_txt_seq_lens, neg_text_ids = self.encode_all_prompt(
                [negative_prompt] * batch_size, [[negative_prompt] * len(ps) for ps in regional_prompts], **text_kwargs
            )

        # 2. cond 画像 -> VAE -> トークン。位置 id は noisy latent と同じ空間座標 (cond_scale_factor 倍)、l=-1
        cond_hidden_states = cond_ids = None
        cond_seq_lens = [0] * batch_size
        pad_seq_lens = [0] * batch_size
        if cond is not None:
            cond = self.image_processor.preprocess(cond, height=cond_height, width=cond_width)
            cond = cond.to(device=device, dtype=self.vae.dtype)
            assert len(cond) == batch_size
            cond_latents = self.encode_vae_image(cond)  # B, 128, h, w
            cond_ids = prepare_image_ids(*cond_latents.shape[-2:], l_coord=-1, pos_scale=cond_scale_factor).to(device)
            cond_hidden_states = self.pack_latents(cond_latents).to(self.transformer.dtype)
            if is_filter_cond_token:
                cond_hidden_states, cond_ids, cond_seq_lens, pad_seq_lens = self.filter_cond_token(
                    cond, cond_hidden_states, cond_ids, vae_scale_factor=self.vae_scale_factor * 2
                )
            else:
                cond_seq_lens = [cond_hidden_states.shape[1]] * batch_size
                cond_ids = cond_ids[None].expand(batch_size, -1, -1)
            cond_hidden_states = cond_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            cond_ids = cond_ids.repeat_interleave(num_images_per_prompt, dim=0)
        cond_seq_lens = [n for n in cond_seq_lens for _ in range(num_images_per_prompt)]
        pad_seq_lens = [n for n in pad_seq_lens for _ in range(num_images_per_prompt)]

        # 3. noisy latent (公式と同じく fp32 で引いてから cast。seed の互換のため)
        dtype = self.transformer.dtype
        if latents is None:
            shape = (batch_size * num_images_per_prompt, self.transformer.config.in_channels, token_h, token_w)
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)
        latents = self.pack_latents(latents.to(device, dtype))
        latent_ids = prepare_image_ids(token_h, token_w).to(device)

        # 4. timesteps (公式と同じ: linspace(1, 1/N) に shift 6.93)
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        # 5. 領域アテンションマスク (FLUX.1 版と同じ関数)
        labels = [lab.clone().to(device) for lab in regional_labels for _ in range(num_images_per_prompt)]
        mask_kwargs = dict(
            attention_mask_method=attention_mask_method, regional_labels=labels, cond_seq_lens=cond_seq_lens,
            pad_seq_lens=pad_seq_lens, height=token_h, width=token_w,
            num_attention_heads=self.transformer.config.num_attention_heads, dtype=prompt_embeds.dtype,
            device=device, cond2image_attention_weight=cond2image_attention_weight,
        )
        attention_mask, hard_attention_mask = self.prepare_attention_mask(txt_seq_lens=txt_seq_lens, **mask_kwargs)
        pos_kwargs = {**attention_kwargs, "attention_mask": attention_mask, "hard_attention_mask": hard_attention_mask}
        if cond_steps is not None and cond_hidden_states is not None:
            # cond を外した後のステップ用。系列に cond が無いのでマスクもその長さで作る (ネガティブ側と同じ)
            nocond_kwargs = {**mask_kwargs, "cond_seq_lens": [0] * len(labels), "pad_seq_lens": [0] * len(labels)}
            attention_mask, hard_attention_mask = self.prepare_attention_mask(txt_seq_lens=txt_seq_lens, **nocond_kwargs)
            pos_nocond_kwargs = {**attention_kwargs, "attention_mask": attention_mask, "hard_attention_mask": hard_attention_mask}
        if self.do_classifier_free_guidance:
            if not negative_use_cond:
                # ネガティブ側は cond トークンなし。系列に cond が無いのでマスクもその長さで作る
                mask_kwargs = {**mask_kwargs, "cond_seq_lens": [0] * len(labels), "pad_seq_lens": [0] * len(labels)}
            attention_mask, hard_attention_mask = self.prepare_attention_mask(txt_seq_lens=neg_txt_seq_lens, **mask_kwargs)
            neg_kwargs = {**attention_kwargs, "attention_mask": attention_mask, "hard_attention_mask": hard_attention_mask}

        # 6. denoising
        def predict(embeds, ids, kwargs, timestep, target_timestep, use_cond=True):
            return self.transformer(
                hidden_states=latents.to(dtype),
                encoder_hidden_states=embeds,
                timestep=timestep,
                img_ids=latent_ids,
                txt_ids=ids,
                cond_hidden_states=cond_hidden_states if use_cond else None,
                cond_ids=cond_ids if use_cond else None,
                target_timestep=target_timestep,
                hard_attn_block_range=hard_attn_block_range,
                joint_attention_kwargs=kwargs,
                return_dict=False,
            )[0]

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                timestep = (t / 1000).expand(latents.shape[0]).to(dtype)
                target_timestep = None
                if self.is_distilled:
                    # flow map: このステップの飛び先 (次の timestep、最後は 0)
                    next_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.zeros_like(t)
                    target_timestep = (next_t / 1000).expand(latents.shape[0]).to(dtype)
                if cond_steps is not None and i >= cond_steps:
                    noise_pred = predict(prompt_embeds, text_ids, pos_nocond_kwargs, timestep, target_timestep, use_cond=False)
                else:
                    noise_pred = predict(prompt_embeds, text_ids, pos_kwargs, timestep, target_timestep)
                if self.do_classifier_free_guidance:
                    neg_pred = predict(neg_embeds, neg_text_ids, neg_kwargs, timestep, None, use_cond=negative_use_cond)
                    noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()

        # 7. デコード
        image = self.decode_latents(latents, token_h, token_w, output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
