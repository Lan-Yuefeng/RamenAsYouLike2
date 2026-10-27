# Seg2Any の FluxRegionalPipeline (pipeline_flux.py) を FLUX.2 [klein] に移植したもの。
# FLUX.1 版からの主な変更:
#   - テキスト: T5 (系列) + CLIP (pooled) -> Qwen3 の hidden states 3 層 (9, 18, 27) を連結した 7680 次元。pooled は無い
#   - 各 caption は Qwen3 のチャットテンプレートで包んでから 1 本ずつエンコードし、パディングを除いて連結する
#   - txt_ids は (0, 0, 0, l)。l は global / 各 regional caption ごとに 0 から振り直す
#   - VAE: AutoencoderKLFlux2 (32ch)。2x2 パッチ化して 128ch にし、vae.bn の統計で正規化する
#   - cond 画像も同じ VAE エンコードを通す。位置 id は noisy latent と同じ (0, h, w, 0)
#   - スケジューラの shift は klein と同じ compute_empirical_mu
#   - CFG は klein と同じく「蒸留版なら無効、Base 版なら guidance_scale > 1 で true CFG」
# 領域アテンションマスクの作り方と cond トークンの間引きは FLUX.1 版の関数をそのまま使う。

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2KleinPipeline,
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput
from diffusers.utils import logging

from .pipeline_flux import FluxRegionalPipeline

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Flux2RegionalPipeline(Flux2KleinPipeline):
    # FLUX.1 版と共通の staticmethod
    prepare_attention_mask = staticmethod(FluxRegionalPipeline.prepare_attention_mask)
    filter_cond_token = staticmethod(FluxRegionalPipeline.filter_cond_token)

    def _wrap_chat_template(self, prompt: str, max_caption_tokens: int) -> str:
        # 長すぎる caption はテンプレートで包む前に caption 部分だけ切る (テンプレートの末尾を残すため)
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        if len(ids) > max_caption_tokens:
            logger.warning(
                f"caption truncated to {max_caption_tokens} tokens: {self.tokenizer.decode(ids[max_caption_tokens:])!r}"
            )
            prompt = self.tokenizer.decode(ids[:max_caption_tokens])
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @torch.no_grad()
    def _get_qwen3_prompt_embeds_nopad(
        self,
        prompts: List[str],
        max_caption_tokens: int,
        device: torch.device,
        hidden_states_layers=(9, 18, 27),
    ) -> List[torch.Tensor]:
        """各 prompt をチャットテンプレートで包んで Qwen3 に通し、パディングを除いた (L_i, 7680) のリストを返す。
        Qwen3 は causal なので右パディングは実トークンの hidden states に影響しない。"""
        texts = [self._wrap_chat_template(p, max_caption_tokens) for p in prompts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding="longest", padding_side="right")
        input_ids = inputs.input_ids.to(self.text_encoder.device)
        attention_mask = inputs.attention_mask.to(self.text_encoder.device)

        output = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
        )
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)  # B, 3, L, D
        b, c, l, d = out.shape
        # 学習時のテキスト専用パイプラインは transformer=None で作るので、その場合はエンコーダの dtype
        dtype = self.transformer.dtype if self.transformer is not None else self.text_encoder.dtype
        out = out.permute(0, 2, 1, 3).reshape(b, l, c * d).to(device=device, dtype=dtype)
        return [out[i][attention_mask[i].to(device) == 1] for i in range(b)]

    def encode_all_prompt(
        self,
        global_prompt: Optional[List[str]],
        regional_prompts: List[List[str]],
        global_max_sequence_length: int = 512,
        regional_max_sequence_length: int = 50,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        regional_text_position: str = "restart",
    ):
        """returns
        prompt_embeds: (B*N, S, 7680)。並びは [global, regional_1, ..., regional_n, zero pad]
        txt_seq_lens : list[list[int]]。[global 長, regional_1 長, ..., (pad 長)]。prepare_attention_mask 用
        text_ids     : (B*N, S, 4)。(0, 0, 0, l)
        regional_text_position:
            "restart"    -> l を global / 各 regional caption ごとに 0 から振り直す (既定)
            "continuous" -> l を系列全体の通し番号にする (比較実験用)
        """
        device = device or self._execution_device
        template_len = len(self.tokenizer(self._wrap_chat_template("", 10**6)).input_ids)

        if isinstance(regional_prompts, list) and isinstance(regional_prompts[0], str):
            regional_prompts = [regional_prompts]
        batch_size = len(regional_prompts)

        global_embeds = None
        if global_prompt is not None:
            global_embeds = self._get_qwen3_prompt_embeds_nopad(
                global_prompt, global_max_sequence_length - template_len, device
            )

        flat = [p for ps in regional_prompts for p in ps]
        flat_embeds = self._get_qwen3_prompt_embeds_nopad(flat, regional_max_sequence_length, device)
        offsets = np.cumsum([0] + [len(ps) for ps in regional_prompts])

        embeds, txt_seq_lens, positions = [], [], []
        for i in range(batch_size):
            segs = ([global_embeds[i]] if global_embeds is not None else []) + list(
                flat_embeds[offsets[i] : offsets[i + 1]]
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
        text_ids = torch.zeros(*l.shape, 4, dtype=torch.long)
        text_ids[..., 3] = l
        text_ids = text_ids.to(device)
        txt_seq_lens = [lens for lens in txt_seq_lens for _ in range(num_images_per_prompt)]
        return prompt_embeds, txt_seq_lens, text_ids

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
        negative_use_cond: bool = False,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        regional_max_sequence_length: int = 50,
    ):
        """引数の意味は FLUX.1 版 FluxRegionalPipeline.__call__ と同じ。違うのは以下。
        hard_attn_block_range: None なら transformer.default_hard_attn_block_range() (klein-4B は [5, 14])
        regional_text_position: encode_all_prompt を参照
        regional_max_sequence_length: caption 部分のトークン数上限 (チャットテンプレートの 12 トークンは含まない)
        guidance_scale: Base 版のときだけ有効 (> 1 で true CFG)。蒸留版では無視される
        negative_use_cond: CFG のネガティブ側にも cond トークン (輪郭) を入れるか。既定 False (2026-10-07 変更)。False だと
            CFG の差分 (positive - negative) に輪郭の情報が残り、guidance_scale 倍で強調される。
            val 55 枚で境界 recall 0.563 -> 0.772 (FLUX.1 は 0.808)、CLIP-T・KID は変わらず (result_f2/val55)
        """
        device = self._execution_device
        self._guidance_scale = guidance_scale
        self._attention_kwargs = dict(joint_attention_kwargs or {})
        self._interrupt = False

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

        # 1. テキスト
        prompt_embeds, txt_seq_lens, text_ids = self.encode_all_prompt(
            global_prompt, regional_prompts, max_sequence_length, regional_max_sequence_length,
            num_images_per_prompt, device, regional_text_position,
        )
        if self.do_classifier_free_guidance:
            neg_embeds, neg_txt_seq_lens, neg_text_ids = self.encode_all_prompt(
                [negative_prompt] * batch_size,
                [[negative_prompt] * len(ps) for ps in regional_prompts],
                max_sequence_length, regional_max_sequence_length, num_images_per_prompt, device,
                regional_text_position,
            )

        # 2. cond 画像 -> VAE -> トークン。位置 id は noisy latent と同じ空間座標 (cond_scale_factor 倍)
        cond_hidden_states = cond_ids = None
        cond_seq_lens = [0] * batch_size
        pad_seq_lens = [0] * batch_size
        if cond is not None:
            cond = self.image_processor.preprocess(cond, height=cond_height, width=cond_width)
            cond = cond.to(device=device, dtype=self.vae.dtype)
            assert len(cond) == batch_size
            cond_latents = self._encode_vae_image(cond, generator=None)  # B, 128, h, w
            cond_ids = self._prepare_latent_ids(cond_latents)[0].to(device)  # h*w, 4
            cond_ids[:, 1:3] *= cond_scale_factor
            cond_hidden_states = self._pack_latents(cond_latents).to(self.transformer.dtype)
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

        # 3. noisy latent
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )

        # 4. timesteps (klein と同じ)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if getattr(self.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        # 5. 領域アテンションマスク (FLUX.1 版と同じ関数)
        token_h, token_w = height // (self.vae_scale_factor * 2), width // (self.vae_scale_factor * 2)
        labels = [lab.clone().to(device) for lab in regional_labels for _ in range(num_images_per_prompt)]
        mask_kwargs = dict(
            attention_mask_method=attention_mask_method, regional_labels=labels, cond_seq_lens=cond_seq_lens,
            pad_seq_lens=pad_seq_lens, height=token_h, width=token_w,
            num_attention_heads=self.transformer.config.num_attention_heads, dtype=prompt_embeds.dtype,
            device=device, cond2image_attention_weight=cond2image_attention_weight,
        )
        attention_mask, hard_attention_mask = self.prepare_attention_mask(txt_seq_lens=txt_seq_lens, **mask_kwargs)
        pos_kwargs = {**self._attention_kwargs, "attention_mask": attention_mask, "hard_attention_mask": hard_attention_mask}
        if self.do_classifier_free_guidance:
            if not negative_use_cond:
                # ネガティブ側は cond トークンなし。系列に cond が無いのでマスクもその長さで作る
                mask_kwargs = {**mask_kwargs, "cond_seq_lens": [0] * len(labels), "pad_seq_lens": [0] * len(labels)}
            attention_mask, hard_attention_mask = self.prepare_attention_mask(txt_seq_lens=neg_txt_seq_lens, **mask_kwargs)
            neg_kwargs = {**self._attention_kwargs, "attention_mask": attention_mask, "hard_attention_mask": hard_attention_mask}

        # 6. denoising
        def predict(embeds, ids, kwargs, latents, timestep, use_cond=True):
            return self.transformer(
                hidden_states=latents.to(self.transformer.dtype),
                encoder_hidden_states=embeds,
                timestep=timestep / 1000,
                img_ids=latent_ids,
                txt_ids=ids,
                cond_hidden_states=cond_hidden_states if use_cond else None,
                cond_ids=cond_ids if use_cond else None,
                guidance=None,
                hard_attn_block_range=hard_attn_block_range,
                joint_attention_kwargs=kwargs,
                return_dict=False,
            )[0]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = predict(prompt_embeds, text_ids, pos_kwargs, latents, timestep)
                if self.do_classifier_free_guidance:
                    neg_pred = predict(neg_embeds, neg_text_ids, neg_kwargs, latents, timestep, use_cond=negative_use_cond)
                    noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()

        # 7. デコード (klein と同じ)
        latents = self._unpack_latents_with_ids(latents, latent_ids, token_h, token_w)
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = self._unpatchify_latents(latents * bn_std + bn_mean)
        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return Flux2PipelineOutput(images=image)
