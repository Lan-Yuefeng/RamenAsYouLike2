"""案 1 (a)：テキストとレイアウト (cond) に別々の強さを掛ける CFG。

3 つの予測を使う：
    e_uu = (ネガティブ文, cond なし)
    e_tu = (ポジティブ文, cond なし)
    e_tc = (ポジティブ文, cond あり)
    e = e_uu + s_txt * (e_tu - e_uu) + s_lay(t) * (e_tc - e_tu)

s_txt = s_lay = s のとき e = e_uu + s * (e_tc - e_uu) となり、今の既定 (negative_use_cond=False) と完全に一致する。
その場合は e_tu を計算しない (時間も既定と同じ)。
s_lay は「最初の lay_switch 割合のステップは s_lay、それ以降は s_lay_late」と 2 段階に変えられる。

Flux3RegionalPipeline.__call__ のうち、CFG の部分だけを書き換えた版。それ以外の処理は同じ。
"""
import os
import sys

import numpy as np
import torch
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipelines.pipeline_flux3 import TEXT_PAD_MULTIPLE, Flux3RegionalPipeline, prepare_image_ids  # noqa: E402


class Flux3DecoupledGuidancePipeline(Flux3RegionalPipeline):
    @torch.no_grad()
    def generate(
        self,
        global_prompt,
        regional_prompts,
        regional_labels,
        cond,
        s_txt=3.5,
        s_lay=3.5,
        s_lay_late=None,
        lay_switch=1.0,
        negative_prompt="",
        attention_mask_method="hard",
        hard_attn_block_range=None,
        cond_scale_factor=1,
        cond2image_attention_weight=1,
        regional_text_position="restart",
        text_pad_multiple=TEXT_PAD_MULTIPLE,
        height=None,
        width=None,
        num_inference_steps=50,
        generator=None,
        max_sequence_length=512,
        regional_max_sequence_length=50,
    ):
        assert not self.is_distilled, "Base 版専用"
        device = self._execution_device
        s_lay_late = s_lay if s_lay_late is None else s_lay_late
        need_tu = not (s_txt == s_lay == s_lay_late)

        if isinstance(global_prompt, str):
            global_prompt = [global_prompt]
        if isinstance(regional_prompts[0], str):
            regional_prompts = [regional_prompts]
        if isinstance(regional_labels, torch.Tensor):
            regional_labels = [regional_labels]
        batch_size = len(regional_prompts)

        multiple = self.vae_scale_factor * 2 * cond_scale_factor
        height, width = int(height) // multiple * multiple, int(width) // multiple * multiple
        cond_height, cond_width = height // cond_scale_factor, width // cond_scale_factor
        token_h, token_w = height // (self.vae_scale_factor * 2), width // (self.vae_scale_factor * 2)

        # 1. テキスト
        text_kwargs = dict(
            global_max_sequence_length=max_sequence_length, regional_max_sequence_length=regional_max_sequence_length,
            num_images_per_prompt=1, device=device,
            regional_text_position=regional_text_position, text_pad_multiple=text_pad_multiple,
        )
        prompt_embeds, txt_seq_lens, text_ids = self.encode_all_prompt(global_prompt, regional_prompts, **text_kwargs)
        neg_embeds, neg_txt_seq_lens, neg_text_ids = self.encode_all_prompt(
            [negative_prompt] * batch_size, [[negative_prompt] * len(ps) for ps in regional_prompts], **text_kwargs
        )

        # 2. cond
        cond = self.image_processor.preprocess(cond, height=cond_height, width=cond_width)
        cond = cond.to(device=device, dtype=self.vae.dtype)
        cond_latents = self.encode_vae_image(cond)
        cond_ids = prepare_image_ids(*cond_latents.shape[-2:], l_coord=-1, pos_scale=cond_scale_factor).to(device)
        cond_hidden_states = self.pack_latents(cond_latents).to(self.transformer.dtype)
        cond_hidden_states, cond_ids, cond_seq_lens, pad_seq_lens = self.filter_cond_token(
            cond, cond_hidden_states, cond_ids, vae_scale_factor=self.vae_scale_factor * 2
        )

        # 3. noisy latent
        dtype = self.transformer.dtype
        shape = (batch_size, self.transformer.config.in_channels, token_h, token_w)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32).to(dtype)
        latents = self.pack_latents(latents)
        latent_ids = prepare_image_ids(token_h, token_w).to(device)

        # 4. timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.scheduler.timesteps
        self.scheduler.set_begin_index(0)

        # 5. マスク：tc (cond あり)、tu / uu (cond なし)
        labels = [lab.clone().to(device) for lab in regional_labels]
        mask_kwargs = dict(
            attention_mask_method=attention_mask_method, regional_labels=labels, cond_seq_lens=cond_seq_lens,
            pad_seq_lens=pad_seq_lens, height=token_h, width=token_w,
            num_attention_heads=self.transformer.config.num_attention_heads, dtype=prompt_embeds.dtype,
            device=device, cond2image_attention_weight=cond2image_attention_weight,
        )
        nocond = {**mask_kwargs, "cond_seq_lens": [0] * len(labels), "pad_seq_lens": [0] * len(labels)}

        def kw(txt_lens, mk):
            am, ham = self.prepare_attention_mask(txt_seq_lens=txt_lens, **mk)
            return {"attention_mask": am, "hard_attention_mask": ham}

        kw_tc = kw(txt_seq_lens, mask_kwargs)
        kw_tu = kw(txt_seq_lens, nocond) if need_tu else None
        kw_uu = kw(neg_txt_seq_lens, nocond)

        def predict(embeds, ids, kwargs, timestep, use_cond):
            return self.transformer(
                hidden_states=latents.to(dtype), encoder_hidden_states=embeds, timestep=timestep,
                img_ids=latent_ids, txt_ids=ids,
                cond_hidden_states=cond_hidden_states if use_cond else None,
                cond_ids=cond_ids if use_cond else None,
                target_timestep=None, hard_attn_block_range=hard_attn_block_range,
                joint_attention_kwargs=kwargs, return_dict=False,
            )[0]

        n_early = int(round(lay_switch * len(timesteps)))
        for i, t in enumerate(timesteps):
            timestep = (t / 1000).expand(latents.shape[0]).to(dtype)
            e_tc = predict(prompt_embeds, text_ids, kw_tc, timestep, True)
            e_uu = predict(neg_embeds, neg_text_ids, kw_uu, timestep, False)
            if need_tu:
                sl = s_lay if i < n_early else s_lay_late
                e_tu = predict(prompt_embeds, text_ids, kw_tu, timestep, False)
                noise_pred = e_uu + s_txt * (e_tu - e_uu) + sl * (e_tc - e_tu)
            else:
                noise_pred = e_uu + s_txt * (e_tc - e_uu)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        image = self.decode_latents(latents, token_h, token_w, "pil")
        return ImagePipelineOutput(images=image)
