# Seg2Any の条件トークン分岐と領域アテンションマスクを FLUX.2 (Flux2Transformer2DModel) に移植したもの。
# FLUX.1 版 (transformer_flux.py / attention.py) と同じ考え方:
#   - 系列は [txt (global + regional), noisy img, cond, cond pad] の順に結合して 1 回の SDPA に通す
#   - attention_mask / hard_attention_mask は pipeline 側で作った (B, H, S, S) の加算マスクをそのまま使う
#   - noisy img トークンは 'default' LoRA、cond トークンは 'cond' LoRA を通す (select_lora で切替)
#   - cond トークンは timestep=0 の modulation を使う
# 重みのキー名は diffusers の Flux2Transformer2DModel と同一なので from_pretrained でそのまま読める。

from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer

from diffusers.models.transformers.transformer_flux2 import (
    Flux2Modulation,
    Flux2SingleTransformerBlock,
    Flux2Transformer2DModel,
    Flux2Transformer2DModelOutput,
    Flux2TransformerBlock,
)
from diffusers.utils import apply_lora_scale


class select_lora:
    """utils.lora_controller.select_lora と同じだが、抜けるときに元の scaling に戻す。
    (元実装は 1 に戻すので lora_alpha != rank や lora_scale != 1 のときに値が壊れる)"""

    def __init__(self, lora_modules, adapter_name: str) -> None:
        self.adapter_name = adapter_name
        self.lora_modules: List[BaseTunerLayer] = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
        self.saved = []

    def __enter__(self) -> None:
        for m in self.lora_modules:
            for name in m.active_adapters:
                if name != self.adapter_name and name in m.scaling:
                    self.saved.append((m, name, m.scaling[name]))
                    m.scaling[name] = 0

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val, exc_tb) -> None:
        for m, name, value in self.saved:
            m.scaling[name] = value
        self.saved = []


def apply_rotary_emb(x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """x: [B, S, H, D]。cos/sin は [S, D] (全バッチ共通) または [B, S, D] (cond のようにバッチごとに異なる位置)。"""
    cos, sin = freqs_cis
    if cos.ndim == 2:
        cos, sin = cos[None, :, None], sin[None, :, None]
    else:
        cos, sin = cos[:, :, None], sin[:, :, None]
    cos, sin = cos.to(x.device), sin.to(x.device)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _sdpa(query, key, value, attention_mask):
    # Flux2 の処理系は [B, S, H, D]、マスクは [B, H, S, S] なので SDPA 用に転置する
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attention_mask
    )
    return out.transpose(1, 2)


class Flux2RegionalAttnProcessor:
    """double-stream ブロック用。Flux2AttnProcessor に cond トークンを足したもの。"""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond_hidden_states: Optional[torch.Tensor] = None,
        cond_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        with select_lora((attn.to_q, attn.to_k, attn.to_v), "default"):
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (-1, attn.head_dim)))
        key = attn.norm_k(key.unflatten(-1, (-1, attn.head_dim)))
        value = value.unflatten(-1, (-1, attn.head_dim))

        encoder_query = attn.norm_added_q(attn.add_q_proj(encoder_hidden_states).unflatten(-1, (-1, attn.head_dim)))
        encoder_key = attn.norm_added_k(attn.add_k_proj(encoder_hidden_states).unflatten(-1, (-1, attn.head_dim)))
        encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (-1, attn.head_dim))

        query = torch.cat([encoder_query, query], dim=1)
        key = torch.cat([encoder_key, key], dim=1)
        value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        num_cond = 0
        if cond_hidden_states is not None:
            num_cond = cond_hidden_states.shape[1]
            with select_lora((attn.to_q, attn.to_k, attn.to_v), "cond"):
                cond_query = attn.to_q(cond_hidden_states)
                cond_key = attn.to_k(cond_hidden_states)
                cond_value = attn.to_v(cond_hidden_states)
            cond_query = attn.norm_q(cond_query.unflatten(-1, (-1, attn.head_dim)))
            cond_key = attn.norm_k(cond_key.unflatten(-1, (-1, attn.head_dim)))
            cond_value = cond_value.unflatten(-1, (-1, attn.head_dim))
            if cond_rotary_emb is not None:
                cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
                cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)
            query = torch.cat([query, cond_query], dim=1)
            key = torch.cat([key, cond_key], dim=1)
            value = torch.cat([value, cond_value], dim=1)

        out = _sdpa(query, key, value, attention_mask).flatten(2, 3).to(query.dtype)

        num_txt = encoder_hidden_states.shape[1]
        encoder_out, img_out, cond_out = out.split_with_sizes(
            [num_txt, out.shape[1] - num_txt - num_cond, num_cond], dim=1
        )
        encoder_out = attn.to_add_out(encoder_out)
        with select_lora((attn.to_out[0],), "default"):
            img_out = attn.to_out[1](attn.to_out[0](img_out))
        if cond_hidden_states is not None:
            with select_lora((attn.to_out[0],), "cond"):
                cond_out = attn.to_out[1](attn.to_out[0](cond_out))
        else:
            cond_out = None
        return img_out, encoder_out, cond_out


class Flux2RegionalParallelSelfAttnProcessor:
    """single-stream ブロック用。QKV+MLP 入力 (to_qkv_mlp_proj) と出力 (to_out) が融合しているので、
    cond はその 2 つの Linear を 'cond' LoRA で通す。"""

    @staticmethod
    def _project(attn, x, adapter):
        with select_lora((attn.to_qkv_mlp_proj,), adapter):
            x = attn.to_qkv_mlp_proj(x)
        qkv, mlp = torch.split(x, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1)
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (-1, attn.head_dim)))
        key = attn.norm_k(key.unflatten(-1, (-1, attn.head_dim)))
        value = value.unflatten(-1, (-1, attn.head_dim))
        return query, key, value, mlp

    @staticmethod
    def _out(attn, attn_out, mlp, adapter):
        with select_lora((attn.to_out,), adapter):
            return attn.to_out(torch.cat([attn_out, attn.mlp_act_fn(mlp)], dim=-1))

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cond_hidden_states: Optional[torch.Tensor] = None,
        cond_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # hidden_states は [txt, img] 結合済み。FLUX.1 版と同じく txt も 'default' を通す
        query, key, value, mlp = self._project(attn, hidden_states, "default")
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        num_cond = 0
        if cond_hidden_states is not None:
            num_cond = cond_hidden_states.shape[1]
            cond_query, cond_key, cond_value, cond_mlp = self._project(attn, cond_hidden_states, "cond")
            if cond_rotary_emb is not None:
                cond_query = apply_rotary_emb(cond_query, cond_rotary_emb)
                cond_key = apply_rotary_emb(cond_key, cond_rotary_emb)
            query = torch.cat([query, cond_query], dim=1)
            key = torch.cat([key, cond_key], dim=1)
            value = torch.cat([value, cond_value], dim=1)

        out = _sdpa(query, key, value, attention_mask).flatten(2, 3).to(query.dtype)
        out, cond_out = out.split_with_sizes([out.shape[1] - num_cond, num_cond], dim=1)

        hidden_states = self._out(attn, out, mlp, "default")
        cond_hidden_states = self._out(attn, cond_out, cond_mlp, "cond") if num_cond > 0 else None
        return hidden_states, cond_hidden_states


class Flux2RegionalTransformerBlock(Flux2TransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn.set_processor(Flux2RegionalAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_img: torch.Tensor,
        temb_mod_txt: torch.Tensor,
        image_rotary_emb=None,
        attention_mask=None,
        cond_hidden_states=None,
        temb_mod_cond=None,
        cond_rotary_emb=None,
    ):
        is_use_cond = cond_hidden_states is not None

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(
            temb_mod_txt, 2
        )

        norm_hidden_states = (1 + scale_msa) * self.norm1(hidden_states) + shift_msa
        norm_encoder_hidden_states = (1 + c_scale_msa) * self.norm1_context(encoder_hidden_states) + c_shift_msa

        norm_cond_hidden_states = None
        if is_use_cond:
            (k_shift_msa, k_scale_msa, k_gate_msa), (k_shift_mlp, k_scale_mlp, k_gate_mlp) = Flux2Modulation.split(
                temb_mod_cond, 2
            )
            norm_cond_hidden_states = (1 + k_scale_msa) * self.norm1(cond_hidden_states) + k_shift_msa

        attn_output, context_attn_output, cond_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            cond_hidden_states=norm_cond_hidden_states,
            cond_rotary_emb=cond_rotary_emb,
        )

        # img stream
        hidden_states = hidden_states + gate_msa * attn_output
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        with select_lora((self.ff.linear_in, self.ff.linear_out), "default"):
            ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp * ff_output

        # txt stream
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp) + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(norm_encoder_hidden_states)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        # cond stream (img stream と重みを共有し、LoRA だけ 'cond' に切り替える)
        if is_use_cond:
            cond_hidden_states = cond_hidden_states + k_gate_msa * cond_attn_output
            norm_cond_hidden_states = self.norm2(cond_hidden_states) * (1 + k_scale_mlp) + k_shift_mlp
            with select_lora((self.ff.linear_in, self.ff.linear_out), "cond"):
                cond_ff_output = self.ff(norm_cond_hidden_states)
            cond_hidden_states = cond_hidden_states + k_gate_mlp * cond_ff_output

        return encoder_hidden_states, hidden_states, cond_hidden_states


class Flux2RegionalSingleTransformerBlock(Flux2SingleTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn.set_processor(Flux2RegionalParallelSelfAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        image_rotary_emb=None,
        attention_mask=None,
        cond_hidden_states=None,
        temb_mod_cond=None,
        cond_rotary_emb=None,
    ):
        # hidden_states は [txt, img] 結合済み
        is_use_cond = cond_hidden_states is not None

        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]
        norm_hidden_states = (1 + mod_scale) * self.norm(hidden_states) + mod_shift

        norm_cond_hidden_states = None
        if is_use_cond:
            k_shift, k_scale, k_gate = Flux2Modulation.split(temb_mod_cond, 1)[0]
            norm_cond_hidden_states = (1 + k_scale) * self.norm(cond_hidden_states) + k_shift

        attn_output, cond_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            cond_hidden_states=norm_cond_hidden_states,
            cond_rotary_emb=cond_rotary_emb,
        )

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if is_use_cond:
            cond_hidden_states = cond_hidden_states + k_gate * cond_attn_output

        return hidden_states, cond_hidden_states


class Flux2RegionalTransformer2DModel(Flux2Transformer2DModel):
    """Seg2Any 用の Flux2Transformer2DModel。

    ブロックを Regional 版に差し替えるだけでパラメータ構成は親クラスと同一。
    forward は KV キャッシュ (参照画像編集用) を落とし、cond トークンと領域マスクを受け取る。
    """

    _no_split_modules = ["Flux2RegionalTransformerBlock", "Flux2RegionalSingleTransformerBlock"]
    _repeated_blocks = ["Flux2RegionalTransformerBlock", "Flux2RegionalSingleTransformerBlock"]
    _cp_plan = None
    _tp_plan = None

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            timestep_guidance_channels=timestep_guidance_channels,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            eps=eps,
            guidance_embeds=guidance_embeds,
        )
        block_kwargs = dict(
            dim=self.inner_dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
            eps=eps,
            bias=False,
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [Flux2RegionalTransformerBlock(**block_kwargs) for _ in range(num_layers)]
        )
        self.single_transformer_blocks = torch.nn.ModuleList(
            [Flux2RegionalSingleTransformerBlock(**block_kwargs) for _ in range(num_single_layers)]
        )

    def default_hard_attn_block_range(self) -> List[int]:
        # FLUX.1 版の既定 [19, 37] は「double 全 19 層の直後から single の前半 19/38 層」。
        # 同じ比率で klein-4B (5 double + 20 single) なら [5, 14] になる。
        n_double = self.config.num_layers
        return [n_double, n_double + self.config.num_single_layers // 2 - 1]

    def _rope(self, ids: torch.Tensor):
        if ids.ndim == 3:
            # cond (有効トークンがバッチごとに違う) と txt (領域構成がバッチごとに違う) は位置もバッチごと
            cos, sin = zip(*[self.pos_embed(ids_i) for ids_i in ids])
            return torch.stack(cos, dim=0), torch.stack(sin, dim=0)
        return self.pos_embed(ids)

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        cond_hidden_states: Optional[torch.Tensor] = None,
        cond_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        hard_attn_block_range: Optional[List[int]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        joint_attention_kwargs = joint_attention_kwargs or {}
        attention_mask = joint_attention_kwargs.get("attention_mask", None)
        hard_attention_mask = joint_attention_kwargs.get("hard_attention_mask", None)
        if hard_attn_block_range is None:
            hard_attn_block_range = self.default_hard_attn_block_range()
        is_use_cond = cond_hidden_states is not None

        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. timestep embedding と modulation。cond トークンは t=0 (ノイズなし) として扱う
        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)

        mod_linears = (self.double_stream_modulation_img.linear, self.single_stream_modulation.linear)
        with select_lora(mod_linears, "default"):
            double_stream_mod_img = self.double_stream_modulation_img(temb)
            single_stream_mod = self.single_stream_modulation(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)

        double_stream_mod_cond = single_stream_mod_cond = None
        if is_use_cond:
            temb_cond = self.time_guidance_embed(torch.zeros_like(timestep), guidance)
            with select_lora(mod_linears, "cond"):
                double_stream_mod_cond = self.double_stream_modulation_img(temb_cond)
                single_stream_mod_cond = self.single_stream_modulation(temb_cond)

        # 2. 入力射影
        with select_lora((self.x_embedder,), "default"):
            hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if is_use_cond:
            with select_lora((self.x_embedder,), "cond"):
                cond_hidden_states = self.x_embedder(cond_hidden_states)

        # 3. RoPE。img は全バッチ共通、cond はバッチごと。
        # txt は regional caption ごとに L を 0 から振り直すため、領域の数や長さが違えばバッチごとに異なる
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3 and bool((txt_ids == txt_ids[:1]).all()):
            txt_ids = txt_ids[0]
        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self._rope(txt_ids)
        if txt_ids.ndim == 3:
            batch = txt_ids.shape[0]
            image_rotary_emb = tuple(e[None].expand(batch, -1, -1) for e in image_rotary_emb)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=-2),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=-2),
        )
        cond_rotary_emb = self._rope(cond_ids) if is_use_cond else None

        def mask_for(global_block_idx):
            if hard_attention_mask is not None and (
                hard_attn_block_range[0] <= global_block_idx <= hard_attn_block_range[1]
            ):
                return hard_attention_mask
            return attention_mask

        # 4. double-stream
        for index_block, block in enumerate(self.transformer_blocks):
            args = (
                hidden_states,
                encoder_hidden_states,
                double_stream_mod_img,
                double_stream_mod_txt,
                concat_rotary_emb,
                mask_for(index_block),
                cond_hidden_states,
                double_stream_mod_cond,
                cond_rotary_emb,
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states, cond_hidden_states = self._gradient_checkpointing_func(
                    block, *args
                )
            else:
                encoder_hidden_states, hidden_states, cond_hidden_states = block(*args)

        # 5. single-stream ([txt, img] を結合)
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for index_block, block in enumerate(self.single_transformer_blocks):
            args = (
                hidden_states,
                single_stream_mod,
                concat_rotary_emb,
                mask_for(index_block + self.config.num_layers),
                cond_hidden_states,
                single_stream_mod_cond,
                cond_rotary_emb,
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, cond_hidden_states = self._gradient_checkpointing_func(block, *args)
            else:
                hidden_states, cond_hidden_states = block(*args)

        hidden_states = hidden_states[:, num_txt_tokens:]

        # 6. 出力
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Flux2Transformer2DModelOutput(sample=output)
