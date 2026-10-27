# Seg2Any の条件トークン分岐と領域アテンションマスクを FLUX 3 Image Klein (Flux3Transformer2DModel) に移植したもの。
# FLUX.2 版 (transformer_flux2.py) と同じ考え方:
#   - 系列は [txt (global + regional + pad), noisy img, cond, cond pad] の順で、pipeline 側で作った
#     (B, H, S, S) の加算マスク (attention_mask / hard_attention_mask) をそのまま受け取る
#   - noisy img と txt は 'default' LoRA、cond トークンは 'cond' LoRA を通す (select_lora で切替)
#   - cond トークンは timestep=0 の modulation を使う
# FLUX.3 固有の点:
#   - 前半 (mode blocks) は txt / img / cond がそれぞれ別の重みで「自分のモード内だけ」で self-attention する。
#     cond は FLUX.3 が参照画像用に持つ image_cond ストリームにそのまま載せる。
#     マスクは全体マスクから各モードの対角ブロックを切り出して使う (regional caption 同士の混ざりを防ぐ)
#   - 後半 (single blocks) は [txt, img, cond] を結合し、重みは共有、modulation はモードごと
#   - ブロック番号 (hard_attn_block_range 用) は mode blocks が 0..num_layers-1、single blocks がその続き
# 重みのキー名は diffusers の Flux3Transformer2DModel と同一なので from_pretrained でそのまま読める。

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from diffusers.models.transformers.transformer_flux3 import (
    Flux3ModeTransformerBlock,
    Flux3SingleTransformerBlock,
    Flux3Transformer2DModel,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale

from src.models.transformer_flux2 import apply_rotary_emb, select_lora


def _linears(block):
    return (block.attn.to_q, block.attn.to_k, block.attn.to_v, block.attn.to_out[0], block.ff.linear_in,
            block.ff.linear_out)


def _qkv(attn, x, rotary_emb):
    """Flux3AttnProcessor と同じく QK norm と RoPE は fp32 で行い、最後に 1 回だけ戻す。"""
    query = attn.to_q(x).unflatten(-1, (attn.heads, -1))
    key = attn.to_k(x).unflatten(-1, (attn.heads, -1))
    value = attn.to_v(x).unflatten(-1, (attn.heads, -1))
    query = attn.norm_q(query.float())
    key = attn.norm_k(key.float())
    if rotary_emb is not None:
        query = apply_rotary_emb(query, rotary_emb)
        key = apply_rotary_emb(key, rotary_emb)
    return query.to(value.dtype), key.to(value.dtype), value


def _sdpa(query, key, value, attention_mask):
    # [B, S, H, D] -> SDPA の [B, H, S, D]。マスクは [B, H, S, S]
    out = F.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attention_mask
    )
    return out.transpose(1, 2).flatten(2, 3)


class Flux3RegionalModeTransformerBlock(Flux3ModeTransformerBlock):
    """モード内 self-attention のブロック。マスクと、どの LoRA を通すか (adapter) を受け取れるようにしたもの。"""

    def forward(self, hidden_states, temb_mod, image_rotary_emb, attention_mask=None, adapter="default"):
        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)
        norm_hidden_states = (1 + mod_scale) * self.norm(hidden_states) + mod_shift

        with select_lora(_linears(self), adapter):
            query, key, value = _qkv(self.attn, norm_hidden_states, image_rotary_emb)
            attn_output = _sdpa(query, key, value, attention_mask).to(query.dtype)
            attn_output = self.attn.to_out[1](self.attn.to_out[0](attn_output))
            ff_output = self.ff(norm_hidden_states)

        return hidden_states + mod_gate * (attn_output + ff_output)


class Flux3RegionalSingleTransformerBlock(Flux3SingleTransformerBlock):
    """[txt, img, cond] 結合の joint ブロック。txt+img は 'default'、cond は 'cond' LoRA で射影し、
    attention だけ 1 本の系列で取る。"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod: torch.Tensor,
        temb_mod_context: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cond_hidden_states: Optional[torch.Tensor] = None,
        temb_mod_cond: Optional[torch.Tensor] = None,
        cond_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        # image_rotary_emb は [txt, img] 結合済みの位置
        is_use_cond = cond_hidden_states is not None
        mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)
        c_mod_shift, c_mod_scale, c_mod_gate = torch.chunk(temb_mod_context, 3, dim=-1)

        norm_hidden_states = (1 + mod_scale) * self.norm(hidden_states) + mod_shift
        norm_encoder_hidden_states = (1 + c_mod_scale) * self.norm(encoder_hidden_states) + c_mod_shift
        combined = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)

        linears = _linears(self)
        with select_lora(linears, "default"):
            query, key, value = _qkv(self.attn, combined, image_rotary_emb)

        num_cond = 0
        if is_use_cond:
            num_cond = cond_hidden_states.shape[1]
            k_shift, k_scale, k_gate = torch.chunk(temb_mod_cond, 3, dim=-1)
            norm_cond_hidden_states = (1 + k_scale) * self.norm(cond_hidden_states) + k_shift
            with select_lora(linears, "cond"):
                cond_query, cond_key, cond_value = _qkv(self.attn, norm_cond_hidden_states, cond_rotary_emb)
            query = torch.cat([query, cond_query], dim=1)
            key = torch.cat([key, cond_key], dim=1)
            value = torch.cat([value, cond_value], dim=1)

        attn_output = _sdpa(query, key, value, attention_mask).to(query.dtype)
        attn_output, cond_attn_output = attn_output.split_with_sizes(
            [attn_output.shape[1] - num_cond, num_cond], dim=1
        )

        with select_lora(linears, "default"):
            output = self.attn.to_out[1](self.attn.to_out[0](attn_output)) + self.ff(combined)
        context_output, content_output = torch.split(
            output, [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim=1
        )
        encoder_hidden_states = encoder_hidden_states + c_mod_gate * context_output
        hidden_states = hidden_states + mod_gate * content_output

        if is_use_cond:
            with select_lora(linears, "cond"):
                cond_output = self.attn.to_out[1](self.attn.to_out[0](cond_attn_output)) + self.ff(
                    norm_cond_hidden_states
                )
            cond_hidden_states = cond_hidden_states + k_gate * cond_output

        return encoder_hidden_states, hidden_states, cond_hidden_states


class Flux3RegionalTransformer2DModel(Flux3Transformer2DModel):
    """Seg2Any 用の Flux3Transformer2DModel。

    ブロックを Regional 版に差し替えるだけでパラメータ構成は親クラスと同一。
    forward は cond トークン (image_cond ストリーム) と領域マスクを受け取る。
    """

    _no_split_modules = ["Flux3RegionalModeTransformerBlock", "Flux3RegionalSingleTransformerBlock"]
    _repeated_blocks = ["Flux3RegionalModeTransformerBlock", "Flux3RegionalSingleTransformerBlock"]

    # from_pretrained は __init__ のシグネチャを見て config を渡すので、*args/**kwargs にせず親と同じ引数を並べる
    def __init__(
        self,
        in_channels: int = 128,
        cond_in_channels: int = 128,
        num_layers: int = 5,
        num_single_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 20480,
        timestep_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 10000,
        eps: float = 1e-6,
        flow_map: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            cond_in_channels=cond_in_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            timestep_channels=timestep_channels,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=axes_dims_rope,
            rope_theta=rope_theta,
            eps=eps,
            flow_map=flow_map,
        )
        cfg = self.config
        block_kwargs = dict(
            dim=self.inner_dim,
            num_attention_heads=cfg.num_attention_heads,
            attention_head_dim=cfg.attention_head_dim,
            mlp_ratio=cfg.mlp_ratio,
            eps=cfg.eps,
        )
        for name in self.mode_blocks:
            self.mode_blocks[name] = torch.nn.ModuleList(
                [Flux3RegionalModeTransformerBlock(**block_kwargs) for _ in range(cfg.num_layers)]
            )
        self.txt_mode_blocks = torch.nn.ModuleList(
            [Flux3RegionalModeTransformerBlock(**block_kwargs) for _ in range(cfg.num_layers)]
        )
        self.single_transformer_blocks = torch.nn.ModuleList(
            [Flux3RegionalSingleTransformerBlock(**block_kwargs) for _ in range(cfg.num_single_layers)]
        )

    def default_hard_attn_block_range(self) -> List[int]:
        # FLUX.1 / FLUX.2 版と同じく「前半ブロックの直後から joint ブロックの前半まで」。
        # FLUX.3 klein (5 mode + 28 single) なら [5, 18]
        n_mode = self.config.num_layers
        return [n_mode, n_mode + self.config.num_single_layers // 2 - 1]

    def _rope(self, ids: torch.Tensor):
        if ids.ndim == 3:
            # cond (有効トークンがバッチごとに違う) と txt (領域構成がバッチごとに違う) は位置もバッチごと
            cos, sin = zip(*[self.pos_embed(ids_i) for ids_i in ids])
            return torch.stack(cos, dim=0), torch.stack(sin, dim=0)
        return self.pos_embed(ids)

    def _ckpt(self, block, *args):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            return self._gradient_checkpointing_func(block, *args)
        return block(*args)

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
        target_timestep: Optional[torch.Tensor] = None,
        hard_attn_block_range: Optional[List[int]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        joint_attention_kwargs = joint_attention_kwargs or {}
        attention_mask = joint_attention_kwargs.get("attention_mask", None)
        hard_attention_mask = joint_attention_kwargs.get("hard_attention_mask", None)
        if hard_attn_block_range is None:
            hard_attn_block_range = self.default_hard_attn_block_range()
        if (target_timestep is not None) != bool(self.config.flow_map):
            raise ValueError("`target_timestep` は flow_map (蒸留版) のときだけ、かつ必ず渡す")
        is_use_cond = cond_hidden_states is not None

        batch_size, num_img = hidden_states.shape[:2]
        num_txt = encoder_hidden_states.shape[1]
        num_cond = cond_hidden_states.shape[1] if is_use_cond else 0
        dtype = hidden_states.dtype

        # 1. timestep embedding。img は timestep、txt と cond は 0。蒸留版は飛び先の timestep を全モードに足す
        image_temb = self.timestep_embedder(self.time_proj(timestep.to(dtype) * 1000).to(dtype))[:, None, :]
        zero_temb = self.time_proj(torch.zeros(batch_size, device=hidden_states.device, dtype=dtype))
        zero_temb = self.timestep_embedder(zero_temb.to(dtype))[:, None, :]
        if target_timestep is not None:
            target_temb = self.time_proj(target_timestep.to(dtype) * 1000)
            target_temb = self.target_timestep_embedder(target_temb.to(dtype))[:, None, :]
            image_temb = image_temb + target_temb
            zero_temb = zero_temb + target_temb
        # modulation は全トークン同じ値なので (B, 1, D) のまま放送させる
        image_temb_act = F.silu(image_temb)
        zero_temb_act = F.silu(zero_temb)

        early, single = self.early_stream_modulations, self.single_stream_modulations
        with select_lora((early["txt"].linear, single["txt"].linear, early["image"].linear,
                          single["image"].linear), "default"):
            txt_early_mod = early["txt"](zero_temb_act)
            txt_single_mod = single["txt"](zero_temb_act)
            image_early_mod = early["image"](image_temb_act)
            image_single_mod = single["image"](image_temb_act)
        if is_use_cond:
            with select_lora((early["image_cond"].linear, single["image_cond"].linear), "cond"):
                cond_early_mod = early["image_cond"](zero_temb_act)
                cond_single_mod = single["image_cond"](zero_temb_act)

        # 2. 入力射影
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        with select_lora((self.x_embedders["image"],), "default"):
            hidden_states = self.x_embedders["image"](hidden_states)
        if is_use_cond:
            with select_lora((self.x_embedders["image_cond"],), "cond"):
                cond_hidden_states = self.x_embedders["image_cond"](cond_hidden_states)

        # 3. RoPE。img は全バッチ共通、cond はバッチごと。
        # txt は regional caption ごとに L を 0 から振り直すため、領域の数や長さが違えばバッチごとに異なる
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3 and bool((txt_ids == txt_ids[:1]).all()):
            txt_ids = txt_ids[0]
        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self._rope(txt_ids)
        cond_rotary_emb = self._rope(cond_ids) if is_use_cond else None
        if txt_ids.ndim == 3:
            joint_image_rotary_emb = tuple(e[None].expand(batch_size, -1, -1) for e in image_rotary_emb)
        else:
            joint_image_rotary_emb = image_rotary_emb
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], joint_image_rotary_emb[0]], dim=-2),
            torch.cat([text_rotary_emb[1], joint_image_rotary_emb[1]], dim=-2),
        )

        def mask_for(global_block_idx):
            if hard_attention_mask is not None and (
                hard_attn_block_range[0] <= global_block_idx <= hard_attn_block_range[1]
            ):
                return hard_attention_mask
            return attention_mask

        # 全体マスク [txt | img | cond] から各モードの対角ブロックを切り出す
        spans = {"txt": (0, num_txt), "image": (num_txt, num_txt + num_img),
                 "image_cond": (num_txt + num_img, num_txt + num_img + num_cond)}

        def sub_mask(mask, mode):
            if mask is None:
                return None
            s, e = spans[mode]
            return mask[:, :, s:e, s:e]

        # 4. mode blocks (各モードは自分のモード内だけで self-attention)
        for i in range(self.config.num_layers):
            mask = mask_for(i)
            encoder_hidden_states = self._ckpt(
                self.txt_mode_blocks[i], encoder_hidden_states, txt_early_mod, text_rotary_emb,
                sub_mask(mask, "txt"), "default",
            )
            hidden_states = self._ckpt(
                self.mode_blocks["image"][i], hidden_states, image_early_mod, image_rotary_emb,
                sub_mask(mask, "image"), "default",
            )
            if is_use_cond:
                cond_hidden_states = self._ckpt(
                    self.mode_blocks["image_cond"][i], cond_hidden_states, cond_early_mod, cond_rotary_emb,
                    sub_mask(mask, "image_cond"), "cond",
                )

        # 5. joint single blocks ([txt, img, cond])
        for j, block in enumerate(self.single_transformer_blocks):
            encoder_hidden_states, hidden_states, cond_hidden_states = self._ckpt(
                block, hidden_states, encoder_hidden_states, image_single_mod, txt_single_mod, concat_rotary_emb,
                mask_for(self.config.num_layers + j), cond_hidden_states,
                cond_single_mod if is_use_cond else None, cond_rotary_emb,
            )

        # 6. 出力 (img トークンのみ)
        hidden_states = self.norm_out(hidden_states, image_temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
