"""学習なしの高速化 2 つ (src/ は触らず、モンキーパッチで入れる)。

① fuse_default_lora: 'default' LoRA を base の重みに足し込む。
   'default' を通す呼び出しは peft を無効にして base だけ (= 融合済み) を通す。
   'cond' を通す呼び出しは default の scaling を -s にして差し引き、cond を +s で足す。
③ CachedForward: joint (single) blocks で txt / cond の行が img の列を見ないようにし (一方向化)、
   txt / cond の modulation を飛び先 timestep=0 の定数に固定する。すると txt / cond の隠れ状態は
   ステップに依らないので、1 ステップ目に各 single block の txt / cond の K/V を貯め、
   2 ステップ目以降は img トークンだけを計算する。
   品質は probe_oneway.py の oneway_both_fixtemb と同じ (学習なしなので崩れる分はそのまま)。
② CachedForward(cache_txt=False): cond だけ一方向化 (cond 行 → img 列を遮断。txt 列はもともと見ない) し、
   cond の modulation を定数に固定して cond の K/V だけ使い回す (OminiControl2 と同じ)。txt と img は毎ステップ計算。
"""
from typing import Optional

import torch
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer

import src.models.transformer_flux3 as T
from src.models.transformer_flux3 import _linears, _qkv, _sdpa


# ---------------------------------------------------------------- ① LoRA 融合
class fused_select_lora:
    """fuse_default_lora 後に transformer_flux3.select_lora と差し替える版。"""

    def __init__(self, lora_modules, adapter_name: str) -> None:
        self.adapter_name = adapter_name
        self.lora_modules = [m for m in lora_modules if isinstance(m, BaseTunerLayer)]
        self.saved = []

    def __enter__(self) -> None:
        if self.adapter_name == "default":
            return  # 既定状態 (peft 無効 = base に融合済みの default だけ) のまま
        for m in self.lora_modules:
            self.saved.append((m, m._disable_adapters, dict(m.scaling)))
            m._disable_adapters = False
            for name in m.active_adapters:
                if name not in m.scaling:
                    continue
                if name == self.adapter_name:
                    m.scaling[name] = m._orig_scaling[name]
                elif name == "default" and m._fused_default:
                    m.scaling[name] = -m._orig_scaling[name]
                else:
                    m.scaling[name] = 0

    def __exit__(self, *exc) -> None:
        for m, disabled, scaling in self.saved:
            m._disable_adapters = disabled
            m.scaling.update(scaling)
        self.saved = []


@torch.no_grad()
def fuse_default_lora(transformer) -> int:
    n = 0
    for m in transformer.modules():
        if not isinstance(m, BaseTunerLayer):
            continue
        m._orig_scaling = dict(m.scaling)
        m._fused_default = "default" in m.lora_A
        if m._fused_default:
            A = m.lora_A["default"].weight.float()
            B = m.lora_B["default"].weight.float()
            w = m.base_layer.weight
            w.copy_((w.float() + (B @ A) * m.scaling["default"]).to(w.dtype))
            n += 1
        m._disable_adapters = True
    T.select_lora = fused_select_lora
    return n


# ---------------------------------------------------------------- ②③ 一方向化 + K/V キャッシュ
def _oneway(mask: Optional[torch.Tensor], n_txt: int, n_img: int, cache_txt: bool = True):
    """cond 行 (cache_txt なら txt 行も) から img 列への注意を遮断する。全 head 共通なので 1 枚だけ書き換える。
    cond 行はもともと txt 列を見ない (Seg2Any のマスク) ので、cond だけなら img 列を塞げば自分以外に依らなくなる。"""
    if mask is None:
        return None
    m = mask[:, :1].clone()
    if cache_txt:
        m[:, :, :n_txt, n_txt:n_txt + n_img] = float("-inf")
    m[:, :, n_txt + n_img:, n_txt:n_txt + n_img] = float("-inf")
    return m


class CachedForward:
    """pipe.transformer.forward と差し替えて使う。生成のたびに reset() する。
    cache_txt=True  : ③ txt と cond を一方向化して使い回す。2 ステップ目以降は img だけ計算
    cache_txt=False : ② cond だけ一方向化して使い回す (OminiControl2 と同じ)。txt は今まで通り毎ステップ計算
    use_cache=False なら毎ステップ全部計算する (一方向化 + 定数 modulation だけ。キャッシュの検算用・自己蒸留の生徒用)。
    ckpt=True ならブロックごとに gradient checkpointing する (学習用。勾配は呼び出し側の grad モードに従う)。"""

    def __init__(self, tr, use_cache: bool = True, cache_txt: bool = True, ckpt: bool = False):
        self.tr = tr
        self.use_cache = use_cache
        self.cache_txt = cache_txt
        self.ckpt = ckpt
        self.reset()

    def _run(self, fn, *args):
        if self.ckpt and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def reset(self):
        self.kv = None  # single block ごとの (k_txt, v_txt, k_cond, v_cond)。② では txt 側は None
        self.static = None  # 1 ステップ目に作るマスクなど

    def __call__(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids,
                 cond_hidden_states=None, cond_ids=None, target_timestep=None, hard_attn_block_range=None,
                 joint_attention_kwargs=None, return_dict=False):
        tr = self.tr
        assert cond_hidden_states is not None and target_timestep is not None, "蒸留版 + cond のみ対応"
        rng = hard_attn_block_range or tr.default_hard_attn_block_range()
        b, num_img = hidden_states.shape[:2]
        num_txt = encoder_hidden_states.shape[1]
        dtype = hidden_states.dtype
        write = self.kv is None or not self.use_cache
        n_mode = tr.config.num_layers

        def temb(embedder, t):
            return embedder(tr.time_proj(t.to(dtype) * 1000).to(dtype))[:, None, :]

        zero = torch.zeros(b, device=hidden_states.device, dtype=dtype)
        image_temb = temb(tr.timestep_embedder, timestep) + temb(tr.target_timestep_embedder, target_timestep)
        # txt / cond の modulation: cond は timestep=0・飛び先 0 の定数。txt は ③ なら同じ定数、② なら今まで通り (飛び先は今のステップ)
        fixed_act = F.silu(temb(tr.timestep_embedder, zero) + temb(tr.target_timestep_embedder, zero))
        txt_act = fixed_act if self.cache_txt else F.silu(
            temb(tr.timestep_embedder, zero) + temb(tr.target_timestep_embedder, target_timestep))
        early, single = tr.early_stream_modulations, tr.single_stream_modulations
        with T.select_lora((early["image"].linear, single["image"].linear), "default"):
            image_early_mod = early["image"](F.silu(image_temb))
            image_single_mod = single["image"](F.silu(image_temb))
        with T.select_lora((tr.x_embedders["image"],), "default"):
            h = tr.x_embedders["image"](hidden_states)
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        img_rope = tr.pos_embed(img_ids)

        run_txt = write or not self.cache_txt
        if run_txt:
            with T.select_lora((early["txt"].linear, single["txt"].linear), "default"):
                txt_early_mod = early["txt"](txt_act)
                txt_single_mod = single["txt"](txt_act)
            e = tr.context_embedder(encoder_hidden_states)
            if txt_ids.ndim == 3 and bool((txt_ids == txt_ids[:1]).all()):
                txt_ids = txt_ids[0]
            txt_rope = tr._rope(txt_ids)
            joint_img_rope = tuple(x[None].expand(b, -1, -1) for x in img_rope) if txt_ids.ndim == 3 else img_rope
            concat_rope = (torch.cat([txt_rope[0], joint_img_rope[0]], dim=-2),
                           torch.cat([txt_rope[1], joint_img_rope[1]], dim=-2))

        if write:
            with T.select_lora((early["image_cond"].linear, single["image_cond"].linear), "cond"):
                cond_early_mod = early["image_cond"](fixed_act)
                cond_single_mod = single["image_cond"](fixed_act)
            with T.select_lora((tr.x_embedders["image_cond"],), "cond"):
                c = tr.x_embedders["image_cond"](cond_hidden_states)
            cond_rope = tr._rope(cond_ids)
            kw = joint_attention_kwargs or {}
            masks = (kw.get("attention_mask"), kw.get("hard_attention_mask"))
            if masks[1] is None:
                masks = (masks[0], masks[0])
            s_t, s_i = (0, num_txt), (num_txt, num_txt + num_img)
            s_c = (num_txt + num_img, num_txt + num_img + c.shape[1])
            sub = lambda m, s: None if m is None else m[:, :, s[0]:s[1], s[0]:s[1]]
            # 2 ステップ目以降に計算する行 (③ は img、② は txt+img)。これらの行は一方向化で変わらないので元のマスクから切る
            r0 = s_i[0] if self.cache_txt else 0
            self.static = {
                "sub": {k: tuple(sub(m, s) for m in masks) for k, s in (("txt", s_t), ("image", s_i), ("cond", s_c))},
                "rows": tuple(None if m is None else m[:, :1, r0:s_i[1], :].contiguous() for m in masks),
                "oneway": tuple(_oneway(m, num_txt, num_img, self.cache_txt) for m in masks),
            }
        st = self.static

        # mode blocks (モード内だけの self-attention)
        for i in range(n_mode):
            mi = self._mask_idx(i, rng)
            if run_txt:
                e = self._run(tr.txt_mode_blocks[i], e, txt_early_mod, txt_rope, st["sub"]["txt"][mi], "default")
            h = self._run(tr.mode_blocks["image"][i], h, image_early_mod, img_rope, st["sub"]["image"][mi], "default")
            if write:
                c = self._run(tr.mode_blocks["image_cond"][i], c, cond_early_mod, cond_rope, st["sub"]["cond"][mi], "cond")

        # single blocks
        kv = []
        for j, block in enumerate(tr.single_transformer_blocks):
            mi = self._mask_idx(n_mode + j, rng)
            if write:
                e, h, c, cache = self._run(self._single_full, block, h, e, c, image_single_mod, txt_single_mod, cond_single_mod,
                                                   concat_rope, cond_rope, st["oneway"][mi])
                if not self.cache_txt:
                    cache = (None, None) + cache[2:]
                kv.append(cache)
            elif self.cache_txt:
                h = self._single_img(block, h, image_single_mod, img_rope, self.kv[j], st["rows"][mi])
            else:
                e, h = self._single_txtimg(block, h, e, image_single_mod, txt_single_mod, concat_rope, self.kv[j],
                                           st["rows"][mi])
        if write:
            self.kv = kv

        out = tr.proj_out(tr.norm_out(h, image_temb))
        return (out,)

    @staticmethod
    def _mask_idx(global_idx, rng):
        return 1 if rng[0] <= global_idx <= rng[1] else 0

    @staticmethod
    def _single_full(block, h, e, c, mod, mod_txt, mod_cond, concat_rope, cond_rope, mask):
        """Flux3RegionalSingleTransformerBlock.forward と同じ計算で、txt / cond の K/V も返す。"""
        shift, scale, gate = torch.chunk(mod, 3, dim=-1)
        t_shift, t_scale, t_gate = torch.chunk(mod_txt, 3, dim=-1)
        k_shift, k_scale, k_gate = torch.chunk(mod_cond, 3, dim=-1)
        nh = (1 + scale) * block.norm(h) + shift
        ne = (1 + t_scale) * block.norm(e) + t_shift
        nc = (1 + k_scale) * block.norm(c) + k_shift
        combined = torch.cat([ne, nh], dim=1)
        linears = _linears(block)
        with T.select_lora(linears, "default"):
            q, k, v = _qkv(block.attn, combined, concat_rope)
        with T.select_lora(linears, "cond"):
            cq, ck, cv = _qkv(block.attn, nc, cond_rope)
        n_txt, n_cond = e.shape[1], c.shape[1]
        cache = (k[:, :n_txt], v[:, :n_txt], ck, cv)
        out = _sdpa(torch.cat([q, cq], 1), torch.cat([k, ck], 1), torch.cat([v, cv], 1), mask).to(q.dtype)
        out, cond_out = out.split_with_sizes([out.shape[1] - n_cond, n_cond], dim=1)
        with T.select_lora(linears, "default"):
            o = block.attn.to_out[1](block.attn.to_out[0](out)) + block.ff(combined)
        o_txt, o_img = torch.split(o, [n_txt, h.shape[1]], dim=1)
        e = e + t_gate * o_txt
        h = h + gate * o_img
        with T.select_lora(linears, "cond"):
            co = block.attn.to_out[1](block.attn.to_out[0](cond_out)) + block.ff(nc)
        c = c + k_gate * co
        return e, h, c, cache

    @staticmethod
    def _single_img(block, h, mod, img_rope, cache, rows):
        """③ の 2 ステップ目以降: img の行だけ。キーは [txt (cache), img, cond (cache)] の順で、全体マスクの列と揃える。"""
        k_txt, v_txt, k_cond, v_cond = cache
        shift, scale, gate = torch.chunk(mod, 3, dim=-1)
        nh = (1 + scale) * block.norm(h) + shift
        linears = _linears(block)
        with T.select_lora(linears, "default"):
            q, k, v = _qkv(block.attn, nh, img_rope)
            out = _sdpa(q, torch.cat([k_txt, k, k_cond], 1), torch.cat([v_txt, v, v_cond], 1), rows).to(q.dtype)
            o = block.attn.to_out[1](block.attn.to_out[0](out)) + block.ff(nh)
        return h + gate * o

    @staticmethod
    def _single_txtimg(block, h, e, mod, mod_txt, concat_rope, cache, rows):
        """② の 2 ステップ目以降: txt と img の行。cond は K/V (cache) だけを列に足す。"""
        _, _, k_cond, v_cond = cache
        shift, scale, gate = torch.chunk(mod, 3, dim=-1)
        t_shift, t_scale, t_gate = torch.chunk(mod_txt, 3, dim=-1)
        nh = (1 + scale) * block.norm(h) + shift
        ne = (1 + t_scale) * block.norm(e) + t_shift
        combined = torch.cat([ne, nh], dim=1)
        linears = _linears(block)
        with T.select_lora(linears, "default"):
            q, k, v = _qkv(block.attn, combined, concat_rope)
            out = _sdpa(q, torch.cat([k, k_cond], 1), torch.cat([v, v_cond], 1), rows).to(q.dtype)
            o = block.attn.to_out[1](block.attn.to_out[0](out)) + block.ff(combined)
        o_txt, o_img = torch.split(o, [e.shape[1], h.shape[1]], dim=1)
        return e + t_gate * o_txt, h + gate * o_img
