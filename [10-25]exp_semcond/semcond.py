"""意味付き cond トークン (semantic cond)：領域ごとの caption 埋め込みを、輪郭 cond トークンに足す。

Seg2Any の輪郭 cond (Entity Contour Map) はどの領域も同じ白線なので、クラスを区別しない。
「何の領域か」は regional caption とアテンションマスクだけで伝わっている ([10-25]exp_ablation)。
そこで cond トークンにも領域の意味を持たせる。

    cond_i <- x_embedder(cond_i) + W · context_embedder( Σ_r a_ir · e_r )

  e_r  : 領域 r の regional caption の Qwen3-VL 埋め込み (20480 次元) を、caption 部分のトークンで平均したもの。
         テキストの埋め込みは transformer に渡すもの (encode_all_prompt の出力) をそのまま使う。
         右パディングの因果 LM なので、パディング済みの出力でも caption 部分は未パディングと同じ値になる
  a_ir : cond トークン i のパッチ (16x16 px) のうち領域 r が占める割合。輪郭上のトークンは両側の領域を混ぜて持つ
  context_embedder : テキスト用の入力射影 (凍結のまま共有)。20480 -> 3072
  W    : 新しく足す Linear(3072, 3072)。ゼロ初期化なので、学習開始時は元の Seg2Any と完全に同じ出力になる

実装：W を transformer.cond_sem_proj として後から付け、cond の最初の mode block (mode_blocks.image_cond.0) の
forward pre-hook で足す。x_embedders.image_cond は LoRA (peft) に差し替えられるので、そこにはフックを掛けない。
足す値は forward の前に transformer._cond_sem に入れておく (B, N_cond, 20480)。勾配チェックポイントの再計算でも
同じ値を使うよう、forward のあとも消さない。
"""
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

SEM_PROJ_FILE = "cond_sem_proj.safetensors"


# ---------------------------------------------------------------- transformer 側
def enable_semcond(transformer):
    """transformer に cond_sem_proj (ゼロ初期化、fp32、学習対象) を付け、フックを掛ける。何度呼んでも 1 回だけ。"""
    if getattr(transformer, "cond_sem_proj", None) is not None:
        return transformer.cond_sem_proj
    dim = transformer.inner_dim
    proj = torch.nn.Linear(dim, dim, bias=True)
    torch.nn.init.zeros_(proj.weight)
    torch.nn.init.zeros_(proj.bias)
    proj.to(device=transformer.device, dtype=torch.float32)
    transformer.cond_sem_proj = proj
    transformer._cond_sem = None

    def pre_hook(module, args):
        sem = transformer._cond_sem
        if sem is None:
            return None
        hidden = args[0]
        assert sem.shape[:2] == hidden.shape[:2], f"cond_sem {tuple(sem.shape)} と cond トークン {tuple(hidden.shape)} の数が合わない"
        ctx = transformer.context_embedder(sem.to(transformer.context_embedder.weight.dtype))
        add = transformer.cond_sem_proj(ctx.float()).to(hidden.dtype)
        return (hidden + add,) + tuple(args[1:])

    transformer.mode_blocks["image_cond"][0].register_forward_pre_hook(pre_hook)
    return proj


def save_sem_proj(transformer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().float().cpu().contiguous() for k, v in transformer.cond_sem_proj.state_dict().items()}
    save_file(sd, os.path.join(out_dir, SEM_PROJ_FILE))


def load_sem_proj(transformer, ckpt_dir):
    enable_semcond(transformer)
    sd = load_file(os.path.join(ckpt_dir, SEM_PROJ_FILE))
    transformer.cond_sem_proj.load_state_dict({k: v.to(transformer.cond_sem_proj.weight.device) for k, v in sd.items()})


# ---------------------------------------------------------------- 入力側
def _template_prefix_suffix(pipe):
    """チャットテンプレートの caption より前 / 後のトークン数 (Qwen3-VL では 3 / 5)"""
    if not hasattr(pipe, "_semcond_ps"):
        tok = pipe.tokenizer.tokenizer
        a = tok(pipe._wrap_chat_template("", 10**6)).input_ids
        b = tok(pipe._wrap_chat_template("menma", 10**6)).input_ids
        p = 0
        while a[p] == b[p]:
            p += 1
        s = 0
        while a[-1 - s] == b[-1 - s]:
            s += 1
        pipe._semcond_ps = (p, s)
    return pipe._semcond_ps


def region_pooled_embeds(pipe, prompt_embeds, txt_seq_lens, regional_prompts, regional_max_sequence_length=50):
    """各サンプルの regional caption ごとの平均埋め込み。返り値は list[(n_regions, 20480)] (fp32)。
    prompt_embeds / txt_seq_lens は encode_all_prompt の出力 (global caption あり、num_images_per_prompt=1)。
    空の caption は caption 部分が無いので 0 ベクトル。"""
    tok = pipe.tokenizer.tokenizer
    pre, suf = _template_prefix_suffix(pipe)
    out = []
    for i, prompts in enumerate(regional_prompts):
        lens = txt_seq_lens[i]
        start = lens[0]  # global caption の区間を飛ばす
        vecs = []
        for r, p in enumerate(prompts):
            real = len(tok(pipe._wrap_chat_template(p, regional_max_sequence_length)).input_ids)
            s, e = start + pre, start + real - suf
            if e > s:
                vecs.append(prompt_embeds[i, s:e].float().mean(0))
            else:
                vecs.append(prompt_embeds.new_zeros(prompt_embeds.shape[-1], dtype=torch.float32))
            start += lens[1 + r]
        out.append(torch.stack(vecs))
    return out


def cond_semantic(pooled, labels, cond_ids, cond_seq_lens, cond_token_hw, pos_scale=1):
    """cond トークンごとの意味ベクトル (B, N_cond, 20480)。
    pooled : region_pooled_embeds の出力。labels : list[(n_regions, H, W)] (0/1。regional caption と同じ順)
    cond_ids : filter_cond_token 後の (B, N_cond, 4)。h, w 座標は (0.5 * pos_scale) 倍されている
    cond_token_hw : cond のトークン格子 (h, w)。pad トークン (cond_seq_lens 以降) は 0"""
    th, tw = cond_token_hw
    B, N = cond_ids.shape[:2]
    sem = pooled[0].new_zeros(B, N, pooled[0].shape[-1])
    for b in range(B):
        frac = F.adaptive_avg_pool2d(labels[b].float()[None].to(sem.device), (th, tw))[0]  # (n, th, tw)
        n = cond_seq_lens[b]
        rows = torch.round(cond_ids[b, :n, 1] / (0.5 * pos_scale)).long()
        cols = torch.round(cond_ids[b, :n, 2] / (0.5 * pos_scale)).long()
        a = frac[:, rows, cols].T  # (n_tok, n_regions)
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-6)
        sem[b, :n] = a @ pooled[b].to(sem.device)
    return sem


@torch.no_grad()
def semcond_for_pipeline_call(pipe, global_prompt, regional_prompts, regional_labels, cond, height, width,
                              cond_scale_factor=1, regional_max_sequence_length=50, max_sequence_length=512,
                              text_pad_multiple=80, regional_text_position="restart"):
    """推論用：Flux3RegionalPipeline.__call__ (バッチ 1、num_images_per_prompt 1) が中で作るのと同じ cond トークン列に
    対応する意味ベクトルを作る。cond は __call__ に渡すもの ([0,1])。返り値を transformer._cond_sem に入れてから __call__ を呼ぶ。"""
    from src.pipelines.pipeline_flux3 import prepare_image_ids

    multiple = pipe.vae_scale_factor * 2 * cond_scale_factor
    height, width = int(height) // multiple * multiple, int(width) // multiple * multiple
    ch, cw = height // cond_scale_factor, width // cond_scale_factor
    pe, lens, _ = pipe.encode_all_prompt(
        [global_prompt], [regional_prompts], global_max_sequence_length=max_sequence_length,
        regional_max_sequence_length=regional_max_sequence_length, device=pipe._execution_device,
        regional_text_position=regional_text_position, text_pad_multiple=text_pad_multiple,
    )
    pooled = region_pooled_embeds(pipe, pe, lens, [regional_prompts], regional_max_sequence_length)
    c = pipe.image_processor.preprocess(cond, height=ch, width=cw).to(pipe._execution_device)
    th, tw = ch // (pipe.vae_scale_factor * 2), cw // (pipe.vae_scale_factor * 2)
    ids = prepare_image_ids(th, tw, l_coord=-1, pos_scale=cond_scale_factor).to(c.device)
    dummy = torch.zeros(1, th * tw, 1, device=c.device)
    _, cond_ids, cond_seq_lens, _ = pipe.filter_cond_token(c, dummy, ids, vae_scale_factor=pipe.vae_scale_factor * 2)
    labels = [regional_labels] if isinstance(regional_labels, torch.Tensor) else regional_labels
    return cond_semantic(pooled, labels, cond_ids, cond_seq_lens, (th, tw), cond_scale_factor)
