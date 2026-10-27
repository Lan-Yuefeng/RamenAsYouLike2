"""推論時の caption 拡張器 (2026-10-26)。領域ごとの短い英語プロンプト → 領域 caption と全体 caption。

テキストだけの LLM (Qwen3-4B-Instruct) に LoRA を付けて、(short_prompts → captions_all) の組で fine-tune する。
正解画像は使わない (教師の caption を作るときに使っただけ)。同じ指示文で LoRA なしにすると zero-shot の比較対象になる。

  学習:  python expander.py train --split ../data/ramen/train.txt --out /export/space0/lan-y/seg2any_ckpt/expander_qwen3_4b
  v2:    train / gen の両方に --regions_only (全体 caption は拡張後の領域 caption の要約、下の INSTRUCTION_REGIONS)
  生成:  python expander.py gen --split ../data/ramen/val.txt --level short --lora <学習の出力> --out expanded/val_short_ft
         (--lora を付けなければ zero-shot)
生成の出力は captions_all と同じ形の JSON ({"global", "regions"}) なので、そのまま Seg2Any の推論に渡せる。
"""
import argparse
import json
import os
import random
import re

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
LEVELS = ["name", "short", "detail"]

INSTRUCTION = (
    "You write captions for a layout-to-image generator that draws a top-down photo of a bowl of ramen. "
    "The user gives one short prompt per region of the image. Expand them into captions.\n"
    "- For each region, write one sentence of at most 25 words that describes how it looks: color, "
    "texture, shape and surface details. Start with the object name. Keep everything the user said.\n"
    '- The soup caption starts with "The <kind> broth", e.g. "The tonkotsu (pork bone) broth".\n'
    "- Also write a global caption of 2 sentences, at most 60 words, for the whole photo.\n"
    'Return only a JSON object: {"global": ..., "regions": {<region>: <caption>, ...}}.\n\n'
    "Regions:\n"
)


# v2 (--regions_only): 全体 caption を拡張器に書かせると、入力にない具 (辛味のキャベツとネギの山など) を
# スープ種ごとの決まり文句として覚えてしまった。v2 では拡張器は領域だけを書き、全体 caption は
# 拡張後の領域 caption を LoRA なしのモデルで要約して作る (新しいものを足さないよう指示する)
INSTRUCTION_REGIONS = (
    INSTRUCTION.split("- Also write a global caption")[0]
    + 'Return only a JSON object: {"regions": {<region>: <caption>, ...}}.\n\n'
    + "Regions:\n"
)

GLOBAL_FROM_REGIONS = (
    "Below are captions for the regions of a top-down photo of a bowl of ramen.\n"
    "Write a caption for the whole photo in 2 sentences, at most 60 words: the kind of ramen and how the "
    "broth looks, the toppings, the bowl, the background. Use only what the region captions say; do not "
    "add any item, garnish or detail that they do not mention. Return only the caption.\n\n"
)


def build_prompt(short, regions_only=False):
    return (INSTRUCTION_REGIONS if regions_only else INSTRUCTION) + "\n".join(f"- {c}: {p}" for c, p in short.items())


def load_pairs(names, level, regions_only=False):
    """level が None なら 3 段階すべてを使う (学習用)。"""
    pairs = []
    for n in names:
        stem = n.replace(".png", "")
        sp = json.load(open(os.path.join(HERE, "short_prompts", stem + ".json")))
        cap = json.load(open(os.path.join(HERE, "captions_all", stem + ".json")))
        target = {"regions": cap["regions"]} if regions_only else {"global": cap["global"], "regions": cap["regions"]}
        target = json.dumps(target, ensure_ascii=False)
        for lv in ([level] if level else LEVELS):
            short = {c: v[lv] for c, v in sp["regions"].items()}
            pairs.append({"image": n, "level": lv, "short": short, "target": target})
    return pairs


def load_model(lora=None, train=False, rank=16):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    if train:
        from peft import LoraConfig, get_peft_model

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.05,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    elif lora:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, lora)
    return model, tok


def encode(tok, prompt, target=None):
    msgs = [{"role": "user", "content": prompt}]
    p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(p_ids, "input_ids"):  # transformers 5 は BatchEncoding を返す
        p_ids = p_ids["input_ids"]
    if target is None:
        return p_ids
    t_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids  # 損失は答えの部分だけ


def train(args):
    names = [l.strip() for l in open(args.split) if l.strip()]
    pairs = load_pairs(names, None, args.regions_only)
    model, tok = load_model(train=True, rank=args.rank)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    model.train()
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(pairs)
        for i in range(0, len(pairs), args.batch_size):
            batch = [encode(tok, build_prompt(p["short"], args.regions_only), p["target"])
                     for p in pairs[i : i + args.batch_size]]
            L = max(len(ids) for ids, _ in batch)
            pad = tok.pad_token_id
            ids = torch.tensor([x + [pad] * (L - len(x)) for x, _ in batch], device="cuda")
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device="cuda")
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch], device="cuda")
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}", flush=True)
        model.save_pretrained(os.path.join(args.out, f"epoch{epoch + 1}"))
    print("done", args.out)


@torch.no_grad()
def summarize(model, tok, regions):
    """拡張後の領域 caption から全体 caption を作る。LoRA は外す (要約は学習していない)。"""
    prompt = GLOBAL_FROM_REGIONS + "\n".join(f"- {c}: {v}" for c, v in regions.items())
    ids = torch.tensor([encode(tok, prompt)], device="cuda")
    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else torch.no_grad()
    with ctx:
        out = model.generate(input_ids=ids, max_new_tokens=160, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def gen(args):
    names = [l.strip() for l in open(args.split) if l.strip()]
    if args.limit:
        names = names[: args.limit]
    model, tok = load_model(lora=args.lora)
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    n_fail = 0
    for p in load_pairs(names, args.level):
        ids = torch.tensor([encode(tok, build_prompt(p["short"], args.regions_only))], device="cuda")
        out = model.generate(input_ids=ids, max_new_tokens=600, do_sample=False)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            d = json.loads(m.group(0))
            regions = {c: d["regions"][c] for c in p["short"]}
            glob = summarize(model, tok, regions) if args.regions_only else d["global"]
            ok = True
        except Exception:  # 崩れたら短いプロンプトをそのまま使う (数えて報告する)
            regions, glob, ok = dict(p["short"]), ", ".join(p["short"].values()), False
            n_fail += 1
        json.dump({"image": p["image"], "level": p["level"], "short": p["short"], "parsed": ok,
                   "global": glob, "regions": regions},
                  open(os.path.join(args.out, p["image"].replace(".png", ".json")), "w"),
                  ensure_ascii=False, indent=1)
        print(p["image"], "ok" if ok else "FAILED", json.dumps(regions, ensure_ascii=False)[:150], flush=True)
    print(f"parse failures: {n_fail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "gen"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", choices=LEVELS, default="short")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regions_only", action="store_true",
                    help="v2: 拡張器は領域だけ書き、全体 caption は要約で作る (学習と生成の両方に付ける)")
    args = ap.parse_args()
    train(args) if args.mode == "train" else gen(args)


if __name__ == "__main__":
    main()
