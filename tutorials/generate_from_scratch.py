#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手写一遍 generate()，证明它只是一个循环。

你问的问题：一次前向只给出"最后一个位置的下一个 token"，
那 generate 怎么产出一整串？答案是：它循环跑了很多次前向。

这个脚本用 8 行核心代码复现官方的贪心生成，然后和官方结果对比。

跑法：python generate_from_scratch.py
"""

import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL = os.path.expanduser("~/models/Qwen3-0.6B")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = 12  # 手动生成多少个 token


def main():
    if not os.path.isdir(MODEL):
        sys.exit("找不到模型目录：%s" % MODEL)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16 if DEV == "cuda" else torch.float32
    ).to(DEV)
    model.eval()

    msgs = [{"role": "user", "content": "What is 2 + 3? Answer with just the number."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    start_len = tok(prompt, return_tensors="pt").input_ids.shape[1]
    print("prompt token 数 :", start_len)

    # ══════════════════════════════════════════════════════════
    # 手写生成：核心就这 4 行，跑 N 次
    # ══════════════════════════════════════════════════════════
    print("\n=== 手写生成：每次前向只加 1 个 token ===")
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEV)

    with torch.no_grad():
        for step in range(N):
            out = model(ids)                     # ① 前向 —— 注意输入每次都在变长
            logits = out.logits[0, -1]           # ② 只取【最后一个位置】的分数
            next_id = torch.argmax(logits)       # ③ 贪心：选分数最高的那个
            ids = torch.cat([ids, next_id.view(1, 1)], dim=1)   # ④ 拼到末尾，下一轮再喂进去
            print("  第%2d 次前向  序列长度 %d→%d  选出的 token = %r"
                  % (step + 1, ids.shape[1] - 1, ids.shape[1], tok.decode([next_id.item()])))

    mine = tok.decode(ids[0][start_len:], skip_special_tokens=True)

    # ══════════════════════════════════════════════════════════
    # 官方 generate：同样步数、同样贪心
    # ══════════════════════════════════════════════════════════
    inputs = tok(prompt, return_tensors="pt").to(DEV)
    with torch.no_grad():
        official = model.generate(
            **inputs,
            max_new_tokens=N,
            do_sample=False,
            repetition_penalty=1.0,   # 显式关掉，保证和手写版条件一致
        )
    theirs = tok.decode(official[0][start_len:], skip_special_tokens=True)

    print("\n=== 对比 ===")
    print("手写版  :", repr(mine))
    print("官方版  :", repr(theirs))
    print()
    if mine == theirs:
        print("[OK] 完全一致 —— 证明 generate() 就是这个循环，没有别的魔法。")
    else:
        print("[!] 有差异。通常是因为官方 generate 还应用了 generation_config.json 里的")
        print("    默认参数（repetition_penalty / top_k 等）。把那些参数显式设成中性值即可对齐。")

    print("\n关键观察：")
    print("  序列长度每轮 +1，而每次前向都要把【整条序列】重新过一遍网络。")
    print("  生成 N 个 token = N 次前向。这就是生成慢的根本原因。")
    print("  真实推理引擎用 KV Cache 缓存前面算过的中间结果，避免重复计算前缀。")


if __name__ == "__main__":
    main()
