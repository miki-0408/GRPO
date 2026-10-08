#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
难度探测 —— 找出模型"刚好会一半"的任务难度。

为什么需要它：
    GRPO 的梯度全靠组内奖励差异。全对或全错 -> 标准差 0 -> 优势 0 -> 梯度 0。
    所以训练数据必须落在模型的能力边界上。这个脚本量出那条边界在哪。

    这正是 JD 里说的「RLHF/RLAIF 数据生产管线」要做的事。

跑法：python probe_difficulty.py
"""

import os
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = os.path.expanduser("~/models/Qwen3-0.6B")
G = 4           # 每个问题生成几条（要和训练时的 G 一致）
N_QUESTIONS = 16
MAX_NEW = 64


def hr(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# ── 候选难度：加你自己想试的 ────────────────────────────────────
DIFFICULTIES = {
    "2位数 x 1位数": lambda r: (r.randint(11, 99), r.randint(2, 9)),
    "3位数 x 1位数": lambda r: (r.randint(100, 999), r.randint(2, 9)),
    "2位数 x 2位数": lambda r: (r.randint(11, 99), r.randint(11, 99)),
    "3位数 + 3位数": lambda r: (r.randint(100, 999), r.randint(100, 999)),
    "2位数 x 2位数 (大)": lambda r: (r.randint(50, 99), r.randint(50, 99)),
}


def build_prompt(tok, a, b, op):
    msgs = [{"role": "user",
             "content": "What is %d %s %d? Answer with just the number." % (a, op, b)}]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def reward_fn(completion, gold):
    nums = re.findall(r"-?\d+", completion)
    return 1.0 if nums and int(nums[-1]) == gold else 0.0


def main():
    if not os.path.isdir(MODEL_DIR):
        sys.exit("找不到模型目录：%s" % MODEL_DIR)

    import random
    rng = random.Random(0)

    hr("加载模型")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    print("就绪。每个难度测 %d 道题，每题生成 %d 条。" % (N_QUESTIONS, G))

    hr("探测结果")
    print("%-22s %10s %10s   %s" % ("难度", "单条准确率", "有差异组占比", "判定"))
    print("-" * 70)

    results = []
    for name, gen in DIFFICULTIES.items():
        op = "+" if "+" in name else "*"
        rewards_all, n_var, n_all = [], 0, 0

        for _ in range(N_QUESTIONS):
            a, b = gen(rng)
            gold = a + b if op == "+" else a * b
            enc = tok(build_prompt(tok, a, b, op), return_tensors="pt").to("cuda")
            plen = enc.input_ids.shape[1]
            with torch.no_grad():
                seqs = model.generate(
                    **enc, do_sample=True, temperature=1.0, top_p=1.0,
                    num_return_sequences=G, max_new_tokens=MAX_NEW,
                    pad_token_id=tok.pad_token_id)
            rs = [reward_fn(tok.decode(s[plen:], skip_special_tokens=True), gold)
                  for s in seqs]
            rewards_all += rs
            n_all += 1
            if max(rs) != min(rs):
                n_var += 1

        acc = sum(rewards_all) / len(rewards_all)
        var_ratio = n_var / n_all
        if var_ratio >= 0.6:
            verdict = "★ 合适"
        elif var_ratio >= 0.3:
            verdict = "可用"
        elif acc > 0.9:
            verdict = "太简单"
        elif acc < 0.1:
            verdict = "太难"
        else:
            verdict = "看情况"
        print("%-22s %9.1f%% %11.1f%%   %s" % (name, 100 * acc, 100 * var_ratio, verdict))
        results.append((name, acc, var_ratio))

    hr("怎么用")
    best = max(results, key=lambda x: x[2])
    if best[2] < 0.3:
        print("没有一个难度落在边界上（有差异组占比都 < 30%）。")
        print("说明这个模型的算术能力整体太强/太弱，试试：")
        print("  - 换任务类型（多步运算、应用题），而不是继续调数字大小")
        print("  - 或者降低 G 之外的采样温度设成 1.2，让输出更散")
    else:
        print("推荐难度：【%s】—— 有差异组占比 %.1f%%" % (best[0], 100 * best[2]))
        print("把它填回 grpo_from_scratch.py 的 sample_question()。")
    print()
    print("提醒：有差异组占比高，不等于每条都学到东西。")
    print("      它高只说明『梯度基本不为 0』，这是训练能推进的必要条件。")


if __name__ == "__main__":
    main()
