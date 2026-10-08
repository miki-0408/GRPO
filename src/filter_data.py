#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 数据筛选 —— 从 GSM8K 里挑出"模型刚好会一半"的题。

为什么要筛：
    GRPO 的梯度全靠组内奖励差异。一道题如果 8 条回答全对或全错，
    组内标准差是 0 -> 优势全 0 -> 梯度 0 -> 这条样本白算。
    Day 6/7 实测：不筛的话约 40~50% 的组是零方差。

    这正是 JD 里「RLHF/RLAIF 数据生产管线」要做的事，
    也是 DAPO「动态采样」的离线版。

★ 方法论：只筛【训练集】，验证集保持原始分布。
    如果把验证集也筛成"刚好一半对"，那 eval_reward 会天然固定在 0.5 附近，
    它就不再衡量"模型在 GSM8K 上有多好"，而只是重复一遍筛选标准。
    验证集必须代表真实分布，否则你量的是自己的筛子，不是模型。

跑法：python filter_data.py
输出：~/grpo_data/train 和 ~/grpo_data/eval（HF dataset 格式）+ stats.json
"""

import json
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 把脚本所在目录【和项目根】都加进搜索路径 ——
# 这样无论 grpo_common.py 在根目录还是同级目录，都能导入到。
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from grpo_common import (MODEL_DIR, build_messages, build_prompt_text,
                         reward_from_texts)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = os.path.expanduser("~/grpo_data")

# ── 配置 ────────────────────────────────────────────────────────────────
G                    = 8      # 必须和训练时一致
MAX_NEW              = 384    # 必须和训练时一致
N_TRAIN_CANDIDATES   = 800    # 从这么多题里筛
N_EVAL               = 80     # 验证集条数（不筛）
TARGET_TRAIN         = 300    # 筛够这么多就停
EVAL_START           = 3000   # 验证集从这里开始取，和候选池不重叠


def hr(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def classify(rewards):
    """给一道题的 G 条回答分类"""
    s = sum(rewards)
    if s == 0:
        return "全错"
    if s == len(rewards):
        return "全对"
    return "有差异"      # ← 只有这类能产生梯度


def main():
    if not os.path.isdir(MODEL_DIR):
        sys.exit("找不到模型目录：%s" % MODEL_DIR)
    if not torch.cuda.is_available():
        sys.exit("没有 GPU")

    hr("准备数据")
    raw = load_dataset("openai/gsm8k", "main", split="train")
    print("GSM8K 总条数:", len(raw))

    cand = raw.select(range(min(N_TRAIN_CANDIDATES, len(raw))))
    eval_raw = raw.select(range(EVAL_START, min(EVAL_START + N_EVAL, len(raw))))
    print("训练候选池:", len(cand), " 验证集:", len(eval_raw), "（不筛，保持原始分布）")

    hr("加载模型")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16).to("cuda")
    model.eval()
    print("就绪")

    # ══════════════════════════════════════════════════════════
    # 逐题采样 G 次，按组内方差分类
    # 注意：一次只处理 1 道题。批量做要多处理 padding 和输出顺序，
    #       对一次性的预处理不值得冒那个险 —— 正确性优先。
    # ══════════════════════════════════════════════════════════
    hr("筛选训练集")
    print("每题采样 %d 次，只保留『有差异』的题。预计要几分钟...\n" % G)

    kept, stats = [], {"全对": 0, "全错": 0, "有差异": 0}
    t0 = time.time()

    for idx in range(len(cand)):
        q = cand[idx]["question"]
        ans = cand[idx]["answer"]

        enc = tok(build_prompt_text(tok, q), return_tensors="pt").to("cuda")
        n_in = enc.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=True, temperature=1.0, top_p=1.0,
                num_return_sequences=G, max_new_tokens=MAX_NEW,
                pad_token_id=tok.pad_token_id)

        texts = [tok.decode(s[n_in:], skip_special_tokens=True) for s in out]
        rewards = reward_from_texts(texts, [ans] * G)
        cls = classify(rewards)
        stats[cls] += 1

        if cls == "有差异":
            kept.append(idx)

        if (idx + 1) % 50 == 0 or idx + 1 == len(cand):
            el = time.time() - t0
            rate = (idx + 1) / el
            eta = (len(cand) - idx - 1) / rate if rate > 0 else 0
            print("  已测 %3d/%d  保留 %3d  有差异率 %.1f%%  已用 %.0fs  预计还需 %.0fs"
                  % (idx + 1, len(cand), len(kept),
                     100 * stats["有差异"] / (idx + 1), el, eta))

        if len(kept) >= TARGET_TRAIN:
            print("\n  已筛够 %d 条，提前停止（测了 %d 道）" % (TARGET_TRAIN, idx + 1))
            break

    total = sum(stats.values())
    print("\n分类统计（共 %d 道）：" % total)
    for k in ["全对", "全错", "有差异"]:
        print("  %-6s %5d  (%.1f%%)" % (k, stats[k], 100 * stats[k] / total))
    print("\n  → 只有『有差异』的 %d 道会产生梯度，占 %.1f%%"
          % (stats["有差异"], 100 * stats["有差异"] / total))

    # ══════════════════════════════════════════════════════════
    # 落盘
    # ══════════════════════════════════════════════════════════
    hr("保存")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds = cand.select(kept).map(lambda x: {"prompt": build_messages(x["question"])})
    eval_ds = eval_raw.map(lambda x: {"prompt": build_messages(x["question"])})

    train_ds.save_to_disk(os.path.join(OUT_DIR, "train"))
    eval_ds.save_to_disk(os.path.join(OUT_DIR, "eval"))

    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({
            "G": G, "MAX_NEW": MAX_NEW,
            "candidates_tested": total,
            "class_counts": stats,
            "kept": len(train_ds),
            "kept_ratio": len(train_ds) / total if total else 0,
            "eval_size": len(eval_ds),
        }, f, ensure_ascii=False, indent=2)

    print("训练集:", len(train_ds), "->", os.path.join(OUT_DIR, "train"))
    print("验证集:", len(eval_ds), "->", os.path.join(OUT_DIR, "eval"))
    print("统计    ->", os.path.join(OUT_DIR, "stats.json"))

    hr("下一步")
    print("把 train_trl.py 里的数据加载改成读这个：")
    print()
    print("    from datasets import load_from_disk")
    print("    ds      = load_from_disk(os.path.expanduser('~/grpo_data/train'))")
    print("    eval_ds = load_from_disk(os.path.expanduser('~/grpo_data/eval'))")
    print()
    print("这样训练用的每一道题都是『有梯度』的，同样的步数有效信号能翻好几倍。")


if __name__ == "__main__":
    main()
