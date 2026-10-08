#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 3 —— Transformers 手感：从加载模型到 GRPO 的 rollout

⚠️ 先读 `Transformers-基础.md`，再动手。

跑法：python transformers_basics.py

⚠️ 首次运行会从 HuggingFace 下载模型（约 1.5GB）。国内慢的话先在另一个终端设镜像：
      export HF_ENDPOINT=https://hf-mirror.com
   （想持久生效就写进 ~/.bashrc，和之前那个 rl 别名一样）

这一天的目标不是"让模型说话"，是搞清楚 GRPO 训练循环的第 ① 步长什么样。
最后一节你会亲手证明：**没有采样多样性，GRPO 的梯度就是 0。**
"""

import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 模型位置 ────────────────────────────────────────────────────────────
# 这里显式指定【本地目录】，脚本绝不自己下载。找不到就直接报错退出。
#
# 下载命令见 `下载模型.md`，或直接照抄这一段（在 WSL 里跑，只需一次）：
#
#     export HF_ENDPOINT=https://hf-mirror.com     # 国内加速，建议写进 ~/.bashrc
#     mkdir -p ~/models
#     python -c "from huggingface_hub import snapshot_download as d; \
#                d('Qwen/Qwen3-0.6B', local_dir='$HOME/models/Qwen3-0.6B')"
#
# 换模型或换路径，只改下面这一行。
MODEL = os.path.expanduser("~/models/Qwen3-0.6B")

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# 一个简单到 0.6B 也有机会做对的算术应用题，用来观察"奖励有没有差异"
QUESTION = "A store has 24 apples. It sells 1/3 of them. How many apples are left?"
EXPECTED = "16"


def hr(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def build_prompt(tok, question, thinking=False):
    """把一个问题按模型训练时的对话格式包起来。

    Qwen3 的模板支持 enable_thinking 参数控制思考模式。不同版本/不同模板
    不一定支持这个参数，所以这里做了兼容——不支持就退回普通调用。
    """
    msgs = [{"role": "user", "content": question}]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
    except TypeError:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )


def looks_correct(text, expected):
    """极简的规则奖励：回答里出现期望的数字就算对。

    真正的 GSM8K 奖励函数要复杂一些（要正则抽取 \\boxed{} 或 #### 后面的答案），
    但"用规则判断对错"这个形式，就是 RLAIF 里那个 verifiable reward。
    """
    return expected in text


# ══════════════════════════════════════════════════════════════
# 0. 加载 tokenizer
# ══════════════════════════════════════════════════════════════
def s0_load_tokenizer():
    hr("0. 加载 tokenizer")
    print("正在加载:", MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL)
    print("词表大小        :", tok.vocab_size)
    print("对话模板可用    :", tok.chat_template is not None)
    return tok


# ══════════════════════════════════════════════════════════════
# 1. 分词与还原
# ══════════════════════════════════════════════════════════════
def s1_tokenize(tok):
    hr("1. 分词与还原")
    text = "A store has 24 apples."
    ids = tok(text)["input_ids"]
    print("原文            :", text)
    print("token ids       :", ids)
    print("token 数        :", len(ids))
    print("还原回去        :", repr(tok.decode(ids)))

    # 逐个 token 看，直观感受「子词分词」是什么意思
    print("\n逐个 token：")
    for i in ids[:12]:
        print("   %-6s -> %r" % (i, tok.decode([i])))

    # ★ 关键点：模型只认数字。分词是「文本 → 数字」的唯一入口，
    #   而且它是可逆的（decode 能还原）。

    # —— 你来写 ——
    # TODO 1: 把上面那句话改成中文（比如 "商店里有 24 个苹果。"）重新分词，
    #         数一数中文的 token 数是多少，和英文比是多还是少？
    # print("...", len(tok("...")["input_ids"]))
    print("中文分词结果    :", len(tok("商店里有 24 个苹果。")["input_ids"]))

# ══════════════════════════════════════════════════════════════
# 2. chat template
# ══════════════════════════════════════════════════════════════
def s2_chat_template(tok):
    hr("2. chat template —— instruct 模型的隐藏开关")
    raw = QUESTION
    tmpl = build_prompt(tok, QUESTION)

    print("【裸文本】")
    print(repr(raw))
    print("\n【套上 chat template 之后】")
    print(repr(tmpl))

    print("\n可见：模板自动加上了角色标记和 <|im_start|> / <|im_end|> 这类特殊 token。")
    print("后训练模型是在这种格式上训练的，直接喂裸文本等于用错了输入格式。")

    # ★ 关键点：这个坑不报错，只是效果悄悄变差。
    #   和你昨天第 3 题里「忘传 parameters()」是同一类安静的错误。

    # —— 你来写 ——
    # TODO 2: 把上面的 build_prompt(tok, QUESTION) 改成
    #         build_prompt(tok, QUESTION, thinking=True)，打印出来。
    #         对比一下：开了思考模式的模板，多出什么标记？
    #         （提示：找找有没有  thinking 之类的东西。找不到也没关系，说明这个版本不需要。）
    tmpl_thinking = build_prompt(tok,QUESTION, thinking=True)
    print("\n【套上 chat template + thinking=True 之后】")
    print(repr(tmpl_thinking))

# ══════════════════════════════════════════════════════════════
# 3. 加载模型，看 logits
# ══════════════════════════════════════════════════════════════
def s3_load_model():
    hr("3. 加载模型并观察 logits")
    print("正在加载模型（约 1.2GB）...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16 if DEV == "cuda" else torch.float32,
    ).to(DEV)
    model.eval()

    n = sum(p.numel() for p in model.parameters())
    print("参数量          : %.2f 亿" % (n / 1e8))
    if DEV == "cuda":
        print("显存占用(GB)    : %.2f" % (torch.cuda.memory_allocated() / 1024**3))

    # ★ 关键点：logits 的形状是 (batch, seq_len, vocab_size)。
    #   每个位置上，模型对词表里每一个 token 都打了一个分。
    #   这就是"预测下一个 token"的全部含义。
    return model


def s3_inspect_logits(tok, model):
    """看模型在某个位置上最想接什么词——这是理解 generate 的基础"""
    print("\n---- 观察 logits ----")
    prompt = build_prompt(tok, QUESTION)
    inputs = tok(prompt, return_tensors="pt").to(DEV)

    with torch.no_grad():
        out = model(**inputs)

    print("logits 形状     :", tuple(out.logits.shape))
    print("                (batch, seq_len, vocab_size) —— 对得上吗？")

    last = out.logits[0, -1]          # 只看最后一个位置
    top = torch.topk(last, 5)
    print("\n模型认为下一个 token 最可能是：")
    for score, idx in zip(top.values, top.indices):
        print("   %-14r %8.3f" % (tok.decode([idx.item()]), score.item()))


# ══════════════════════════════════════════════════════════════
# 4. generate —— 贪心 vs 采样
# ══════════════════════════════════════════════════════════════
def s4_generate(tok, model):
    hr("4. generate —— 贪心 vs 采样")
    prompt = build_prompt(tok, QUESTION)
    inputs = tok(prompt, return_tensors="pt").to(DEV)
    n_in = inputs["input_ids"].shape[1]

    # —— 贪心 ——
    with torch.no_grad():
        g = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    print("【贪心 do_sample=False】")
    print(tok.decode(g[0][n_in:], skip_special_tokens=True).strip())

    # —— 采样 ——
    with torch.no_grad():
        s = model.generate(
            **inputs, max_new_tokens=64, do_sample=True,
            temperature=0.8, top_p=0.95, num_return_sequences=2,
        )
    print("\n【采样 do_sample=True, temperature=0.8】")
    for i, seq in enumerate(s):
        print("  第%d条: %s" % (i + 1, tok.decode(seq[n_in:], skip_special_tokens=True).strip()))

    # ★ 关键点：贪心是确定性的（同输入同输出），采样是随机的。
    #   看起来只是"输出稳不稳定"的区别，但对 GRPO 来说这是生死攸关的——
    #   下一节会证明为什么。

    # —— 你来写 ——
    # TODO 3: 把 temperature 改成 0.1 再跑一次采样，观察两条回答还一样吗？
    #         再改成 1.5 试试。体会一下温度在控制什么。
    #         （这一步不用改代码结构，把参数改一下重跑整个脚本就行）
    with torch.no_grad():
        s_temp_01 = model.generate(
            **inputs, max_new_tokens=64, do_sample=True,
            temperature=0.1, top_p=0.95, num_return_sequences=2,
        )
    print("\n【采样 do_sample=True, temperature=0.1】")
    for i, seq in enumerate(s_temp_01):
        print("  第%d条: %s" % (i + 1, tok.decode(seq[n_in:], skip_special_tokens=True).strip()))

    with torch.no_grad():
        s_temp_15 = model.generate(
            **inputs, max_new_tokens=64, do_sample=True,
            temperature=1.5, top_p=0.95, num_return_sequences=2,
        )
    print("\n【采样 do_sample=True, temperature=1.5】")
    for i, seq in enumerate(s_temp_15):
        print("  第%d条: %s" % (i + 1, tok.decode(seq[n_in:], skip_special_tokens=True).strip()))

# ══════════════════════════════════════════════════════════════
# 5. 核心验收：模拟 GRPO 的 rollout
# ══════════════════════════════════════════════════════════════
def s5_rollout(tok, model):
    hr("5. 核心验收：GRPO 的 rollout 长什么样")
    """
    GRPO 的第一步就是这个：

        responses = model.generate(prompts, num_return_sequences=G)   # ← 下面就是它
        rewards   = reward_fn(prompts, responses)
        advantages = group_normalize(rewards)      # 组内归一化
        loss = policy_gradient_loss(responses, advantages)

    这一节要把「采样 → 奖励差异 → 优势信号」这条链走通，
    并且证明：**如果 G 条回答全一样，优势就是 0，梯度就是 0，训练什么都没发生。**
    """
    G = 4
    prompt = build_prompt(tok, QUESTION)
    inputs = tok(prompt, return_tensors="pt").to(DEV)
    n_in = inputs["input_ids"].shape[1]

    def extract(seqs):
        return [tok.decode(s[n_in:], skip_special_tokens=True).strip() for s in seqs]

    def report(name, answers):
        rewards = [1.0 if looks_correct(a, EXPECTED) else 0.0 for a in answers]
        print("【%s】" % name)
        for i, (a, r) in enumerate(zip(answers, rewards)):
            print("  第%d条  奖励=%.0f  %s" % (i + 1, r, a[:90]))

        # 这就是 GRPO 的组内归一化优势： A_i = (r_i - 组内均值) / 组内标准差
        n = len(rewards)
        mean = sum(rewards) / n
        std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
        adv = [(r - mean) / std if std > 1e-8 else 0.0 for r in rewards]

        print("  ----")
        print("  组内奖励    :", rewards)
        print("  组内均值    : %.3f    组内标准差: %.3f" % (mean, std))
        print("  优势 A_i    :", ["%+.2f" % a for a in adv], " ← (r_i - 均值) / 标准差")

        if std <= 1e-8:
            print("  [X] 组内奖励完全一致 -> 标准差为 0 -> 归一化本身失去定义")
            print("      -> 优势全为 0 -> 梯度为 0 -> 训练什么都没发生")
            return False
        print("  [OK] 组内奖励有差异 -> 标准差非 0 -> 优势非 0 -> 有梯度 -> 训练能推进")
        return True

    # —— 采样 ——
    with torch.no_grad():
        sampled = model.generate(
            **inputs, max_new_tokens=96, do_sample=True,
            temperature=0.9, top_p=0.95, num_return_sequences=G,
        )
    ok_sample = report("采样 G=%d，temperature=0.9" % G, extract(sampled))

    # —— 贪心 ——
    print()
    try:
        with torch.no_grad():
            greedy = model.generate(
                **inputs, max_new_tokens=96, do_sample=False, num_return_sequences=G
            )
        report("贪心 G=%d" % G, extract(greedy))
    except Exception as e:
        print("【贪心 G=%d】" % G)
        print("  库直接拒绝了 num_return_sequences=%d + do_sample=False：" % G)
        print("  ", type(e).__name__, str(e)[:120])
        print("  -> 因为库知道：贪心生成的 G 条必然一模一样，凑这个数没有意义。")
        print("  -> 这本身就是最好的说明。")

    # —— 验收 ——
    hr("Day 3 验收")
    if ok_sample:
        print("[OK] 你亲手复现了 GRPO 的 rollout，并且看到了组内奖励差异。")
        print("     这个 G 条回答的列表，就是 GRPO 训练循环的第 ① 步的输出。")
        print("\n     记住这条链：")
        print("     采样多样性 -> 组内奖励差异 -> 优势非 0 -> 梯度非 0 -> 参数更新")
        print("     任何一环断了，训练都会安静地失败（不报错，只是没效果）。")
    else:
        print("[!] 采样出来的 G 条回答奖励也全一样——这在 0.6B 上很正常。")
        print("    把 QUESTION 换一道更简单的题（比如 'What is 2 + 3?' EXPECTED='5'），")
        print("    或者把 temperature 调高，再试一次。")


# ══════════════════════════════════════════════════════════════
def main():
    print("\nDay 3 —— Transformers 手感")

    # 先检查模型在不在，别等到 from_pretrained 才报一个难懂的错
    if not os.path.isdir(MODEL):
        print("\n[X] 找不到模型目录：%s" % MODEL)
        print("    本脚本只加载本地模型，不会自动下载。")
        print("    请先按 `下载模型.md` 下载，或修改脚本顶部 MODEL 那一行。")
        return
    print("模型目录        :", MODEL)

    if DEV != "cuda":
        print("[!] 没检测到 GPU，会用 CPU 跑（0.6B 也能跑，就是慢一点）")

    tok = s0_load_tokenizer()
    s1_tokenize(tok)
    s2_chat_template(tok)

    model = s3_load_model()
    s3_inspect_logits(tok, model)
    s4_generate(tok, model)
    s5_rollout(tok, model)


if __name__ == "__main__":
    main()
