#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 5 —— 手写一个最小 GRPO

⚠️ 先读 `GRPO-原理.md`，再动手。

这个脚本不依赖 TRL。所有零件你都见过：
  - 训练循环骨架 ...... Day 2
  - rollout + 组内优势 ... Day 3
  - logprob / ratio / KL . 今天

有 4 个 TODO，全部填完才能跑通。跑法：

    source ~/rl-llm/bin/activate
    python grpo_from_scratch.py

成功的话你会看到 reward 曲线上升，以及"零方差组"的占比统计。
"""

import os
import re
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_DIR = os.path.expanduser("~/models/Qwen3-0.6B")

# ── 超参 ──────────────────────────────────────────────────────────────
G          = 8        # 每个问题生成几条回答（组内样本数）
MAX_NEW    = 64      # 最多生成多少 token
STEPS      = 40       # 训练多少步
LR         = 1e-5
EPS        = 0.2      # PPO clip 范围
BETA       = 0.01     # KL 惩罚系数
LORA_R     = 16


def hr(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


# ══════════════════════════════════════════════════════════════
# 数据：随机两位数加法。这个难度对 0.6B 刚好在能力边界上，
#       所以组内回答会有的对有的错 —— 这正是 GRPO 需要的。
# ══════════════════════════════════════════════════════════════
def sample_question(rng):
    a = rng.randint(50, 99)
    b = rng.randint(50, 99)
    return a, b, a * b





def build_prompt(tok, a, b):
    msgs = [{"role": "user",
             "content": "What is %d * %d? Answer with just the number." % (a, b)}]

    # ⚠️ 必须显式传 enable_thinking=False 才能关掉思考模式。
    #
    # Qwen3 模板里写的是：
    #     {%- if enable_thinking is defined and enable_thinking is false %}
    #         {{- '<think>\n\n</think>\n\n' }}
    #     {%- endif %}
    # 只有【显式传 False】才会插入那个空的 think 块（= 让模型跳过思考直接答）。
    # 不传参数 = 条件不成立 = 什么都不加 = 模型自己生成一大段思考过程 = 思考模式开着。
    #
    # 不关掉的后果很具体：思考过程被 MAX_NEW 截断，答案根本没生成出来，
    # reward_fn 抽到的是中间步骤的数字 → 奖励恒为 0 → 组内零方差 → 梯度为 0。
    # 而且它不报错，你只会看到 reward 一直不动。
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # 换成了不支持这个参数的模型/模板，退回普通调用
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ══════════════════════════════════════════════════════════════
# 奖励：可验证规则。抽取回答里最后一个数字，和正确答案比。
#       —— 这就是 RLVR 里那个 verifiable reward。
# ══════════════════════════════════════════════════════════════
def reward_fn(completion, gold):
    nums = re.findall(r"-?\d+", completion)
    if not nums:
        return 0.0
    return 1.0 if int(nums[-1]) == gold else 0.0


# ══════════════════════════════════════════════════════════════
# TODO 1 —— 取每个 token 的 log 概率
# ══════════════════════════════════════════════════════════════
def token_logprobs(model, input_ids, attention_mask):
    """返回 (batch, seq-1)，每个位置是"模型给实际那个 token 打的 log 概率"。

    提示（三步）：
      1. out = model(input_ids, attention_mask=attention_mask)
         logits = out.logits[:, :-1, :]     # 去掉最后一个位置
         target = input_ids[:, 1:]          # 右移一位，才是要预测的目标
      2. logp = F.log_softmax(logits.float(), dim=-1)
         # 形状 (batch, seq-1, vocab_size)
      3. 从最后一维里，把 target 对应位置的值取出来：
         logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    为什么要错位：位置 t 的输出，是用来预测位置 t+1 的 token 的。
    """
    out = model(input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]  # 去掉最后一个位置
    target = input_ids[:, 1:]       # 右移一位，才是要预测的目标
    logp = F.log_softmax(logits.float(), dim=-1)  # 形状 (batch, seq-1, vocab_size)
    return logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    raise NotImplementedError("TODO 1: token_logprobs 还没实现")


# ══════════════════════════════════════════════════════════════
# TODO 2 —— 组内归一化优势
# ══════════════════════════════════════════════════════════════
def group_advantage(rewards):
    """rewards: list[float]，同一问题的 G 条回答的奖励。返回同长度的 list。

    这就是你在 Day 3 亲手算过的：
        A_i = (r_i - 组内均值) / 组内标准差

    注意标准差为 0 的情况（全对或全错）—— 那时优势应该全是 0。
    避免除零：分母加个极小量，或者判断 std < 1e-8 时直接返回全 0。
    """
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    if std < 1e-8:
        return [0.0] * len(rewards)
    return [(r - mean) / std for r in rewards]
    raise NotImplementedError("TODO 2: group_advantage 还没实现")


# ══════════════════════════════════════════════════════════════
# TODO 3 + 4 —— GRPO 的损失
# ══════════════════════════════════════════════════════════════
def grpo_loss(logp, logp_old, logp_ref, adv, mask):
    """
    logp, logp_old, logp_ref : (B*G, seq-1)  三个模型对同一批 token 的 log 概率
    adv                      : (B*G, 1)      每条回答的优势（广播到所有 token）
    mask                     : (B*G, seq-1)  1=回答部分，0=prompt 部分

    返回标量 loss。

    TODO 3 —— ratio 与 clip：
        ratio   = torch.exp(logp - logp_old)
        clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS)
        surrogate = torch.min(ratio * adv, clipped * adv)

      ratio 是"新旧策略给同一个 token 的概率之比"。

      ⚠️ 重要：这个脚本是"生成一次 → 更新一次"，所以 logp_now 和 logp_old
         必然完全相同，ratio 恒等于 1.000，**clip 实际上一行都没生效**。

         这不是 bug。clip 真正起作用需要【一个 batch 的样本被更新多次】
         （多轮 inner update，mini-batch 循环）——第二次起策略才和采样时不同。
         TRL 的 GRPOTrainer 有 num_iterations 参数控制这件事。
         你现在只要理解 clip 在防什么就够了。
    
    TODO 4 —— KL 惩罚（GRPO 论文用的 k3 估计量，恒 ≥ 0）：
        delta = logp_ref - logp
        kl    = torch.exp(delta) - delta - 1

      注意：GRPO 的 KL 是【直接加在损失里】的，
      而 PPO 是折进每个 token 的奖励里 —— 别搞混。

    最后：
        per_token = -surrogate + BETA * kl
        loss = (per_token * mask).sum() / mask.sum()
      —— 只在 mask=1 的（回答）位置上求平均。
    """
    ratio = torch.exp(logp - logp_old)
    clipped = torch.clamp(ratio, 1 - EPS, 1 + EPS)
    surrogate = torch.min(ratio * adv, clipped * adv)
    delta = logp_ref - logp
    kl = torch.exp(delta) - delta - 1
    per_token = -surrogate + BETA * kl
    loss = (per_token * mask).sum() / mask.sum()
    return loss
    raise NotImplementedError("TODO 3+4: grpo_loss 还没实现")


# ══════════════════════════════════════════════════════════════
def main():
    if not os.path.isdir(MODEL_DIR):
        sys.exit("找不到模型目录：%s" % MODEL_DIR)
    if not torch.cuda.is_available():
        sys.exit("没有 GPU。这个脚本在 CPU 上跑不动。")

    torch.manual_seed(0)
    import random
    rng = random.Random(0)

    hr("加载模型")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token          # Qwen3 默认没有 pad token，补一个

    # policy：要训练的模型，挂上 LoRA（否则 0.6B 全参微调 + 参考模型会爆显存）
    policy = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda")
    policy = get_peft_model(policy, LoraConfig(
        r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.0,
        task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
    ))
    policy.print_trainable_parameters()

    # reference：RL 之前的那个模型，冻结。就是同一份基础权重再加载一次。
    ref = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16).to("cuda")
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    print("显存占用(GB) : %.2f" % (torch.cuda.memory_allocated() / 1024**3))

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=LR
    )

    hr("开始训练")
    print("每步：1 个问题 → 生成 %d 条回答 → 算奖励 → 算优势 → 更新" % G)
    print("注意观察两件事：reward 有没有上升；『零方差组』占比高不高。\n")

    policy.train()
    hist_reward, hist_zero = [], []

    for step in range(STEPS):
        a, b, gold = sample_question(rng)
        prompt = build_prompt(tok, a, b)
        enc = tok(prompt, return_tensors="pt").to("cuda")
        prompt_len = enc.input_ids.shape[1]

        # ── ① rollout（Day 3 学过）─────────────────────────────
        with torch.no_grad():
            seqs = policy.generate(
                **enc,
                do_sample=True, temperature=1.0, top_p=1.0,
                num_return_sequences=G,
                max_new_tokens=MAX_NEW,
                pad_token_id=tok.pad_token_id,
            )
        # print("step %2d | %2d+%2d | 第 %d 条回答：%s" % (step, a, b, 1, tok.decode(seqs[0, prompt_len:], skip_special_tokens=True)))
        # ── ② 奖励（规则函数）─────────────────────────────────
        completions = [tok.decode(s[prompt_len:], skip_special_tokens=True) for s in seqs]
        rewards = [reward_fn(c, gold) for c in completions]
        if step % 8 == 0:
            for c, r in zip(completions, rewards):
                print("    [%d] %s" % (r, c[:60].replace("\n", " ")))

        # ── ③ 组内优势 ───────────────────────────────────────
        adv_list = group_advantage(rewards)
        mean_r = sum(rewards) / len(rewards)
        zero_var = (max(rewards) == min(rewards))

        # ── ④ logprob：三个模型给同一批 token 打分 ────────────
        attn = torch.ones_like(seqs)
        with torch.no_grad():
            logp_old = token_logprobs(policy, seqs, attn)     # rollout 那一刻的策略
            logp_ref = token_logprobs(ref,    seqs, attn)     # 参考模型
        logp_now = token_logprobs(policy, seqs, attn)         # 当前策略，要梯度

        # 掩码：只在【回答】部分算损失，prompt 部分不算
        mask = torch.zeros_like(logp_now)
        mask[:, prompt_len - 1:] = 1.0

        adv = torch.tensor(adv_list, dtype=torch.float32, device="cuda").unsqueeze(1)

        # ── ⑤ 损失 ──────────────────────────────────────────
        loss = grpo_loss(logp_now, logp_old, logp_ref, adv, mask)

        # ── ⑥ 更新（Day 2 学过）──────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        hist_reward.append(mean_r)
        hist_zero.append(float(zero_var))

        if step % 4 == 0 or step == STEPS - 1:
            print("step %2d | %2d*%2d | 奖励 %.2f | 零方差组 %s | loss %+.4f | ratio≈%.3f"
                  % (step, a, b, mean_r, "是" if zero_var else "否",
                     loss.item(),
                     float(torch.exp(logp_now - logp_old)[mask.bool()].mean().detach())
                  ))


    hr("结果")
    n = len(hist_reward)
    first, last = sum(hist_reward[:n // 4]) / (n // 4), sum(hist_reward[-n // 4:]) / (n // 4)
    print("前 1/4 步平均奖励 : %.3f" % first)
    print("后 1/4 步平均奖励 : %.3f" % last)
    print("零方差组占比      : %.1f%%" % (100 * sum(hist_zero) / n))
    print()
    if last > first:
        print("[OK] 奖励上升了 —— 你手写的 GRPO 跑通了。")
    else:
        print("[!] 奖励没明显上升。常见原因：步数太少 / LR 不合适 / 问题太难")
        print("    先用 『零方差组占比』判断：如果它很高，说明模型对这类问题的回答")
        print("    全都对或全都错，优势恒为 0 —— 那就是 DAPO 动态采样要解决的问题。")
    print()
    print("对照检查：ratio 那一列应该一直非常接近 1.000。")
    print("如果不是，说明 logp_old 取错了 —— 它必须在【更新之前】用 no_grad 取。")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        hr("还没写完")
        print(e)
        print("\n本脚本有 4 个 TODO：token_logprobs / group_advantage / grpo_loss 里的 ratio-clip 与 KL。")
        print("全部写完后重跑：python grpo_from_scratch.py")
