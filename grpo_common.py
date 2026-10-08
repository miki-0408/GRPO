#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 项目公用逻辑。

为什么单独抽一个文件：
    数据筛选（filter_data.py）和训练（train_trl.py）必须用【完全一致】的
    奖励规则和 prompt 格式。各写一份的话，两者一旦漂移，筛出来的
    "有梯度"数据到训练时就变成"没梯度"——而且不报错、不警告。
    这类静默的不一致是能毁掉整条数据管线的 bug。

    所以：**任何跟"怎么算对错""怎么构造 prompt"有关的逻辑，都只准写在这里。**
"""

import os
import re

# 国内直连 huggingface.co 会卡住（不是报错），必须在 import 之前设
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_DIR = os.path.expanduser("~/models/Qwen3-0.6B")

# ⚠️ 训练和筛选必须用【一模一样】的后缀。
#    改了这里，两边同时生效 —— 这正是抽成公用模块的意义。
PROMPT_SUFFIX = ("\n\nAnswer with just the number. "
                 "Do not show your work. Do not explain.")


# ══════════════════════════════════════════════════════════════
# 答案抽取
# ══════════════════════════════════════════════════════════════
def extract_gold(answer_text):
    """GSM8K 标准答案格式：'... #### 72'"""
    return str(answer_text).split("####")[-1].strip()


def extract_pred(completion):
    """从模型输出里抽答案。优先级：\\boxed{} > '#### x' > answer is N > 最后一个数字。

    抽取规则太松 → 模型学会刷分；太严 → 正确回答拿不到分。
    这个取舍本身就是 RLVR 的工程难点之一。
    """
    completion = to_text(completion)
    m = re.search(r"\\boxed\{([^}]+)\}", completion)
    if m:
        return m.group(1).strip()
    m = re.search(r"####\s*([^\n]+)", completion)
    if m:
        return m.group(1).strip()
    # 不能用 [\d,\.]+ —— 会把句末句号也吞进去（"300." -> "300."）
    m = re.search(r"(?:answer|equals?|is)\s*[:=]?\s*\$?(-?\d[\d,]*(?:\.\d+)?)", completion, re.I)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+\.?\d*", completion)
    return nums[-1] if nums else ""


def norm(s):
    """归一化：'300.' / '300' / '300.0' / '1,200' 都变成同一个字符串"""
    try:
        f = float(str(s).replace(",", "").strip())
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(s).strip()


def to_text(c):
    """把 TRL 传进来的 completion 统一成字符串。

    TRL 会因为数据格式不同而传不同的东西：
      - 纯文本 prompt   -> list[str]
      - 对话格式 prompt -> list[list[dict]]，形如 [[{'role':'assistant','content':'...'}]]
    我们用对话格式（为了能传 chat_template_kwargs 关思考模式），所以必须处理第二种。
    """
    if isinstance(c, str):
        return c
    if isinstance(c, (list, tuple)) and len(c) > 0:
        last = c[-1]
        if isinstance(last, dict):
            return last.get("content", "") or ""
        if isinstance(last, str):
            return last
        return to_text(last)
    return str(c)


def is_correct(completion, gold_answer):
    """判断一条回答对不对。**筛选和训练都用这一个函数。**"""
    return norm(extract_pred(completion)) == norm(extract_gold(gold_answer))


def reward_from_texts(texts, golds):
    """纯文本批处理版本 —— 筛选脚本直接用这个"""
    return [1.0 if is_correct(t, g) else 0.0 for t, g in zip(texts, golds)]


def make_reward_fn(debug=False):
    """生成 TRL GRPOTrainer 用的 reward_funcs（批量、从 kwargs 拿标准答案）。

    debug=True 时第一次调用会打印收到了哪些字段、以及前两条的抽取结果 ——
    排查"奖励恒为 0"时非常有用。
    """
    seen = {"kw": False, "shown": False}

    def reward_fn(completions, **kwargs):
        if debug and not seen["kw"]:
            print("\n[诊断] reward_fn 收到的 kwargs 键：", sorted(kwargs.keys()))
            print("        completions 类型 =", type(completions).__name__,
                  "| 第 0 个元素类型 =", type(completions[0]).__name__)
            seen["kw"] = True

        gold = None
        for key in ("answer", "gold", "gold_answer", "solution", "label"):
            if key in kwargs and kwargs[key] is not None:
                gold = kwargs[key]
                break
        if gold is None:
            raise KeyError("reward_fn 没拿到标准答案；看上面 [诊断] 打印的真实列名")

        out = []
        for i, (c, g) in enumerate(zip(completions, gold)):
            ok = is_correct(c, g)
            if debug and not seen["shown"] and i < 2:
                print("        [抽取] pred=%-10r gold=%-10r -> %s"
                      % (norm(extract_pred(c)), norm(extract_gold(g)), "对" if ok else "错"))
            out.append(1.0 if ok else 0.0)
        seen["shown"] = True
        return out

    return reward_fn


# ══════════════════════════════════════════════════════════════
# prompt 构造
# ══════════════════════════════════════════════════════════════
def build_messages(question):
    """返回 TRL 数据集需要的 prompt 字段（对话格式）"""
    return [{"role": "user", "content": question + PROMPT_SUFFIX}]


def build_prompt_text(tok, question, thinking=False):
    """直接生成 prompt 字符串（筛选脚本自己 generate 时用）"""
    try:
        return tok.apply_chat_template(
            build_messages(question), tokenize=False,
            add_generation_prompt=True, enable_thinking=thinking)
    except TypeError:
        return tok.apply_chat_template(
            build_messages(question), tokenize=False, add_generation_prompt=True)
