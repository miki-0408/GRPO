#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 6 —— 用 TRL 的 GRPOTrainer 跑 GRPO（GSM8K）

⚠️ 先读 `TRL-实现对照.md`，重点看第 2 节那张映射表。

和 grpo_from_scratch.py 的关系：
    你手写的 277 行 = 这个脚本里【你唯一要写的 reward 函数】+ Trainer 内部实现。
    今天的产出不是"跑起来"，是能指着 grpo_trainer.py 说出每一块对应你手写的哪几行。

环境实测（probe_trl_api.py 的输出）：
    transformers 5.18.0 / trl 1.14.1 / peft 0.21.2 / torch 2.14.1+cu130
    vllm 未安装（所以 use_vllm 保持 False）

跑法：python train_trl.py
"""

import os
import sys

# ⚠️ 必须在 import datasets / transformers 之前设。
# 国内直连 huggingface.co 会"卡住"（一直等超时），而不是报错——
# 排查起来很费时间。这行让脚本不依赖 shell 里有没有 export 过。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# 答案抽取 / prompt 构造 / 奖励函数都在这里 ——
# 和 filter_data.py 共用同一份，杜绝"筛选用一套、训练用另一套"的静默不一致
# 把脚本所在目录【和项目根】都加进搜索路径 ——
# 这样无论 grpo_common.py 在根目录还是同级目录，都能导入到。
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from grpo_common import MODEL_DIR, build_messages, make_reward_fn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# MODEL_DIR 从 grpo_common 导入（那边是唯一的真源）
OUT_DIR = os.path.expanduser("~/grpo_out")
DATA_DIR = os.path.expanduser("~/grpo_data")

# ── 超参（对照你 day5 的那几个）──────────────────────────────────────
G            = 8      # num_generations —— 你手写的 G
MAX_COMPLETE = 384    # max_completion_length —— 你手写的 MAX_NEW
                      # Day 6 实测 192 时大半回答被截断（max_length 撞上限、
                      # max_terminated_length=0），零方差组一半是这么来的。放宽到 384
BETA         = 0.04   # KL 系数 —— 你手写的 BETA（注意 TRL 默认是 0.0！）
EPSILON      = 0.2    # clip 范围 —— 你手写的 EPS
LR           = 1e-5
MAX_STEPS    = 200
N_QUESTIONS  = 300    # 训练集条数（先用一小撮 GSM8K，跑通了再加）
N_EVAL       = 60     # 验证集条数 —— 从 GSM8K 里【另取】一批，和训练集不重叠
EVAL_EVERY   = 20     # 每多少步验证一次

# 注意：TRL 要求 有效 batch（batch × 梯度累积）能被 num_generations 整除
BATCH, GRAD_ACC = 1, 8


def hr(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# ══════════════════════════════════════════════════════════════
# 奖励函数
#
# 抽答案 / 判对错 / 批量包装 的逻辑全在 grpo_common.py ——
# 和 filter_data.py 共用同一份。这样"筛选用什么规则"和"训练用什么规则"
# 永远一致；不一致的话，筛出来的"有梯度"数据到训练时会静默变成"没梯度"。
#
# debug=True：第一次调用时打印收到的 kwargs 和前两条抽取结果，
#             排查"奖励恒为 0"时第一眼就看它。
# ══════════════════════════════════════════════════════════════
reward_fn = make_reward_fn(debug=True)



def main():
    if not os.path.isdir(MODEL_DIR):
        sys.exit("找不到模型目录：%s" % MODEL_DIR)
    if not torch.cuda.is_available():
        sys.exit("没有 GPU")

    hr("准备数据")
    # ★ 优先读【筛选过】的数据集（filter_data.py 的产物）。
    #   没筛过的 GSM8K 里大量题是"8 条全对"或"8 条全错"，组内标准差为 0，
    #   优势恒为 0 —— 白算。Day 6/7 实测约 40~50% 的组是这样。
    train_path = os.path.join(DATA_DIR, "train")
    eval_path = os.path.join(DATA_DIR, "eval")
    if os.path.isdir(train_path):
        ds = load_from_disk(train_path)
        eval_ds = load_from_disk(eval_path)
        print("读的是【筛选后】的数据:", DATA_DIR)
    else:
        print("[!] 没找到 %s —— 退回用未筛选的 GSM8K。" % DATA_DIR)
        print("    先跑 python filter_data.py 会好很多（零方差组能少一半以上）。")
        raw = load_dataset("openai/gsm8k", "main", split="train")
        fmt = lambda x: {"prompt": build_messages(x["question"])}
        ds = raw.select(range(min(N_QUESTIONS, len(raw)))).map(fmt)
        eval_ds = raw.select(range(N_QUESTIONS,
                                   min(N_QUESTIONS + N_EVAL, len(raw)))).map(fmt)

    print("训练条数:", len(ds), " 验证条数:", len(eval_ds))
    print("\n样例 prompt:")
    print(ds[0]["prompt"][0]["content"][:200])
    print("对应答案:", ds[0]["answer"])


    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    hr("配置 GRPO")
    cfg = GRPOConfig(
        output_dir=OUT_DIR,
        # ↓ 对应你手写的 G
        num_generations=G,
        # ↓ 对应你手写的 MAX_NEW。TRL 默认 512，8GB 上必须压
        max_completion_length=MAX_COMPLETE,
        temperature=1.0,
        top_p=1.0,
        # ↓ 对应你手写的 BETA。⚠️ TRL 默认是 0.0 = 不加 KL 惩罚，这里显式开
        beta=BETA,
        # ↓ 对应你手写的 EPS
        epsilon=EPSILON,
        # ↓ 对应你 Day 5 讨论的 K（inner update 轮数）。=1 时 ratio 恒为 1
        num_iterations=1,
        # ↓ 默认是 'dapo'！显式写成 'grpo'，Day 8 你正好可以对照
        loss_type="grpo",
        # ↓ 被截断的回答不参与 loss —— 它们必然没有完整答案，只贡献噪声。
        #   Day 6 实测：约 1/4 的组是"8 条全被截断"，全给 0 分。这也是 DAPO/R1 的做法
        mask_truncated_completions=True,
        # ★★★ 关掉 Qwen3 思考模式 —— 就是这里！
        #     不设的话模型会先写一大段思考被截断，reward 恒为 0
        chat_template_kwargs={"enable_thinking": False},
        # 训练规模
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        max_steps=MAX_STEPS,
        # ★ 默认是 "linear"，会在 max_steps 处一路衰减到 0。
        #   Day 6 实测：60 步的 run 里 LR 从 1e-5 掉到 1.67e-7 —— 后一半几乎没在学。
        #   步数少的时候必须用 constant，否则等于自废一半训练。
        lr_scheduler_type="constant",
        # 验证：eval_dataset 本身是 Trainer 的参数（见下面），
        # 但"什么时候验证"是配置项 
        eval_strategy="steps",
        eval_steps=EVAL_EVERY,
        # ⚠️ 规则和训练一样：全局 batch 必须能被 num_generations 整除
        #    （一个 prompt 的 G 条回答必须在同一个 batch 里，不能跨 batch 拆开）
        #    验证时没有梯度累积，所以全局 batch 就是 per_device_eval_batch_size
        per_device_eval_batch_size=4,
        num_generations_eval=4,        # 验证时每组采 4 条，够估出通过率又不费时
        logging_steps=1,
        save_steps=MAX_STEPS,          # 只存最后
        bf16=True,
        gradient_checkpointing=True,
        # 8GB 上省显存
        optim="adamw_8bit",
        # 直接打几条回答出来看
        log_completions=True,
        num_completions_to_print=2,
        report_to="none",
        seed=0,
    )

    hr("创建 GRPOTrainer")
    print("注意：__init__ 里【没有 ref_model 参数】。")
    print("TRL 1.x 用同一个模型、关掉 adapter 来当参考模型 —— 省掉一整份权重。")
    print("（机制在 grpo_trainer.py 第 2779 行：use_adapter(model, adapter_name=None)）")
    print("（找不到 disable_adapter 是正常的，这版用的是 use_adapter）\n")

    trainer = GRPOTrainer(
        model=MODEL_DIR,                 # 传路径，Trainer 自己加载
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        # ★ eval_dataset 是【Trainer 的参数】，不是 GRPOConfig 的字段。
        #   判据：凡是"数据集"都在 Trainer，凡是"怎么跑"都在 Config。
        eval_dataset=eval_ds,
        processing_class=tok,            # ← 是 processing_class，不是 tokenizer
        peft_config=LoraConfig(          # ← LoRA 直接配在这里
            r=16, lora_alpha=32, lora_dropout=0.0,
            task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
        ),
    )

    hr("开始训练")
    print("观察三件事：")
    print("  1. reward 有没有往上走")
    print("  2. completion length 是不是稳定（暴涨说明思考模式没关掉）")
    print("  3. 第一条 [诊断] 打印告诉你 reward_fn 拿到了哪些字段\n")

    trainer.train()

    hr("完成")
    print("模型存在:", OUT_DIR)
    print()
    print("接下来（今天真正的重点）：打开")
    print("  %s/trl/trainer/grpo_trainer.py" % sys.prefix)
    print("对照 TRL-实现对照.md 第 2 节，找出这几块在第几行：")
    print("  - 生成 rollouts           （你手写的 ①）")
    print("  - 调用 reward_funcs       （你手写的 ②）")
    print("  - 算 advantages           （你手写的 ③）")
    print("  - 算 per-token logprobs   （你手写的 ④）")
    print("  - ratio / clip / KL       （你手写的 ⑤）")
    print("  - mask 只在 completion 上算 loss（你手写的那两行）")


if __name__ == "__main__":
    main()
