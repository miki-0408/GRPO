#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探测你装的 TRL 版本的实际 API。

为什么要这一步：
    TRL 的参数名在不同版本间改过很多次（比如 num_generations / num_return_sequences，
    max_new_tokens / max_completion_length，tokenizer / processing_class）。
    照抄网上的博客，八成撞 unexpected keyword argument。
    与其猜，不如直接从装好的包里问出来。

这个脚本不加载模型、不需要 GPU，几秒就跑完。

跑法：python probe_trl_api.py
然后把输出整段发给 Claude。
"""

import importlib
import inspect
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def hr(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


def version_of(name):
    try:
        m = importlib.import_module(name)
    except ImportError:
        return None
    return getattr(m, "__version__", "(无 __version__)")


def main():
    hr("0. 已安装的包")
    for pkg in ["torch", "transformers", "trl", "peft", "datasets",
                "accelerate", "bitsandbytes", "vllm", "unsloth"]:
        v = version_of(pkg)
        print("  %-16s %s" % (pkg, v if v else "未安装"))

    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError as e:
        print("\n[X] 导不出 GRPOTrainer / GRPOConfig：", e)
        print("    先确认 trl 装好了：pip install -U trl")
        return

    # ── GRPOTrainer 的构造签名 ────────────────────────────────
    hr("1. GRPOTrainer.__init__ 的参数")
    try:
        sig = inspect.signature(GRPOTrainer.__init__)
        for p in sig.parameters.values():
            if p.name == "self":
                continue
            d = "" if p.default is inspect.Parameter.empty else "  = %r" % (p.default,)
            print("  %-28s%s" % (p.name, d))
    except Exception as e:
        print("  取签名失败:", e)

    # ── 奖励函数是怎么传的 ────────────────────────────────────
    hr("2. 奖励函数相关的参数（找 reward）")
    try:
        names = [p.name for p in sig.parameters.values()]
        hit = [n for n in names if "reward" in n.lower()]
        print("  ", hit if hit else "没找到含 'reward' 的参数 —— 去文档确认传法")
    except NameError:
        pass

    # ── GRPOConfig 里你关心的字段 ─────────────────────────────
    hr("3. GRPOConfig 里需要关注的字段")
    print("  （这是 dataclass，下面只挑和 Day 5 手写版对应的）\n")

    keys = ["generation", "num_generations", "beta", "epsilon", "loss_type",
            "completion", "temperature", "top_p", "top_k",
            "scale_rewards", "num_iterations", "mask_truncated",
            "chat_template", "max_prompt", "reward_weights", "log_completions",
            "use_vllm", "importance_sampling"]
    try:
        fields = GRPOConfig.__dataclass_fields__
        shown = set()
        for name, spec in fields.items():
            if any(k in name for k in keys):
                d = getattr(spec, "default", "?")
                if d is not inspect.Parameter.empty and repr(d) != "<dataclasses._MISSING_TYPE object>":
                    pass
                print("  %-30s = %r" % (name, d))
                shown.add(name)
        if not shown:
            print("  （一个都没匹配上，说明这个版本的命名差别较大）")
    except Exception as e:
        print("  读取失败:", e)

    # ── 全部字段名（备用，方便你 grep）───────────────────────
    hr("4. GRPOConfig 全部字段名（只列名字，方便你搜索）")
    try:
        allnames = sorted(GRPOConfig.__dataclass_fields__.keys())
        print("  共 %d 个。含这些关键词的：" % len(allnames))
        for k in ["beta", "epsilon", "gen", "reward", "loss", "kl", "temp",
                  "len", "iter", "vllm", "scale", "mask"]:
            hit = [n for n in allnames if k in n.lower()]
            if hit:
                print("    [%s] %s" % (k, ", ".join(hit)))
    except Exception as e:
        print("  失败:", e)

    # ── 源码在哪（今天的重点：去读它）─────────────────────────
    hr("5. 源码位置 —— 这是你今天真正要读的东西")
    for cls, label in [(GRPOTrainer, "GRPOTrainer"), (GRPOConfig, "GRPOConfig")]:
        try:
            print("  %-14s %s" % (label, inspect.getfile(cls)))
        except Exception as e:
            print("  %-14s 取不到: %s" % (label, e))

    print("\n  打开 GRPOTrainer 那个文件，对照 TRL-实现对照.md 第 2 节的映射表，")
    print("  找出这几块分别在哪一行：")
    print("    - 生成 rollouts（对应你手写的 ①）")
    print("    - 算 rewards（你传进去的函数在这里被调用）")
    print("    - 算 advantages（对应 ③）")
    print("    - 算 per-token logprobs（对应 ④）")
    print("    - ratio / clip / KL（对应 ⑤）")
    print("    - mask 是怎么做的（你手写的那几行）")

    hr("6. 数据集的字段（GSM8K）")
    print("  跑这段可以确认 GSM8K 的字段名：")
    print("    from datasets import load_dataset")
    print("    d = load_dataset('openai/gsm8k', 'main', split='train')")
    print("    print(d.column_names); print(d[0])")


if __name__ == "__main__":
    main()
