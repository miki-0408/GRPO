#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 trainer_state.json 里读训练全貌。

为什么需要它：
    终端里的输出会被那个大表格（log_completions）淹没，而且滚过去就没了。
    HF Trainer 其实把【每一步】的指标都记在 trainer_state.json 里了——
    那才是完整记录。这个脚本把它整理成一张能看的表。

跑法：
    python show_metrics.py                      # 默认读 ~/grpo_out/trainer_state.json
    python show_metrics.py 别的路径/trainer_state.json
    python show_metrics.py --all                # 把所有指标都列出来
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def find_state(root):
    """transformers 5.x 把 trainer_state.json 写在 checkpoint 子目录里，
    不一定是 output_dir 顶层。这里自动往下找，取最新的那个。"""
    root = os.path.expanduser(root)
    if os.path.isfile(root):
        return root
    hits = []
    for dirpath, _, files in os.walk(root):
        if "trainer_state.json" in files:
            hits.append(os.path.join(dirpath, "trainer_state.json"))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


show_all = "--all" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ROOT = args[0] if args else "~/grpo_out"

PATH = find_state(ROOT)
if PATH is None:
    sys.exit("在 %s 下找不到 trainer_state.json\n（跑完 train_trl.py 之后才有）" % ROOT)

state = json.load(open(PATH, encoding="utf-8"))
hist = state.get("log_history", [])
if not hist:
    sys.exit("log_history 是空的")

# ── 收集所有出现过的指标名 ──────────────────────────────────────
keys = []
for e in hist:
    for k in e:
        if k not in keys:
            keys.append(k)

print("=" * 78)
print("文件: %s" % PATH)
print("记录条数: %d   global_step 范围: %s ~ %s"
      % (len(hist), state.get("global_step"), hist[0].get("step")))
print("=" * 78)

# ── 挑出真正关心的指标 ─────────────────────────────────────────
PREFERRED = ["reward", "reward_std", "rewards/reward_fn/mean",
             "completions/mean_length", "completions/max_length",
             "kl", "loss", "learning_rate", "entropy",
             "num_tokens", "completions/clipped_ratio"]
if show_all:
    picked = keys
else:
    picked = [k for k in keys
              if any(p in k.lower() for p in
                     ["reward", "kl", "loss", "length", "entropy", "lr", "learning"])]

print("\n全部指标名（%d 个）:" % len(keys))
for k in keys:
    print("   ", k)

print("\n" + "=" * 78)
print("关键指标逐步表")
print("=" * 78)

hdr = "step".ljust(6) + "".join(k[-16:].rjust(18) for k in picked)
print(hdr)
print("-" * len(hdr))

for e in hist:
    if "loss" not in e and not any(p in str(e) for p in ["reward", "kl"]):
        continue                       # 跳过只有 epoch 之类的小记录
    row = str(e.get("step", "")).ljust(6)
    for k in picked:
        v = e.get(k)
        if v is None:
            row += "-".rjust(18)
        elif isinstance(v, float):
            row += ("%.5g" % v).rjust(18)
        else:
            row += str(v).rjust(18)
    print(row)

# ── 首尾对比 ───────────────────────────────────────────────────
print("\n" + "=" * 78)
print("首尾对比（判断有没有在学）")
print("=" * 78)
for k in picked:
    vals = [e[k] for e in hist if k in e and isinstance(e[k], (int, float))]
    if len(vals) < 4:
        continue
    n = max(1, len(vals) // 4)
    a, b = sum(vals[:n]) / n, sum(vals[-n:]) / n
    arrow = "↑" if b > a * 1.02 else ("↓" if b < a * 0.98 else "≈")
    print("  %-34s 前1/4=%.5g   后1/4=%.5g   %s" % (k, a, b, arrow))

print()
print("奖励不动 / loss ≈ 0，通常是『零方差组』太多 —— 见 day5 的讨论。")
print("数据筛选是 Day 7 的任务。")
