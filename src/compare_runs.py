#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两次训练的指标 —— 用来做对照实验。

用法：
    python compare_runs.py ~/grpo_out ~/grpo_out_dapo
    python compare_runs.py ~/grpo_out ~/grpo_out_dapo --last 100   # 只看最后 100 条记录

为什么需要它：
    show_metrics.py 只能看一次 run。但"这个改动到底有没有用"必须两边比。
    这个脚本把两次 run 的关键指标并排打出来，并自动算首尾变化。

⚠️ 对比前先确认两件事（否则结论作废）：
    1. 两次用的是【同一份验证集】吗？
    2. 唯一的变量是什么？（只改一个地方才是干净的对照）
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 要对比的指标（按重要性排序）
TRACK = [
    ("eval_reward",                  "验证集通过率",  "高"),
    ("eval_reward_std",              "验证集奖励方差", "中"),
    ("eval_frac_reward_zero_std",    "验证集零方差组占比", "低"),
    ("eval_completions/mean_length", "验证集平均长度", "低"),
    ("eval_kl",                      "验证集 KL",     "低"),
    ("reward",                       "训练集奖励",    "高"),
    ("reward_std",                   "训练集奖励方差", "中"),
    ("frac_reward_zero_std",         "训练集零方差组占比", "高"),
    ("completions/mean_length",      "训练集平均长度", "高"),
    ("kl",                           "训练集 KL",     "中"),
    ("entropy",                      "熵（探索程度）", "中"),
    ("completions/clipped_ratio",    "截断比例",      "中"),
    ("loss",                         "损失",          "低"),
]


def find_state(root):
    root = os.path.expanduser(root)
    if os.path.isfile(root):
        return root
    hits = []
    for dp, _, fs in os.walk(root):
        if "trainer_state.json" in fs:
            hits.append(os.path.join(dp, "trainer_state.json"))
    return max(hits, key=os.path.getmtime) if hits else None


def load(root):
    p = find_state(root)
    if p is None:
        return None, None
    st = json.load(open(p, encoding="utf-8"))
    return p, st.get("log_history", [])


def series(hist, key):
    return [e[key] for e in hist if key in e and isinstance(e[key], (int, float))]


def quarters(vals):
    """返回 (前1/4 均值, 后1/4 均值)"""
    n = len(vals)
    if n < 4:
        return (sum(vals) / n if n else 0.0,) * 2
    q = max(1, n // 4)
    return sum(vals[:q]) / q, sum(vals[-q:]) / q


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    last = None
    if "--last" in sys.argv:
        i = sys.argv.index("--last")
        if i + 1 < len(sys.argv):
            last = int(sys.argv[i + 1])

    if len(args) < 2:
        sys.exit(__doc__)

    paths, hists = [], []
    for a in args[:2]:
        p, h = load(a)
        if h is None:
            sys.exit("在 %s 下找不到 trainer_state.json" % a)
        paths.append(p)
        hists.append(h[-last:] if last else h)

    print("=" * 96)
    for a, p in zip(args[:2], paths):
        print("  %-28s -> %s" % (a, p))
    print("  A = 第一次，B = 第二次。对比时确认【只有一处变量不同】。")
    if last:
        print("  只取最后 %d 条记录" % last)
    print("=" * 96)

    hdr = ("指标".ljust(20) + "A 前1/4".rjust(11) + "A 后1/4".rjust(11) +
           "B 前1/4".rjust(11) + "B 后1/4".rjust(11) +
           "A 变化".rjust(10) + "B 变化".rjust(10) + "  重要")
    print(hdr)
    print("-" * len(hdr))

    for key, label, imp in TRACK:
        sa, sb = series(hists[0], key), series(hists[1], key)
        if not sa and not sb:
            continue
        row = label[:18].ljust(20)
        a0 = a1 = b0 = b1 = float("nan")
        if sa:
            a0, a1 = quarters(sa)
            row += ("%.4g" % a0).rjust(11) + ("%.4g" % a1).rjust(11)
        else:
            row += "-".rjust(11) + "-".rjust(11)
        if sb:
            b0, b1 = quarters(sb)
            row += ("%.4g" % b0).rjust(11) + ("%.4g" % b1).rjust(11)
        else:
            row += "-".rjust(11) + "-".rjust(11)

        def trend(x0, x1):
            if x0 != x0 or x1 != x1 or x0 == 0:
                return "-"
            d = (x1 - x0) / abs(x0) * 100
            return "%+.1f%%" % d

        row += trend(a0, a1).rjust(10) + trend(b0, b1).rjust(10)
        row += "  " + imp
        print(row)

    print()
    print("=" * 96)
    print("怎么读这张表")
    print("=" * 96)
    print("  · 看【后1/4】那一列 —— 它是训练末期的水平，比首尾差更能说明问题")
    print("  · eval_reward 是唯一能证明『模型真的变强』的指标；train reward 涨不代表学没学到")
    print("  · completions/mean_length 一直涨要警惕：可能是 reward hacking（见 实验问题与解决办法.md 第 13 条）")
    print("  · frac_reward_zero_std 越高，越多算力白费（第 8、12 条）")
    print()
    print("  ⚠️ 两次 run 的验证集必须是同一份，否则 eval_* 的对比全部作废（第 14、15 条）")


if __name__ == "__main__":
    main()
