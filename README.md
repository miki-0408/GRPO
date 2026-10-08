# GRPO 实现与 RLVR 数据管线

在 **Qwen3-0.6B** 上用 **GRPO（Group Relative Policy Optimization）** + **可验证奖励（RLVR）** 做 GSM8K 数学推理训练。

包含两部分：**不依赖任何 RL 库、用 PyTorch 手写的 GRPO 完整实现**，以及在其之上的 **TRL `GRPOTrainer` 生产级训练链路**与**数据筛选管线**。

---

## 核心结果

在同一模型、同样步数、同样超参下，**只把训练数据从"随机采样 GSM8K"换成"按组内奖励方差筛选"**：

| 指标 | 未筛选数据 | 筛选后数据 |
|---|---|---|
| 验证集通过率**趋势** | 0.283 → **0.250**（下降） | 0.263 → **0.469**（上升） |
| 训练集奖励 | ~0.32（持平） | 0.32 → ~0.55 |
| 零方差样本组占比 | ~40–50% | 显著下降 |

> ⚠️ **两次 run 使用的是不同的验证集**（GSM8K 的不同子集），因此绝对数值不可直接比较；可比较的是**各自 run 内的变化趋势**。这是本项目在实验设计上的一处疏漏，记录在 [`docs/实验问题与解决办法.md`](docs/实验问题与解决办法.md) 第 15 条。

> **这是一个受控对照实验**：唯一的变量是训练数据。未筛选时约 **70% 的样本组奖励方差为零、优势恒为 0、不产生任何梯度**——算力大部分被浪费。
>
> 完整的实验记录与踩坑分析见 [`docs/实验问题与解决办法.md`](docs/实验问题与解决办法.md)。

---

## 两个实现

### 1. 手写 GRPO（`src/grpo_from_scratch.py`）

不依赖 TRL 等 RL 库，用 PyTorch 从零实现完整训练循环：

- **rollout**：同一 prompt 采样 G 条回答
- **组内归一化优势**：`A_i = (r_i - 组内均值) / 组内标准差`
- **per-token 对数概率**：含自回归的错位取法（`logits[:, :-1]` ↔ `input_ids[:, 1:]`）
- **PPO 裁剪目标**：`-min(ratio·A, clip(ratio)·A)`
- **KL 惩罚**：`exp(Δ) - Δ - 1`（GRPO 论文的无偏估计量）
- **回答段损失掩码**：prompt 部分不计入损失

### 2. TRL 版本（`src/train_trl.py`）

用 `GRPOTrainer` 做生产级训练，并**与手写版逐项对照**，验证实现一致性：

| 手写版 | TRL `grpo_trainer.py` |
|---|---|
| `group_advantage()` | 第 2841 / 2855 行 |
| `ratio` + `clip` | 第 3179–3182 行 |
| `kl = exp(Δ) - Δ - 1` | 第 3162 行（**公式完全一致**） |
| `token_logprobs()` | `_get_per_token_logps_and_entropies`（第 1370 行） |

对照结果整理在 [`docs/GRPOTrainer-源码地图.md`](docs/GRPOTrainer-源码地图.md)，含**从实际安装的源码中定位出的行号**。

> 对照中发现一处实质差异：手写版按全局 token 平均损失，TRL 先按序列平均再对序列取平均（**长回答的权重不同**）。这正是 DAPO「token 级 loss」要解决的问题。

---

## 数据管线

`src/filter_data.py`：对每道候选题目采样 G 次，**只保留组内奖励有差异的题**（全对或全错的题不产生梯度，直接剔除）。

实测分类统计（800 道候选）：

| 类别 | 数量 | 占比 |
|---|---|---|
| 全对 | 342 | 42.8% |
| 全错 | 228 | 28.5% |
| **有差异（保留）** | **230** | **28.8%** |

**验证集不做筛选**，保持原始分布——否则 `eval_reward` 会天然固定在 0.5 附近，量的是筛选标准而不是模型能力。

---

## 目录

```
.
├── grpo_common.py              公用逻辑（答案抽取 / prompt 构造 / 奖励函数）
├── src/
│   ├── grpo_from_scratch.py    手写 GRPO 完整实现
│   ├── train_trl.py            基于 TRL GRPOTrainer 的训练
│   ├── filter_data.py          数据筛选管线
│   ├── probe_difficulty.py     探测模型的能力边界（筛选器的前身）
│   ├── show_metrics.py         从 trainer_state.json 读训练全貌
│   ├── compare_runs.py         两次训练并排对比（做消融）
│   └── probe_trl_api.py        探测当前 TRL 版本的 API
├── docs/
│   ├── GRPO-原理.md            优势 / 基线 / ratio / clip / KL
│   ├── TRL-实现对照.md          手写版与 GRPOTrainer 的映射
│   ├── GRPOTrainer-源码地图.md   源码行号对照表
│   ├── DAPO-对照.md            DAPO 的四个改进 ⇄ 本项目踩过的四个坑
│   └── 实验问题与解决办法.md     16 个实测故障的现象/根因/解决/启示
└── tutorials/
    ├── generate_from_scratch.py  4 行复现 model.generate()，证明自回归只是个循环
    ├── pytorch_basics.py         PyTorch 手感练习
    ├── transformers_basics.py    transformers 手感练习
    ├── PyTorch-基础.md / Transformers-基础.md
    └── 下载模型.md
```

---

## 快速开始

```bash
pip install -r requirements.txt

# 0. 把 Qwen3-0.6B 下到本地（见 tutorials/下载模型.md），改 grpo_common.py 里的 MODEL_DIR

# 1. 筛选训练数据（一次性，约 10~25 分钟）
python src/filter_data.py

# 2. 训练
python src/train_trl.py

# 3. 看指标
python src/show_metrics.py
```

**国内必须先设 HuggingFace 镜像**（否则会静默卡住而不是报错）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

（`grpo_common.py` 已在 import 前 `setdefault` 兜底。）

---

## 实测发现的工程问题

完整版见 [`docs/实验问题与解决办法.md`](docs/实验问题与解决办法.md)，摘要：

1. **截断是零方差的主因** —— 约 1/4 的样本组是"8 条回答全被长度上限截断"，全给 0 分。只调题目难度治不好。
2. **默认学习率调度在短训练下自残** —— `linear` 调度在 `max_steps` 处衰减到 0；60 步的 run 里后一半几乎没在学。
3. **Qwen3 思考模式默认是开的** —— 其 chat template 只在**显式传 `enable_thinking=False`** 时才关闭，不传参数等于开启。
4. **未筛选时 70%+ 的算力花在不产生梯度的样本上** —— 数据质量比调超参重要得多。
5. **回答越训越长（length hacking）** —— 奖励函数抽取"最后一个数字"，模型学会用长度去刷分。

> 这些问题**绝大多数不报错**，只是让训练悄无声息地不工作。

---

## 环境

在 **单卡 RTX 4060 Laptop（8GB）** 上完成：

| | |
|---|---|
| 模型 | Qwen3-0.6B（LoRA r=16，4-bit 无需量化即可单卡运行） |
| 优化器 | AdamW 8-bit |
| 生成 | `max_completion_length=384`，`num_generations=8`，`temperature=1.0` |
| 验证版本 | torch 2.14.1+cu130 / transformers 5.18.0 / **trl 1.14.1** / peft 0.21.2 |

---

## 已知限制

- **规模小**：0.6B 模型、300 条训练题、200 步。结论的方向可信，绝对数值不具备可比性。
- **56.9% 在 GSM8K 上不算高**（0.6B 模型本就处于此量级）。本项目的价值在于**受控对照实验的方法**，而非绝对性能。
- **离线筛选会过期**：筛选基于训练初始模型，随训练推进部分题目会从"半会"变成"全会"，需要重新筛选。DAPO 的**在线动态采样**没有这个问题。
- **未实现 DAPO 的完整方案**：`loss_type="dapo"` 只切换了 token 级损失，动态采样 / clip-higher / 超长奖励塑形均未实现。
