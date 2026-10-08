# TRL 实现对照 —— 手写版与 GRPOTrainer 逐块对应

> Day 6 配套。先读这份，再跑 `probe_trl_api.py`，最后用 `train_trl.py` 训练。
>
> ⚠️ **本文刻意不写死参数名。** TRL 的 API 在版本间改动很大，写死的名字你八成会撞
> `unexpected keyword argument`。第 5 节告诉你**去代码里查什么**，以你装的那版为准。

---

## 0. 全局：今天要干什么

你已经手写过一遍 GRPO。今天的任务**不是"学一个新东西"，而是"看工程实现"**：

| | 你的 `grpo_from_scratch.py` | TRL 的 `GRPOTrainer` |
|---|---|---|
| 代码量 | 277 行，每一行你都懂 | 算法藏在库里 |
| 数据 | 随机算术题 | **GSM8K** |
| 价值 | **理解** | **生产级**：vLLM 加速、梯度检查点、分布式、日志 |

**今天的真正目标**：能对着 TRL 的代码说出**每一部分对应你手写版的哪几行**。

做到这一点，你就能在简历上写"熟悉 TRL 的 GRPO 实现"，而不是"用过 TRL"。

---

## 1. TRL 是什么

`trl` = Transformer Reinforcement Learning，HuggingFace 官方的后训练库。

它把 RLHF/GRPO 这一整套流程封装成几种 Trainer：

| Trainer | 干什么 |
|---|---|
| `SFTTrainer` | 监督微调（RLHF 第一阶段） |
| `DPOTrainer` | 直接偏好优化（跳过 RM 和 RL） |
| **`GRPOTrainer`** | **GRPO（你要的）** |
| `PPOTrainer` | PPO |

它建立在 HF 的 `Trainer` 之上，所以**训练循环那一套（前向 / backward / step / 日志 / 存盘）是复用的**——你手写的第 ⑥ 步在这里是标准件。

---

## 2. 核心：一一映射表

**这是今天最该带走的东西。**

| 你手写的（`grpo_from_scratch.py`） | TRL 里在哪 |
|---|---|
| ① `policy.generate(..., num_return_sequences=G)` | `GRPOTrainer` 内部。生产版会用 vLLM 加速 |
| ② `reward_fn(completion, gold)` | **你传给 `reward_funcs` 的那个函数**——这是你唯一要写的东西 |
| ③ `group_advantage(rewards)` | 内部算 advantages。对应配置里的分组数（`num_generations`）和奖励缩放选项 |
| ④ `token_logprobs(model, seqs)` | 内部算 per-token log 概率，**同时算 policy / old policy / reference 三份** |
| ⑤ `ratio` + `clip` | clip 范围对应配置里的 `epsilon`（你叫 `EPS`） |
| ⑤ `kl = exp(Δ) - Δ - 1` | 对应配置里的 `beta`（你叫 `BETA`） |
| mask（只在回答上算 loss） | **内部自动处理**——你手写的那两行 mask 逻辑，库里替你做了 |
| ⑥ `zero_grad / backward / step` | HF `Trainer` 的标准训练循环 |
| `GRPOTrainer` 的 `train()` | 把上面全部串起来 |

**看出来了吗——你手写的每一个部件，TRL 里都有对应的位置。** 你不是在学新东西，是在认人。

### 一个关键差异：TRL 会显式加载几个模型

你手写版加载了 **2 个**（policy + reference）。

TRL 的 `GRPOTrainer` 一般加载：

| 模型 | 作用 |
|---|---|
| policy | 训练对象（+ LoRA） |
| **reference** | KL 惩罚 |
| **RM（可选）** | 如果你传的是学习的 RM 而不是规则函数 |

**注意**：你今天用的是规则函数，所以 RM 这一份仍然不需要——**这也说明了"奖励从哪来"和"用不用 GRPO"是两件独立的事**（你 Day 5 第 4 题问的那个点）。

---

## 3. 数据格式：GRPOTrainer 期望什么

你手写版是自己造的题。TRL 期望的是**一个带 `prompt` 字段的数据集**：

```python
{"prompt": [{"role": "user", "content": "..."}]}   # 对话格式，会走 chat template
```

或者纯文本：

```python
{"prompt": "What is 24 * 8?"}
```

**GSM8K 的题目已经是现成的**，你只需要：
1. 取出 question 字段
2. 套成上面那个结构
3. （重要）**按难度筛一遍**——用你 `probe_difficulty.py` 的思路

> GSM8K 里有大量 0.6B 必然做错的题。不筛的话，你会看到和昨天一样的现象：
> 大量零方差组，梯度为 0，reward 不动。

---

## 4. `reward_funcs` 的约定

这是**唯一必须你自己写**的部分。约定大致是：

```python
def my_reward(completions, **kwargs):
    """completions: list[str]，一个 batch 的所有回答
       kwargs: 数据集里其他字段会按列名传进来（比如 'answer'）
       返回: list[float]，长度和 completions 一致
    """
    return [1.0 if ... else 0.0 for c in completions]
```

**要注意的三点**：

1. **它是批量的**——一次处理一个 batch 的 completions，不是一条
2. **数据集的其他字段通过 `**kwargs` 传进来**——GSM8K 的标准答案就从这里拿
3. 返回**标量奖励**，不是优势——优势由 Trainer 自己算（对应你手写的第 ③ 步）

### GSM8K 的答案抽取

GSM8K 的标准答案长这样：

```
Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72
```

**标准做法**：抽 `####` 后面的数字。比你现在那个"抽最后一个数字"可靠得多——因为模型输出里可能有一串中间计算。

```python
import re
def extract_gold(answer_text):
    return answer_text.split("####")[-1].strip()

def extract_pred(completion):
    # 抽 \boxed{} 或 #### 后面的数字；都没有就退回最后一个数字
    ...
```

**这里就是"reward hacking 的第一道防线"**：抽取规则太松，模型会学会刷分；太严，正确的回答拿不到分。这个取舍你要在报告里写清楚。

---

## 5. 你需要关心的配置项（去代码里查确切名字）

跑 `probe_trl_api.py` 会把实际字段名打出来。你要找这几类：

| 概念 | 你手写版的变量 | 在配置里找什么关键词 |
|---|---|---|
| 组内样本数 | `G` | `num_generations` |
| clip 范围 | `EPS` | `epsilon` |
| KL 系数 | `BETA` | `beta` |
| 最多生成多少 | `MAX_NEW` | `completion`（长度相关） |
| 采样温度 | `temperature=1.0` | `temperature` / `top_p` |
| 生成参数组 | — | `generation`（有些版本打包成一个字典） |
| 多轮 inner update | — | `num_iterations`（对应你 Day 5 讨论的 K） |
| loss 形式 | — | `loss_type`（GRPO 有若干变体） |

**别照抄网上的博客**——版本不匹配是这类教程最常见的坑。以 `probe_trl_api.py` 的输出为准。

---

## 6. TRL 额外给你的东西（不只是"封装"）

这些是你手写版没有、生产环境必需的：

| 能力 | 为什么重要 |
|---|---|
| **vLLM 加速生成** | rollout 是 RL 里最慢的部分，vLLM 能快数倍 |
| **梯度检查点 / 4-bit** | 显存不够时的救命手段 |
| **分布式训练** | 多卡时不用自己写 |
| **自动日志** | reward / KL / 长度曲线直接出图 |
| **稳定的数值实现** | 你自己写容易在 KL、logprob 上出数值问题 |

**所以"用 TRL"不是偷懒**——是把它当基准，同时你清楚里面每一块在干什么。

---

## 7. 今天动手的顺序

1. **读这份地图**（10 分钟）
2. **跑 `probe_trl_api.py`** —— 探明你装的版本的实际 API，输出发给 Claude
3. **写 `train_trl.py`** —— 用 GRPOTrainer + GSM8K 训练
4. **对照检查** —— 对着第 2 节那张映射表，指认出库里每一块的实现位置

> **第 4 步才是重点。** 训练跑通只是副产品；能指认每一块，才是你 Day 6 的产出。
