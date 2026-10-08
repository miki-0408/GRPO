# GRPO 原理

> 和前两份同一套写法：**是什么 → 为什么存在 → 在代码里对应什么**。
> Day 5 配套 `grpo_from_scratch.py`。先读完再动手。

---

## 0. 全局：Day 5 在拼一张什么图

你手上已经有三块拼图了：

```
Day 2  训练循环骨架       optimizer.zero_grad() / backward() / step()
Day 3  rollout + 优势       model.generate(G) + (r_i - 均值) / 标准差
Day 5  ← 补上中间缺的两块
```

缺的是：

```
prompt → [rollout] → 回答 → [奖励] → 优势 → ??? → loss → [更新]
                                              ↑
                                    今天要补的就是这里
```

具体是两块新东西：

| 新概念 | 一句话 |
|---|---|
| **logprob（对数概率）** | 从 logits 里取出"模型给它自己选的那个 token 打了多少概率" |
| **ratio + clip** | PPO 的核心：限制一次更新能走多远 |

**补上这两块，GRPO 就完整了。** 剩下所有东西你都见过。

---

## 1. 完整循环长什么样

先看全局，再逐个拆：

```python
for step in range(STEPS):
    # ① rollout —— Day 3 学过
    seqs = policy.generate(prompts, num_return_sequences=G, do_sample=True)

    # ② 奖励 —— 规则函数，可验证
    rewards = [reward_fn(q, decode(s)) for s in seqs]

    # ③ 组内归一化优势 —— Day 3 亲手算过
    adv = (rewards - rewards.mean()) / rewards.std()

    # ④ 取 logprob —— 今天新学（下面第 2 节）
    logp_now = token_logprobs(policy,        seqs)   # 要梯度
    logp_old = token_logprobs(policy,        seqs)   # no_grad，rollout 那一刻的
    logp_ref = token_logprobs(ref_model,     seqs)   # no_grad

    # ⑤ 损失 —— 今天新学（下面第 3、4 节）
    ratio   = exp(logp_now - logp_old)
    clipped = clamp(ratio, 1-EPS, 1+EPS)
    loss = -min(ratio*adv, clipped*adv) + BETA * KL(logp_now, logp_ref)

    # ⑥ 更新 —— Day 2 学过
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**注意 ①②③⑥ 都是你已经会的。今天只学 ④⑤。**

---

## 2. logprob —— 今天第一个新概念

### 从 logits 到 logprob

昨天你已经见过 logits 了：位置 `t` 上，模型对**全词表 151936 个 token** 各打了一个分。

```
logits[t]        151936 个分数
   ↓ log_softmax
logprobs[t]      151936 个数，归一化成 log 概率
   ↓ 只取实际生成的那个 token 对应的那一个
token logprob    1 个数          ← GRPO 要的是这个
```

**为什么要取"实际那个"**：我们要问的是"模型给**它自己实际选的这个** token 打了多少概率"。别的 token 的概率无关紧要。

### 关键：同一个 token，三个模型各打一次分

这是整个 PPO/GRPO 的结构核心：

| 谁打分 | 符号 | 条件 | 含义 |
|---|---|---|---|
| 当前策略 | `logp` | 要梯度 | 更新前，策略认为这个 token 的概率 |
| 生成时的策略 | `logp_old` | no_grad | **采样那一刻**的概率 |
| 参考模型 | `logp_ref` | no_grad | 原始（SFT）模型的概率 |

三个数都对应**同一个 token**，只是打分的人不同。后面的 ratio 来自前两个，KL 来自第一个和第三个。

### 代码怎么写

```python
def token_logprobs(model, input_ids, attention_mask):
    out    = model(input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]          # 去掉最后一个位置（它没有"下一个 token"）
    target = input_ids[:, 1:]               # 右移一位：每个位置真正要预测的目标
    logp   = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)   # (batch, seq-1)
```

> `logits[:, :-1]` 和 `input_ids[:, 1:]` 这个**错位**是自回归训练的标准写法：位置 `t` 的输出，用来预测位置 `t+1` 的 token。你昨天从 logits 的形状出发已经推过这件事了。

### ⚠️ 只在回答部分算

prompt 部分的 token 不是模型生成的，不该参与训练。所以要用 mask 把 prompt 段切掉。

这是新手最容易漏的地方——**漏了不会报错，只是训练效果变差**。

---

## 3. ratio 和 clip —— 今天第二个新概念

### ratio 是什么



**理论上，rollout 完立刻更新时，两个策略是同一个模型，ratio 应该恰好等于 1。** 跑起来你会看到它非常接近 1.000 —— 这是个很好的自检点。

那为什么还要算它？因为：

- 一个 batch 里要更新**多次**（多个 epoch / mini-batch），第二次起策略就变了
- 而且用 ratio 表达更新，比直接对 logprob 求导**数值更稳定**

### 为什么必须 clip

$$\text{loss} = -\min\Big(\text{ratio}\cdot A,\ \ \text{clip}(\text{ratio}, 1-\epsilon, 1+\epsilon)\cdot A\Big)$$

直觉：**一次更新不允许把某个动作的概率改动超过 ±20%（ε=0.2）。**

为什么？因为策略梯度是**用旧策略采的样本**去估计新策略的梯度。样本只对"离旧策略不远"的新策略有效。**跑太远，估计就失真了，训练会崩。**

`min` 的作用是**取更保守的那个**——这就是 PPO 著名的"悲观裁剪"。

### 一个你熟悉的类比

这在概念上很像**信任域**：只在旧策略附近一小块区域内做更新。TRPO 用显式的约束实现，PPO 用 clip 这个更简单的技巧达到类似效果。

---

## 4. KL 惩罚

GRPO 的损失里还有一项：

$$\text{loss} = -\min(\text{ratio}\cdot A,\ \text{clip}(\text{ratio})\cdot A) + \beta \cdot D_{KL}$$

**作用**：防止模型为了刷奖励而偏离原始模型太远（reward hacking 的第一道防线）。

**放在哪**——这是 GRPO 和 PPO 的一个实际区别：

| | KL 放在哪 |
|---|---|
| PPO（InstructGPT） | 折进**每个 token 的奖励**里，再交给 GAE |
| **GRPO** | **直接作为损失里的一项**，优势完全不碰它 |

所以 GRPO 的优势**纯粹来自结果奖励**，没有任何 token 级的塑形成分。

**代码里用的形式**：

```python
delta = logp_ref - logp_now
kl = torch.exp(delta) - delta - 1      # GRPO 论文用的无偏估计（k3），恒 ≥ 0
```

不用 `logp_now - logp_ref` 也行，但那是 KL 的有偏近似。R1 论文用的是上面这个。

---

## 5. 这次要加载几个模型

对比一下（假设 0.6B，bf16）：

| | PPO | GRPO |
|---|---|---|
| Policy（可训练） | 2 + 梯度2 + Adam8 = **12 B/参数** | 同 |
| Reference（冻结） | 2 | 2 |
| Reward Model（冻结） | 2 | **不要**（奖励是规则函数） |
| **Value Model（可训练）** | **12** | **不要** |
| **每参数合计** | **28 B** | **16 B** |

**GRPO 砍掉的是 RM 和 VM。** VM 省得最多——因为它也是**可训练的**，带着梯度和优化器状态（回忆你问过的那个问题）。

**你这次的方案**：

- Policy 用 **LoRA**：只有 adapter 可训练 → 那 12 B/参数只作用在极小一部分参数上
- Reference 就是**同一个基础模型**再加载一份（冻结）——因为参考模型的定义就是"RL 之前的那个模型"

算下来 policy + reference 不到 3GB，8GB 够用。

---

## 6. 你需要学到什么程度

- [x] 知道 `token_logprobs` 为什么要错位、为什么只取实际那个 token
- [x] 知道 ratio 理论上应该 ≈ 1，以及 clip 在防什么
- [x] 知道 GRPO 的 KL 和 PPO 的 KL 放在不同位置
- [x] 能说出 GRPO 相比 PPO 砍掉了哪两个模型

**不用学**：重要性采样的数学证明、k3 估计量为什么无偏、PPO 的收敛性分析。

> 你这一天的产出不是"跑通一个脚本"，而是**能对着自己写的代码讲清 GRPO 的每一行**。
> 面试问"你实现过 GRPO 吗"，你能打开这个文件逐行讲——这比说"我调过 TRL"强得多。
