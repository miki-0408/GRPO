# Transformers 概念地图

> 和 `PyTorch-基础.md` 同一套写法：**是什么 → 为什么存在 → 和什么有关 → 在 GRPO 里用在哪**。
> Day 3 的练习配套这份文档。先读完，再动代码。

---

## 0. 先建立全局：Day 3 到底在学什么

表面上你在学"怎么加载一个模型然后让它说话"。实际上——

**你在学 GRPO 训练循环的第 ① 步：rollout（生成回答）。**

回顾 Day 2 第 4 题的答案：

```python
for prompts in dl:
    responses = model.generate(prompts, num_return_sequences=G)  # ① ← 就是今天
    rewards   = reward_fn(prompts, responses)
    advantages = group_normalize(rewards)
    loss = policy_gradient_loss(responses, advantages)            # ②
    optimizer.zero_grad(); loss.backward(); optimizer.step()      # ③④⑤
```

我们今天学的 `model.generate()`，就是 ①。

**所以这不是"玩一下生成文本"，是在搭 GRPO 的前半截。**

---

## 1. 一条因果链

```
文本 "1+1=?"
   │  ①分词
   ↓
token id [16, 10, 30, 13]          ← 模型只认数字，不认字
   │  ②过模型
   ↓
logits（每个位置对全词表的打分）
   │  ③采样
   ↓
下一个 token id
   │  ④接回去，重复
   └──────→ 回到 ②（这就是"自回归"）
   ↓
generate() 把 ②③④ 循环 n 次，返回完整回答
```

下面逐个拆。

---

## 2. Tokenizer —— 文本怎么变成数字

### 是什么
把文本切成 **token**，再把每个 token 映射成一个整数 id。反向也能把 id 拼回文本。

### 为什么需要它
**神经网络只能做数值计算，不能处理字符串。** 所以必须先有一个确定的、可逆的编码方案。

### 为什么不是"一个字一个 id"
那样词表会爆炸（中文几万字、多语言几十万），而且模型学不到构词规律。所以主流用 **subword（子词）分词**：常见词整体成一个 token，罕见词拆成几个片段。

好处：词表可控（一般 10 万上下），且任何文本都能编码，不会遇到"没见过的字"。

### 你会直接接触到的
```python
tok("你好，世界")
# {'input_ids': [108386, 3837, 271, ...], 'attention_mask': [1, 1, 1, ...]}
```

### 在 GRPO 里用在哪
GSM8K 的题目文本 → `input_ids` → 喂给模型生成。**这是整条数据管线的入口。**

---

## 3. Chat Template —— instruct 模型的隐藏开关

### 是什么
把 `[{"role":"user","content":"1+1=?"}]` 这种对话结构，套上一段**模型训练时见过的固定格式**，再送去 tokenize。

### 为什么需要它
后训练过的模型（instruct / chat 版本）是在**特定对话格式**上训练的。你直接喂裸文本，等于用错了输入格式——模型不知道"这是用户说的话"还是"这是我要续写的内容"，输出会明显变差。

这属于**不报错、但效果悄悄变差**的坑，和你 Day 2 第 3 题里"忘传 parameters()"是同一类。

### 你会直接接触到的
```python
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "1+1=?"}],
    tokenize=False,
    add_generation_prompt=True,
)
```

### ⚠️ Qwen3 特有的坑
Qwen3 有**思考模式（thinking mode）**，默认可能先输出一段 ` thinking...<｜end▁of▁thinking｜>` 再给答案。

这对你在 GSM8K 上做实验**影响很大**——你测的到底是"答案对不对"还是"思考过程对不对"？

**动手前先去 HuggingFace 的 model card 确认两件事**：你加载的是 Base 还是 Instruct 版本、`apply_chat_template` 的 thinking 相关参数怎么设。我不替你断言，以 model card 为准。

### 在 GRPO 里用在哪
GRPO 的 prompt 必须按这个格式构造。**格式错了，模型输出质量下降，奖励分数低，你就误以为是训练算法有问题**——其实只是输入格式错了。

---

## 4. AutoModelForCausalLM —— 自回归语言模型

### 是什么
**Causal LM（因果语言模型）**：给定前面的 token，预测下一个 token 的概率分布。GPT 系列、Qwen 系列都是这一类。

"Auto" 是说你不用记住具体类名（`Qwen3ForCausalLM`），给个模型名它会自动选对。

### 为什么叫"因果"
因为它只能**看左边**，不能看右边——生成第 5 个 token 时，第 6 个还不存在。这个约束保证了训练和推理的一致性。

### 输出的 logits 是什么
形状 `(batch, seq_len, vocab_size)`。含义是：**每个位置上，对词表里 10 万个 token 各打了一个分**。

例如输入 `"1+1="`，最后一个位置的 logits 就是"下一个 token 是 `2` 的概率最高"。

### 在 GRPO 里用在哪
- **这个是你要训练的模型**（policy）
- GRPO 会同时持有它的**副本**作为参考模型（reference model），用来算 KL 惩罚——保证训练后模型不会偏离原始模型太远
- 你 8GB 显存要同时装这两个 + LoRA，所以必须量化，这就是计划里说的显存压力来源

---

## 5. 采样 —— 怎么从 logits 里选出一个 token

### 是什么
logits 是分数，要变成"选哪个"需要一个策略。两种极端：

- **贪心（greedy）**：永远选分数最高的。确定性的，同一个输入永远同一个输出。
- **采样（sampling）**：按概率分布随机抽。同一个输入每次结果不同。

### 为什么需要采样
贪心会生成**重复、呆板**的文本，而且——**对 GRPO 是致命的**：

> GRPO 的核心是**组内归一化**：同一个 prompt 生成 G 条回答，用这一组的平均奖励当作基线，算出每条的优势。
>
> **如果 G 条回答全都一样**（贪心必然如此），组内奖励全相同，归一化后**优势全为 0，梯度就是 0，训练什么也没发生。**

这就是我在学习计划里列的坑之一。**没有采样，就没有 GRPO。**

### 关键参数
| 参数 | 含义 | 直觉 |
|---|---|---|
| `temperature` | 分布的"平坦程度" | 越低越确定，越高越随机 |
| `top_k` | 只在分数最高的 k 个里抽 | 砍掉长尾乱码 |
| `top_p` | 只在累计概率达到 p 的最小集合里抽 | 比 top_k 更自适应 |
| `num_return_sequences` | **一次生成几条** | **这就是 GRPO 的 G** |
| `max_new_tokens` | 最多生成多长 | 直接决定显存和时间 |
| `do_sample` | 是否启用采样 | `False` 就是贪心 |

### 一个对你有用的映射
`temperature` 控制的是**探索与利用的权衡**——这和你强化学习里的 ε-greedy、熵正则是一回事。

温度太低 → 不探索 → 组内无差异 → 学不到东西。
温度太高 → 输出乱 → 全错 → 组内也无差异 → 同样学不到。

**这个平衡你在开悟比赛里调奖励时体会过。**

---

## 6. `generate()` —— 自回归的循环

### 是什么
一个循环：预测下一个 token → 接回去 → 再预测 → 直到结束或达到长度上限。

### 为什么不能一次算出全部
因为**下一个 token 依赖上一个的输出**。生成第 5 个之前，必须先把第 4 个定下来。这就是"自回归"的含义。

（这也是为什么推理比训练慢、为什么要有 KV Cache 这种优化——不过那是后面的内容，现在不用管。）

### 在 GRPO 里用在哪
**这就是 rollout，GRPO 训练循环的 ① 步。**

而且这也是 GRPO 慢的原因：一次训练迭代要生成 `G × batch` 条完整回答，每一步都要跑一遍完整前向。**生成是 RL 训练里最耗时的部分**，你之后会深有体会。

---

## 7. `attention_mask` 与 padding —— 批处理的两个配角

### 是什么
- **padding**：一个 batch 里的句子长度不一，短的补到一样长才能拼成张量
- **attention_mask**：告诉模型"哪些位置是真内容，哪些是补出来的，别当真"

### 为什么需要
GPU 要批量处理，张量必须规整；但真实文本天然不等长。所以补位 + 打标记。

### 一句话记住
`attention_mask` 里 **1 = 真内容，0 = 补的**。

（等你看完 Transformer 的注意力机制会发现"mask"这个词到处都是，但含义不同——那种是防止看到未来。这里只是标记有效位置。）

---

## 8. dtype 与 device —— 把模型放进显卡

```python
model = AutoModelForCausalLM.from_pretrained(
    name,
    torch_dtype=torch.bfloat16,   # 半精度，省一半显存
).to("cuda")
```

- 你昨天验证过 `bf16` 是 `True`，所以可以用
- 8GB 卡上，之后还要叠加 4-bit 量化（`bitsandbytes`），那是 Day 6 的事

---

## 9. 这一整天的东西，在 GRPO 里的对应表

| Day 3 概念 | GRPO 里的角色 |
|---|---|
| Tokenizer | 把 GSM8K 题目变成 `input_ids` |
| Chat template | 构造符合模型训练格式的 prompt |
| AutoModelForCausalLM | **policy 模型**（要训练的那个）+ reference 模型 |
| logits | 生成时逐步转成 token 分数 |
| **采样参数** | **决定 G 条回答的多样性——没有多样性就没有 GRPO 信号** |
| `num_return_sequences` | **GRPO 的 G** |
| **`generate()`** | **rollout，训练循环的 ① 步** |
| attention_mask | 批处理必须 |

---

## 10. 你需要学到什么程度

和 PyTorch 那天一样，**不需要精通**：

- [x] 知道 tokenizer 在做什么、`input_ids` 和 `attention_mask` 是什么
- [x] 知道 chat template 是干什么的、不做会怎样
- [x] 知道 logits 的形状和含义
- [x] 能用 `generate()` 写出可控的生成（会调 temperature / top_p / num_return_sequences）
- [x] **理解"采样多样性 → GRPO 组内差异 → 优势信号"这条因果链**

**不用学**：注意力机制的数学细节、位置编码、KV Cache 实现、模型架构变体。那些等你读 TRL 源码遇到再说。

> 你的目标始终是同一个：**能读懂并改动 TRL 的 GRPO 训练代码**。
> Day 2 给了你训练循环的骨架，Day 3 给你 rollout。**合起来就是 GRPO 的全部。**
