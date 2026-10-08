# GRPOTrainer 源码地图

> 文件：`~/rl-llm/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py`（**3465 行**）
> 版本：trl 1.14.1
>
> 这份地图是**从你实际安装的那份文件里 grep 出来的行号**，不是通用文档。
> 今天的产出：能指着下面每一行说"这块对应我手写版的哪几行"。

---

## 一、总览：文件结构

| 行号 | 内容 |
|---|---|
| 304 | `__init__`（900 行，大部分是参数处理，**可以跳过**） |
| 1370 | `_get_per_token_logps_and_entropies` ← **④ 取 logprob** |
| 1607 | `_prepare_inputs` |
| 1664 | `_calculate_rewards` ← **② 调用你传的 reward_funcs** |
| 1848 | `_generate_single_turn` |
| 2251 | `_generate` ← **① rollout** |
| 2377 | `_generate_and_score_completions` ← **主战场：生成+奖励+优势** |
| 2995 | `compute_loss`（入口，几行） |
| 3081 | `_compute_loss` ← **⑤ ratio / clip / KL** |

**别从头读。** 直接跳到 2377 和 3081 这两个函数——**它们加起来就是你那 277 行的全部**。

---

## 二、逐块对应

### ① rollout —— 你手写的 `policy.generate(...)`

```
2251:  def _generate(self, prompts)                    # 批量生成入口
1848:  def _generate_single_turn(...)                   # 单轮生成（多轮工具调用用）
2377:  def _generate_and_score_completions(...)         # ← 从这里开始读
```

你手写的一行 `policy.generate(..., num_return_sequences=G)`，在这里展开成了一个 600 行的函数——**多出来的全是你没有的东西**：多模态、工具调用、vLLM 后端、分布式分片。核心就一行 `generate`。

### ② 奖励 —— 你传进去的 `reward_fn`

```
1664:  def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
1711:      output_reward_func = reward_func(...)          # ← 你的函数在这里被调用
```

**这是你唯一写的部分。** 打开 1711 附近能看到 TRL 怎么把 `prompts` / `completions` / 数据集其他列打包成 kwargs 传给你的函数——就是你运行时看到的那条 `[诊断]` 打印。

### ③ 优势 —— 你手写的 `group_advantage()`

```
2841:  advantages = rewards - mean_grouped_rewards
2843:      advantages = advantages / (std_rewards + 1e-4)
   ⋮
2855:  advantages = (rewards - torch.nanmean(rewards)) / (std_rewards + 1e-4)
2866:  advantages = torch.nan_to_num(advantages, nan=0.0)
```

**和你写的 `(r - 均值) / 标准差` 完全一致。**

注意 `+ 1e-4`：分母加一个小量避免除零。**这正是你 Day 5 写的那个"std < 1e-8 时返回全 0"的工程写法**——TRL 用加 eps 达到同样效果。

（2841 和 2855 是两个分支，对应 `scale_rewards` 取 `'group'` 还是别的值。）

### ④ per-token logprob —— 你手写的 `token_logprobs()`

```
1370:  def _get_per_token_logps_and_entropies(self, model, *args, batch_size=None, ...)
1388:  def _chunked_logps(...)         # 分块实现，省显存
1481:  def _full_logits_logps(...)     # 一次性算完
```

**注意它的签名第一个参数是 `model`** —— 因为它要被复用来算**三份** logprob：policy 的、old policy 的、reference 的。你的函数只算一份。

找一下里面 `logits_to_keep` 和那个错位——**和你推出来的右移一位是同一件事**。

### ⑤ ratio / clip / KL —— 你手写的 `grpo_loss()`

```
3179:  coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
3180:  per_token_loss1 = coef_1 * advantages
3181:  per_token_loss2 = coef_2 * advantages
3182:  per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
```

**第 3182 行就是你写的 `-torch.min(ratio * adv, clipped * adv)`**，一字不差。

（`coef_1` 就是 ratio，在上面几行算出来的。）

### KL —— 你手写的 `kl = exp(Δ) - Δ - 1`

```
3158:  if self.beta != 0.0:
3159:      ref_per_token_logps = inputs["ref_per_token_logps"]
3160:      per_token_kl = (
3162:          torch.exp(ref_per_token_logps - per_token_logps)
3163:          - (ref_per_token_logps - per_token_logps) - 1
3164:      )
```

**这就是你写的那两行**：

```python
delta = logp_ref - logp
kl = torch.exp(delta) - delta - 1
```

**完全一致。** 你 Day 5 的实现是对的。

而且注意它在**损失里**（3209-3210）：

```
3209:  if self.beta != 0.0:
3210:      per_token_loss = per_token_loss + self.beta * per_token_kl
```

**直接加到 per-token loss 上**，不是在奖励里。**这就是你 Day 5 第 3 题答的那条区别**——GRPO 的 KL 在损失里，PPO 的在奖励里。

### mask —— 你手写的那两行

```
1435:  completion_mask = attention_mask[:, -logits_to_keep:].bool()
3215:  loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
```

你的两行是 `mask[:, prompt_len-1:] = 1` 和 `(per_token_loss * mask).sum() / mask.sum()`。**思路一样，但归一化方式有差别 —— 见第四节。**

---

## 三、参考模型：你猜对了

```
966:  self.beta = args.beta
969:  if self.beta == 0.0:
971:      self.ref_model = None        # beta=0 就不需要参考模型
972:  elif is_peft_model(model):
975:      self.ref_model = None        # ← 你用 LoRA，走这条
985:  else:
        self.ref_model = create_model_from_path(...)   # 否则真的再加载一份
```

**你用 `peft_config` → `is_peft_model(model)` 为真 → `ref_model = None` → 不额外加载权重。**

实际怎么算参考 logprob（2779-2784）：

```python
model = self.accelerator.unwrap_model(self.model)
with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):
    ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(self.model, ...)
```

**`adapter_name=None` = 关掉所有 adapter = 回到基础模型。** 所以你的 8GB 只装一份 0.6B。

> 你脚本里那句注释是对的。

---

## 四、两个和你手写版不一样的地方

### ① loss 的归一化方式

| | 写法 | 含义 |
|---|---|---|
| **你** | `(per_token * mask).sum() / mask.sum()` | **全局**按 token 平均——**长回答权重大** |
| **TRL**（3215） | `((per_token * mask).sum(-1) / mask.sum(-1)).mean()` | **先按序列各自平均，再对序列取平均**——**每条回答等权** |

差别很实际：一条 300 token 的回答和一条 20 token 的回答，在你的实现里前者影响大 15 倍，在 TRL 里两者一样重。

**这正是 DAPO 的"token 级 loss"在争论的问题**——Day 8 你会看到他们为什么推荐"每条等权"。你现在已经在源码里碰到它了。

### ② `old_per_token_logps` 的条件

```
2695:  generate_every = self.args.steps_per_generation * self.num_iterations
2696:  if self.args.gradient_accumulation_steps % generate_every != 0 or (self.use_vllm and ...):
2698:      old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(...)
2709:  else:
2710:      old_per_token_logps = None
```

**注意这个 `else: None`。**

TRL 的注释说得很明白：**如果生成和优化是"对齐"的（每次生成后恰好走完一个完整优化步），那样本一定来自当前策略，重要性采样就不需要，于是直接不算 old logps。**

你的配置 `gradient_accumulation_steps=8`、`num_iterations=1` 就属于"对齐"的情况 → `old_per_token_logps = None` → **ratio 恒等于 1**。

**这从源码层面证实了你 Day 5 的困惑**：你观察到 ratio ≈ 1.000，我当时解释"因为 K=1"——现在你看到 TRL 里有个**显式的分支**在判断这件事，连注释都写着"importance sampling isn't necessary"。

> 你手写版是"算了但恒等于1"；TRL 是"判断出不需要，直接不算"。**省了一次前向。**

---

## 五、今天该交的产出

把上面这张表**用你自己的话重写一遍**（不用抄，抄没意义）。标准是：合上这份文档，你能说出——

1. 你那 277 行里的每一块，跑到了 3465 行的哪个函数
2. `ref_model = None` 那条分支为什么走你这条
3. `old_per_token_logps = None` 的条件是什么，为什么你的配置命中它
4. loss 归一化你和 TRL 的差别，以及这为什么是 DAPO 关心的点

**第 4 条是你自己读源码才能发现的东西**——它不在任何教程里，而且直通 Day 8。
