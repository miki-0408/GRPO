#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day 2 —— PyTorch 手感训练

⚠️ 先读 `PyTorch-基础.md`，再动手。不读的话你能跑通代码，
   但不知道自己在写什么、为什么要这么写、以后用在哪。

目的：只练 GRPO 真正会用到的那些东西，不做无谓的全面学习。
     跑法：python pytorch_basics.py

用法：0~4 是「演示 + 你来写」，5 是核心验收。
     每个 TODO 下面写了要做什么，写完后重跑脚本，看输出对不对。
     全部通过时，脚本最后会打印「[OK] Day 2 通过」。

一句话记住全局：PyTorch 本质只有两件事 —— 张量 + 自动微分，
其余全是封装。而下面这个循环骨架，就是 GRPO 训练循环去掉 RL 部分的最小版本：

    optimizer.zero_grad()   # 清梯度
    loss.backward()         # 反向
    optimizer.step()        # 更新
"""

import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Windows 的 cmd 默认用 GBK 编码，遇到 ⚠ ✅ 这类字符会直接抛 UnicodeEncodeError 把脚本打断。
# WSL / Linux 本来就是 UTF-8，这行是幂等的，加上只是让脚本不挑环境。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 全脚本统一的设备。有 GPU 用 GPU，没有就退回 CPU（这样在没配好的机器上也能跑通前面的练习）
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def hr(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


# ══════════════════════════════════════════════════════════════
# 0. 环境自检
# ══════════════════════════════════════════════════════════════
def check_env():
    hr("0. 环境自检")
    print("torch 版本      :", torch.__version__)
    print("CUDA 可用       :", torch.cuda.is_available())
    print("设备            :", DEV)
    if DEV == "cuda":
        print("显卡            :", torch.cuda.get_device_name(0))
        print("显存(GB)        :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
        print("支持 bf16       :", torch.cuda.is_bf16_supported())
    else:
        print("[!] 没检测到 GPU。前面的练习照样能跑，但请回去把 CUDA 版 torch 装好。")


# ══════════════════════════════════════════════════════════════
# 1. Tensor 与设备
# ══════════════════════════════════════════════════════════════
def ex1_tensor():
    hr("1. Tensor 与设备")
    # —— 演示 ——
    a = torch.randn(4, 3)
    print("CPU 上的 tensor :", a.shape, a.dtype, a.device)

    a_gpu = a.to(DEV)                      # 搬到 GPU
    print("搬之后          :", a_gpu.device)

    # 矩阵乘法： (4,3) @ (3,5) -> (4,5)
    b = torch.randn(3, 5).to(DEV)
    print("矩阵乘结果 shape:", (a_gpu @ b).shape)

    # ★ 关键点：模型和数据必须在同一个设备上，否则报 RuntimeError。
    #   这是新手最高频的错误，看到 "Expected all tensors to be on the same device" 就回来看这里。

    # —— 你来写 ——
    # TODO 1: 造一个 shape (64, 3) 的随机 tensor，搬到 DEV，
    #         和 shape (3, 8) 的 tensor 相乘，打印结果 shape。期望 (64, 8)。
    # result = ...
    # print("TODO 1 结果      :", ...)
    result = torch.randn(64,3)
    result_gpu = result.to(DEV)
    result2 = torch.randn(3,8).to(DEV)
    print("TODO 1 结果      :", (result_gpu @ result2).shape)

# ══════════════════════════════════════════════════════════════
# 2. autograd —— 自动求导
# ══════════════════════╗
def ex2_autograd():
    hr("2. autograd 自动求导")
    # —— 演示 ——
    x = torch.tensor(2.0, requires_grad=True)   # requires_grad 才会记录梯度
    y = x ** 2                                   # y = x²
    y.backward()                                 # 反向传播，求 dy/dx
    print("x =", x.item(), " y = x² =", y.item(), " dy/dx =", x.grad.item(), "(应为 2x = 4)")

    # ★ 关键点：梯度会【累加】而不是覆盖。所以训练循环里每一步开头都要 zero_grad()，
    #   否则梯度会越攒越大，训练直接崩。这是第二高频的坑。
    y2 = x ** 2
    y2.backward()
    print("再 backward 一次，x.grad =", x.grad.item(), "← 变成 8 了，因为累加了")

    # —— 你来写 ——
    # TODO 2: 求 f(x) = 3x² + 2x 在 x = 2 处的导数。
    #         手算：f'(x) = 6x + 2，x=2 时 = 14。用 autograd 验证是不是 14。
    # x2 = torch.tensor(2.0, requires_grad=True)
    # ...
    # print("TODO 2 结果      :", ...)
    x2 = torch.tensor(2.0,requires_grad=True)
    f = 3*x2**2 + 2*x2
    f.backward()
    print("TODO 2 结果      :", x2.grad.item())  # 应该打印 14

# ══════════════════════════════════════════════════════════════
# 3. nn.Module —— 定义网络
# ══════════════════════════════════════════════════════════════
class DemoMLP(nn.Module):
    """两层 MLP：3 -> 8 -> 1"""

    def __init__(self):
        super().__init__()                       # 必须调，否则 PyTorch 追踪不到你的层
        self.fc1 = nn.Linear(3, 8)               # 在 __init__ 里【声明】层
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))   # 在 forward 里【定义】计算
    
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

def ex3_module():
    hr("3. nn.Module 定义网络")
    m = DemoMLP().to(DEV)
    print("结构：\n", m)

    n_params = sum(p.numel() for p in m.parameters())
    print("可学习参数总数  :", n_params, "(3*8+8 = 32，8*1+1 = 9，合计 41)")

    out = m(torch.randn(5, 3).to(DEV))           # 调 m(x) 实际执行的是 forward(x)
    print("输入 (5,3) 输出 :", tuple(out.shape))

    # ★ 关键点：__init__ 声明层，forward 定义计算，两者不要混。
    #   写网络结构时写的 self.xxx 都在 __init__；算的时候在 forward。

    # —— 你来写 ——
    # TODO 3: 定义一个线性回归模型 LinearRegression，输入 1 维输出 1 维（就用 nn.Linear(1,1)）。
    #         实例化、搬到 DEV，打印它有多少可学习参数（期望 2：一个 weight 一个 bias）。
    # class LinearRegression(nn.Module):
    #     ...

    model = LinearRegression().to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    print("可学习参数总数  :", n_params, "(1*1+1 = 2，合计 2)")
    # 实例化、搬到 DEV，打印它有多少可学习参数（期望 2：一个 weight 一个 bias）。

# ══════════════════════════════════════════════════════════════
# 4. Dataset / DataLoader
# ══════════════════════════════════════════════════════════════
class LineDataset(Dataset):
    """造一批 y = 2x + 1 + 噪声 的数据"""

    def __init__(self, n=1000):
        self.x = torch.randn(n, 1) * 3
        self.y = 2 * self.x + 1 + torch.randn(n, 1) * 0.1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class LineDataset2(Dataset):
    """造一批 y = 3x - 2 的数据"""

    def __init__(self, n=500):
        self.x = torch.randn(n, 1) * 3
        self.y = 3 * self.x - 2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def ex4_dataloader():
    hr("4. Dataset / DataLoader")
    ds = LineDataset(1000)
    print("数据集大小      :", len(ds))
    x0, y0 = ds[0]
    print("单条样本        : x =", x0.item(), " y =", y0.item())

    dl = DataLoader(ds, batch_size=32, shuffle=True)
    bx, by = next(iter(dl))
    print("一个 batch      :", tuple(bx.shape), tuple(by.shape))

    # ★ 关键点：Dataset 负责「取第 i 条」，DataLoader 负责「分批 + 打乱 + 并行加载」。
    #   后面加载 GSM8K 就是这个模式，只不过 __getitem__ 返回的是文本而不是张量。

    # —— 你来写 ——
    # TODO 4: 造 500 条 y = 3x - 2 的数据，用 DataLoader 按 batch_size=64 取，
    #         打印第一个 batch 的 shape（期望 (64, 1) (64, 1)）。
    # 提示：可以直接复用 LineDataset 的写法改一下，或者自己写个新的 Dataset 类。
    ds2 = LineDataset2(500)
    dl2 = DataLoader(ds2,batch_size=64, shuffle=True)
    bx2,by2 = next(iter(dl2))
    print("一个 batch      :", tuple(bx2.shape), tuple(by2.shape))


# ══════════════════════════════════════════════════════════════
# 5. 训练循环 —— 核心验收，这一节没有任何演示，全靠你自己写
# ══════════════════════════════════════════════════════════════



def ex5_train_loop():
    hr("5. 训练循环 —— 核心验收")
    """
    目标：用 LinearRegression 拟合 y = 2x + 1（数据带噪声），
          训练到 loss < 0.05，并且学到的 weight ≈ 2、bias ≈ 1。

    一个训练循环的标准骨架（按顺序）：

        for epoch in range(N):
            for bx, by in dataloader:
                bx, by = bx.to(DEV), by.to(DEV)    # ① 数据搬到设备
                pred = model(bx)                    # ② 前向
                loss = loss_fn(pred, by)            # ③ 算损失
                optimizer.zero_grad()               # ④ 清空上一步的梯度 ← 别忘
                loss.backward()                     # ⑤ 反向求梯度
                optimizer.step()                    # ⑥ 更新参数

    把下面写出来。写完后重跑脚本，看能否打印「✅ Day 2 通过」。
    """
    torch.manual_seed(0)
    ds = LineDataset(1000)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = LinearRegression().to(DEV)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)   # 或 SGD(lr=0.1)
    times = 0
    for epoch in range(5):
        total_loss = 0.0
        times += 1
        for bx, by in dl:
            bx, by = bx.to(DEV), by.to(DEV)
            pred = model(bx)
            loss = loss_fn(pred,by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("epoch %d, loss %.4f" % (epoch, total_loss / len(dl)))
            

    # —— 下面是自动验收，不用改 ——
    model.eval()
    with torch.no_grad():
        x_test = torch.linspace(-5, 5, 200).view(-1, 1).to(DEV)
        y_pred = model(x_test)
        final_loss = nn.MSELoss()(y_pred, 2 * x_test + 1).item()

    w = model.fc.weight.item()
    b = model.fc.bias.item()
    print("\n学到的 weight   : %.3f  (真值 2)" % w)
    print("学到的 bias     : %.3f  (真值 1)" % b)
    print("在干净数据上 loss: %.4f" % final_loss)

    if final_loss < 0.05:
        print("\n[OK] Day 2 通过 —— 你已经会写训练循环了")
        return True
    print("\n[FAIL] 还没收敛。检查：zero_grad 调了吗？lr 是不是太大/太小？轮数够吗？")
    return False


# ══════════════════════════════════════════════════════════════
def main():
    print("\nDay 2 —— PyTorch 手感训练")
    print("设备:", DEV)
    check_env()
    ex1_tensor()
    ex2_autograd()
    ex3_module()
    ex4_dataloader()
    try:
        ok = ex5_train_loop()
    except NotImplementedError as e:
        hr("5. 训练循环 —— 尚未完成")
        print("[!]", e)
        print("\n打开本文件，找到 ex5_train_loop()，把 TODO 5 写完再重跑：")
        print("    python pytorch_basics.py")
        return
    hr("完成情况")
    print("1~4 请自行对照 TODO 注释检查输出；5 =", "通过" if ok else "未通过")


if __name__ == "__main__":
    main()
