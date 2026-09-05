# M1 代码练习 · NumPy 两层 MLP

## 做什么

把 `mlp.py` 里 `backward()` 函数的 6 个 TODO 填完。前向传播、代价函数、训练循环都已经写好了。

## 步骤

```bash
cd 00-foundations/numpy_mlp
```

**1. 先在纸上推。** 不要打开编辑器就开始写。
先推 m=1 的标量情形，确认每一步的链式法则；再推广到 m 个样本的向量化形式。
推完对照 `assessments/L0-entry-diagnostic.md` 的 C2 题 —— 那题问的就是这个。

**2. 填代码。**

```bash
python mlp.py    # 会提示你去跑 grad_check
```

**3. 验证。**

```bash
python grad_check.py
```

四项相对误差都 `< 1e-7` 才算对。这个脚本会告诉你错在哪一环。

**4. 训练。**

```bash
python train.py              # 快，秒级，先用这个
python train.py --mnist      # 真 MNIST，首次会下载
```

## 卡住了怎么办

**先自己想满 15 分钟。** 这个过程本身就是训练。

再卡住，把你**纸上的推导过程**发给我（拍照或打字都行），不要只发代码。
我要看的是你哪一步的链式法则断了，看代码看不出来。

## 几个提示（按需展开，别一次全看）

<details>
<summary>提示 1 — dZ2 推不出来</summary>

分两步：先求 dL/dA2（对交叉熵求导），再求 dA2/dZ2（sigmoid 的导数）。
两个相乘之后会发生大量约分，最终结果非常简洁。
如果你得到的是一个带 A2(1-A2) 的复杂分式，说明还没约完。
</details>

<details>
<summary>提示 2 — db2 不知道怎么处理维度</summary>

关键问题：前向传播时 `Z2 = W2 @ A1 + b2`，b2 的形状是 (1,1)，Z2 的形状是 (1,m)。
b2 是被**广播**到 m 个样本上的。

广播 = 同一个变量参与了 m 条计算路径。
链式法则规定：一个变量影响多条路径时，梯度要**相加**。

所以 `np.sum(..., axis=?, keepdims=True)`，axis 选哪个？
</details>

<details>
<summary>提示 3 — dA1 的转置方向老是搞混</summary>

不要靠记忆，靠维度倒推：
- `dA1` 必须是 (n_h, m)
- 手头有 `dZ2` (1, m) 和 `W2` (1, n_h)
- 什么样的矩阵乘法能从这两个得到 (n_h, m)？只有一种可能。

这个"用维度倒推转置"的技巧，后面写 attention 时你会天天用。
</details>

<details>
<summary>提示 4 — dZ1 穿过 ReLU</summary>

ReLU 的导数：z > 0 时为 1，z < 0 时为 0。
所以从 dA1 到 dZ1，是逐元素地"把 Z1 中非正位置对应的梯度清零"。

用 NumPy 怎么写？`(Z1 > 0)` 会得到一个布尔数组，它可以直接参与乘法。
</details>

## 完成后

在对话里说「M1 代码做完了」，我会：
- 检查你的实现（不只看对错，还看写法是否自然）
- 问你 `train.py` 输出里的三道思考题
- 评掌握度并写进 `PROGRESS.md`
