"""
两层 MLP —— 反向传播练习

结构：输入 -> 隐藏层(ReLU) -> 输出层(sigmoid) -> 二分类交叉熵

约定（与吴恩达课程一致，样本在列方向）：
    X: (n_x, m)   n_x = 特征数, m = 样本数
    Y: (1, m)     标签 0/1

【你的任务】把 backward() 里的 TODO 填完，然后运行：

    python grad_check.py

数值梯度校验通过（相对误差 < 1e-7）才算做对。

提示：
  - 不要从网上抄。卡住就先在纸上推 m=1 的标量情形，再推广到向量化。
  - 每写一行就检查维度：dW1 必须和 W1 同形状，db1 必须和 b1 同形状。
    维度对不上，说明推导错了 —— 这是最好用的自检手段。
"""

import numpy as np


# ---------------------------------------------------------------- 激活函数

def sigmoid(z):
    # 数值稳定版本：避免 z 很负时 exp(-z) 溢出
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def relu(z):
    return np.maximum(0.0, z)


# ---------------------------------------------------------------- 参数初始化

def init_params(n_x, n_h, n_y=1, seed=1):
    """He 初始化（配 ReLU）。为什么是 sqrt(2/n) 而不是 sqrt(1/n)？—— M2 会讲。"""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.standard_normal((n_h, n_x)) * np.sqrt(2.0 / n_x),
        "b1": np.zeros((n_h, 1)),
        "W2": rng.standard_normal((n_y, n_h)) * np.sqrt(2.0 / n_h),
        "b2": np.zeros((n_y, 1)),
    }


# ---------------------------------------------------------------- 前向传播

def forward(X, params):
    """
    返回 A2 和 cache。

    Z1 = W1 @ X  + b1     (n_h, m)
    A1 = relu(Z1)         (n_h, m)
    Z2 = W2 @ A1 + b2     (n_y, m)
    A2 = sigmoid(Z2)      (n_y, m)
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    Z1 = W1 @ X + b1
    A1 = relu(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# ---------------------------------------------------------------- 代价函数

def compute_cost(A2, Y):
    """二分类交叉熵，对 m 个样本取平均。eps 防止 log(0)。"""
    m = Y.shape[1]
    eps = 1e-12
    A2 = np.clip(A2, eps, 1.0 - eps)
    return float(-np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m)


# ---------------------------------------------------------------- 反向传播【你来写】

def backward(params, cache, Y):
    """
    计算 dW1, db1, dW2, db2（都是对平均代价 J 的偏导）。

    ================== 你的任务从这里开始 ==================

    按这个顺序推，每一步都是上一步的结果乘以局部梯度：

      1) dZ2  —— sigmoid + 交叉熵合并求导后会得到一个非常简洁的形式。
                 如果你推出来的是一坨复杂表达式，说明还能化简。
                 形状 (n_y, m)

      2) dW2  —— dZ2 和谁做矩阵乘法？谁需要转置？别忘了 1/m。
                 形状必须等于 W2.shape

      3) db2  —— b2 是 (n_y, 1)，但 dZ2 是 (n_y, m)。
                 想清楚：前向时 b2 被广播到了 m 个样本上，
                 那反向时这 m 条路径的梯度该怎么合并？
                 形状必须等于 b2.shape

      4) dA1  —— 从 dZ2 往前传一层。注意方向：这次谁转置？

      5) dZ1  —— 穿过 ReLU。ReLU 的导数是什么？怎么用 Z1 表达？

      6) dW1, db1 —— 和第 2、3 步同理，只是把 A1 换成 X。

    自检：每算完一个就 assert 形状。维度对不上 = 推导错了。
    ======================================================
    """
    m = Y.shape[1]
    W2 = params["W2"]
    X, Z1, A1, A2 = cache["X"], cache["Z1"], cache["A1"], cache["A2"]

    # TODO 1: dZ2 = ?
    dZ2 = 1/m*(A2-Y)

    # TODO 2: dW2 = ?
    dW2 = dZ2 @ A1.T

    # TODO 3: db2 = ?
    db2 = np.sum(dZ2,axis=1,keepdims=True)

    # TODO 4: dA1 = ?
    dA1 = W2.T @ dZ2 

    # TODO 5: dZ1 = ?
    dZ1 = dA1 * (Z1>0)

    # TODO 6: dW1, db1 = ?
    dW1 = dZ1 @ X.T
    db1 = np.sum(dZ1,axis=1,keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    # 形状自检 —— 不要删掉，这是最有用的一道防线
    for k, g in grads.items():
        p = params[k[1:]]
        assert g.shape == p.shape, f"{k} 形状 {g.shape} != {k[1:]} 形状 {p.shape}"

    return grads


# ---------------------------------------------------------------- 参数更新

def update_params(params, grads, lr):
    for k in params:
        params[k] = params[k] - lr * grads["d" + k]
    return params


# ---------------------------------------------------------------- 训练循环

def train(X, Y, n_h=16, lr=0.1, epochs=2000, seed=1, verbose=True):
    n_x, _ = X.shape
    params = init_params(n_x, n_h, 1, seed)
    history = []
    for i in range(epochs):
        A2, cache = forward(X, params)
        cost = compute_cost(A2, Y)
        grads = backward(params, cache, Y)
        params = update_params(params, grads, lr)
        history.append(cost)
        if verbose and i % max(1, epochs // 10) == 0:
            print(f"  epoch {i:5d}  cost = {cost:.6f}")
    return params, history


def predict(X, params, threshold=0.5):
    A2, _ = forward(X, params)
    return (A2 > threshold).astype(int)


if __name__ == "__main__":
    print("直接运行 grad_check.py 来验证你的 backward 实现：")
    print("    python grad_check.py")
