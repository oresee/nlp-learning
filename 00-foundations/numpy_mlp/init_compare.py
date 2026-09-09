r"""
M2 练习 1 —— 四种初始化对比

【你的任务】把 init_by_mode() 里的 4 个 TODO 填完，然后运行：

    G:\Project\nlp-learning\.venv\Scripts\python.exe init_compare.py

脚本会做三件事：
  1. 【方差诊断】打印训练开始前每一层的方差 —— 这是你推的公式在真实网络里的读数
  2. 【训练】四种初始化各训一次，其余超参完全相同
  3. 【画图】四条训练曲线存到 init_compare.png

注意：诊断部分比准确率更重要。先看懂那张表，再看训练结果。
"""

import numpy as np
from mlp import forward, compute_cost, backward, update_params, predict


# ============================================================ 你要填的部分

def init_by_mode(n_x, n_h, n_y, mode, seed=1):
    """
    四种初始化。b 一律为 0（不在本次对比范围内）。

    形状约定（和 mlp.py 一致）：
        W1: (n_h, n_x)      W1 的 fan-in 是 n_x
        W2: (n_y, n_h)      W2 的 fan-in 是 n_h

    提示：rng.standard_normal(shape) 返回标准正态 N(0, 1)。
          要得到 N(0, s²)，乘以 s 即可 —— 注意乘的是**标准差**，不是方差。
    """
    rng = np.random.default_rng(seed)
    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))

    if mode == "zeros":
        # TODO 1: 全零。W1、W2 都是零矩阵，形状要对。
        W1 = np.zeros((n_h,n_x))
        W2 = np.zeros((n_y,n_h))

    elif mode == "small":
        # TODO 2: W ~ N(0, 0.01²)，两层都用同一个固定值 0.01，不随 fan-in 变。
        W1 = rng.standard_normal((n_h,n_x))*0.01
        W2 = rng.standard_normal((n_y,n_h))*0.01

    elif mode == "large":
        # TODO 3: W ~ N(0, 1²)，也就是直接用标准正态，不缩放。
        W1 = rng.standard_normal((n_h,n_x))
        W2 = rng.standard_normal((n_y,n_h))

    elif mode == "he":
        # TODO 4: He 初始化。每层用**自己的 fan-in**。
        #         你推出来的是 Var(w) = 2/n，注意这里要乘的是标准差。
        W1 = rng.standard_normal((n_h,n_x))*np.sqrt(2/n_x)
        W2 = rng.standard_normal((n_y,n_h))*np.sqrt(2/n_h)

    else:
        raise ValueError(f"未知的 mode: {mode}")

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# ============================================================ 以下不用改

def variance_report(params, X):
    """
    训练开始前，量一遍每层的二阶矩。对照你推的公式看。

    注意量的是 E[z²] 而不是 np.var(z)。
    推导里 Var(z) = E[z²] 成立的前提是 E[z] = 0，而那只在「对 W 取期望」
    的意义下成立；对某一次具体的 W 抽样，Z2 在样本方向上有非零均值，
    np.var() 会把它减掉，于是漏掉均值携带的那部分能量。
    理论用二阶矩，仪表就得量二阶矩。
    """
    _, cache = forward(X, params)
    Z1, A1, Z2, A2 = cache["Z1"], cache["A1"], cache["Z2"], cache["A2"]
    ez1, ez2 = (Z1 ** 2).mean(), (Z2 ** 2).mean()
    return {
        "E[Z1^2]":   ez1,
        "E[A1^2]":   (A1 ** 2).mean(),
        "E[Z2^2]":   ez2,
        "k":         (ez2 / ez1) if ez1 > 0 else float("nan"),
        "A2 范围":    f"[{A2.min():.4f}, {A2.max():.4f}]",
        "A1 死亡率": f"{(A1 <= 0).mean():.1%}",
    }


def train_with(params, X, Y, lr=0.5, epochs=1500):
    history = []
    for _ in range(epochs):
        A2, cache = forward(X, params)
        history.append(compute_cost(A2, Y))
        params = update_params(params, backward(params, cache, Y), lr)
    return params, history


def main():
    from sklearn.datasets import load_digits
    d = load_digits()
    X_all = d.data / 16.0
    y_all = (d.target % 2 == 1).astype(np.float64)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]
    split = int(0.8 * len(X_all))
    X, Xte = X_all[:split].T, X_all[split:].T
    Y, Yte = y_all[:split].reshape(1, -1), y_all[split:].reshape(1, -1)

    n_x, n_h, n_y = X.shape[0], 32, 1
    modes = ["zeros", "small", "large", "he"]
    labels = {"zeros": "① 全零", "small": "② N(0,0.01²)",
              "large": "③ N(0,1²)", "he": "④ He = N(0,2/n)"}
    # 画图用纯 ASCII，避免 matplotlib 默认字体缺中文字形
    plot_labels = {"zeros": "(1) zeros", "small": "(2) N(0, 0.01^2)",
                   "large": "(3) N(0, 1^2)", "he": "(4) He = N(0, 2/n)"}

    print(f"\n输入 X: {X.shape}   E[X^2] = {(X**2).mean():.4f}   "
          f"（注意 X 不是零均值，均值 = {X.mean():.4f}）")

    print("\n" + "=" * 78)
    print("  【一】训练开始前的方差诊断")
    print("=" * 78)
    print(f"  {'初始化':<16} {'E[Z1^2]':>11} {'E[A1^2]':>11} {'E[Z2^2]':>11} "
          f"{'k=比值':>9}  {'A1死亡率':>8}  A2 范围")
    print("  " + "-" * 86)
    inits = {}
    for m in modes:
        p = init_by_mode(n_x, n_h, n_y, m)
        inits[m] = p
        r = variance_report(p, X)
        print(f"  {labels[m]:<16} {r['E[Z1^2]']:>11.3e} {r['E[A1^2]']:>11.3e} "
              f"{r['E[Z2^2]']:>11.3e} {r['k']:>9.3f}  {r['A1 死亡率']:>8}  {r['A2 范围']}")
    print("k = E[Z2^2] / E[Z1^2] —— 就是推导里那个 k。等于 1 才叫「方差被保住」。")

    print("\n" + "=" * 78)
    print("  【二】训练结果（lr=0.5, epochs=1500, 其余完全相同）")
    print("=" * 78)
    histories = {}
    for m in modes:
        p, h = train_with(inits[m], X, Y)
        histories[m] = h
        tr = float(np.mean(predict(X, p) == Y))
        te = float(np.mean(predict(Xte, p) == Yte))
        flag = "  ← 卡在 ln2，模型在输出常数" if abs(h[-1] - np.log(2)) < 0.01 else ""
        print(f"  {labels[m]:<16} 最终cost {h[-1]:>9.6f}   "
              f"训练 {tr:>6.2%}   测试 {te:>6.2%}{flag}")

    print(f"\n  参考：ln(2) = {np.log(2):.6f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5))
        for m in modes:
            plt.plot(histories[m], label=plot_labels[m])
        plt.axhline(np.log(2), ls="--", c="gray", lw=1, label="ln(2)")
        plt.xlabel("epoch"); plt.ylabel("cost"); plt.yscale("log")
        plt.title("Training cost by initialization (log scale)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("init_compare.png", dpi=110)
        print("  曲线已存到 init_compare.png\n")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
