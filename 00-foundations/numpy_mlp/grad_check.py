"""
数值梯度校验 —— 验证你在 mlp.py 里写的 backward 是否正确

原理：
    梯度的定义是极限。用中心差分近似：

        dJ/dθ ≈ [ J(θ + ε) - J(θ - ε) ] / (2ε)

    用一个很小的 ε（1e-7）逐个扰动每个参数，得到"数值梯度"。
    再和你 backward 算出的"解析梯度"比较相对误差：

        relative_error = ||grad_num - grad_ana|| / (||grad_num|| + ||grad_ana||)

    < 1e-7  → 正确
    ~ 1e-5  → 可能有小问题（比如漏了 1/m，或某处转置写反）
    > 1e-3  → 明确写错了

【这是深度学习工程里最重要的调试技巧之一】
自己实现任何新层时都该这样验证。等到阶段 2 手写 attention，你还会再用到它。

用法：
    python grad_check.py
"""

import numpy as np
from mlp import init_params, forward, compute_cost, backward


def numerical_gradients(params, X, Y, eps=1e-7):
    """对每个参数元素做中心差分，返回和 params 同结构的数值梯度。"""
    num_grads = {}
    for name in params:
        P = params[name]
        g = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"], op_flags=["readonly"])
        while not it.finished:
            idx = it.multi_index
            orig = P[idx]

            P[idx] = orig + eps
            cost_plus = compute_cost(forward(X, params)[0], Y)

            P[idx] = orig - eps
            cost_minus = compute_cost(forward(X, params)[0], Y)

            P[idx] = orig  # 恢复，很重要
            g[idx] = (cost_plus - cost_minus) / (2 * eps)
            it.iternext()
        num_grads["d" + name] = g
    return num_grads


def relative_error(a, b):
    num = np.linalg.norm(a - b)
    den = np.linalg.norm(a) + np.linalg.norm(b)
    return num / den if den > 0 else 0.0


def main():
    rng = np.random.default_rng(0)
    n_x, n_h, m = 5, 4, 7          # 故意用小规模，数值梯度很慢
    X = rng.standard_normal((n_x, m))
    Y = rng.integers(0, 2, (1, m)).astype(np.float64)

    params = init_params(n_x, n_h, 1, seed=3)

    _, cache = forward(X, params)
    ana = backward(params, cache, Y)
    num = numerical_gradients(params, X, Y)

    print("=" * 58)
    print("  数值梯度校验")
    print("=" * 58)

    all_ok = True
    for k in ["dW1", "db1", "dW2", "db2"]:
        err = relative_error(num[k], ana[k])
        if err < 1e-7:
            status, mark = "通过", "OK "
        elif err < 1e-5:
            status, mark = "可疑 —— 检查是否漏了 1/m 或转置写反", "?? "
            all_ok = False
        else:
            status, mark = "错误", "XX "
            all_ok = False
        print(f"  [{mark}] {k:5s}  相对误差 = {err:.3e}   {status}")

    print("=" * 58)
    if all_ok:
        print("  全部通过。你的反向传播是对的。")
        print("  下一步：python train.py")
    else:
        print("  有未通过项。排查顺序（从后往前，因为误差会向前传播）：")
        print("    1. dW2 / db2 错  → 问题出在 dZ2，先检查它")
        print("    2. dW2 / db2 对，dW1 / db1 错")
        print("                     → 问题出在 dA1（转置方向）或 dZ1（ReLU 导数）")
        print("    3. 误差在 1e-5 量级 → 大概率是 1/m 的位置放错了")
        print("    4. 只有某一项误差在 1e-4~1e-2 且反复检查推导无误 → 可能撞上了 ReLU 在 0")
        print("       处的不可导点（概率很低）。换个 seed 重跑确认。")
        print()
        print("  卡住超过 15 分钟，把你的推导过程发给我，我帮你看哪一步断了。")
    print()


if __name__ == "__main__":
    main()
