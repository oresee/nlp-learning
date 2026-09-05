"""
用你实现的 backward 训练一个真实分类任务

⚠️ 先跑通 grad_check.py 再跑这个。梯度错了训练也可能"看起来在下降"，
   但那是假象 —— 这正是数值梯度校验存在的意义。

两个数据集：
    python train.py            # sklearn digits（8x8，内置，无需下载，秒级）
    python train.py --mnist    # 真 MNIST（28x28，首次运行会下载约 15MB）

任务是二分类（偶数 vs 奇数），因为当前 mlp.py 的输出层是单个 sigmoid。
M2 会把它扩展成 softmax 多分类，那时的目标是 MNIST 十分类 95%+。
"""

import argparse
import numpy as np

from mlp import train, predict, forward, compute_cost


def load_digits_data():
    from sklearn.datasets import load_digits
    d = load_digits()
    X = d.data / 16.0                      # (1797, 64)
    y = (d.target % 2 == 1).astype(np.float64)   # 奇数 -> 1
    return X, y, "sklearn digits (8x8)"


def load_mnist_data():
    from sklearn.datasets import fetch_openml
    print("正在获取 MNIST（首次运行需要下载，请耐心等待）...")
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = d.data[:20000] / 255.0
    y = (d.target[:20000].astype(int) % 2 == 1).astype(np.float64)
    return X, y, "MNIST (28x28, 取前 20000 条)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", action="store_true", help="用真 MNIST 而非 sklearn digits")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=1500)
    args = ap.parse_args()

    X, y, name = load_mnist_data() if args.mnist else load_digits_data()

    # 打乱 + 8:2 切分
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(0.8 * len(X))

    # 转成 (n_x, m) —— 样本在列方向
    X_train, X_test = X[:split].T, X[split:].T
    Y_train, Y_test = y[:split].reshape(1, -1), y[split:].reshape(1, -1)

    print(f"\n数据集: {name}")
    print(f"训练集: X {X_train.shape}, Y {Y_train.shape}")
    print(f"测试集: X {X_test.shape}, Y {Y_test.shape}")
    print(f"任务: 二分类（奇数 vs 偶数）")
    print(f"网络: {X_train.shape[0]} -> {args.hidden} (ReLU) -> 1 (sigmoid)\n")

    params, history = train(
        X_train, Y_train,
        n_h=args.hidden, lr=args.lr, epochs=args.epochs, verbose=True,
    )

    train_acc = float(np.mean(predict(X_train, params) == Y_train))
    test_acc = float(np.mean(predict(X_test, params) == Y_test))
    final_cost = compute_cost(forward(X_train, params)[0], Y_train)

    print(f"\n{'=' * 46}")
    print(f"  最终代价      {final_cost:.6f}")
    print(f"  训练集准确率  {train_acc:.2%}")
    print(f"  测试集准确率  {test_acc:.2%}")
    print(f"{'=' * 46}")

    if final_cost > 0.6:
        print("\n⚠️  代价几乎没降。backward 大概率还是错的 —— 回去跑 grad_check.py。")
    elif test_acc < 0.85:
        print("\n⚠️  代价降了但准确率不高。可能是学习率或隐藏层大小的问题，试着调调看。")
    else:
        print("\n✅ 训练正常。M1 的代码部分完成了。")
        print("   思考题（下次对话我会问你）：")
        print("     1. 训练集和测试集准确率的差距说明了什么？")
        print("     2. 把 --hidden 从 32 调到 256，准确率会怎么变？先预测再实验。")
        print("     3. 学习率调到 5.0 会发生什么？为什么？")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(history)
        plt.xlabel("epoch"); plt.ylabel("cost"); plt.title("Training cost")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("training_curve.png", dpi=110)
        print("\n训练曲线已保存到 training_curve.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
