# 环境搭建与跨设备同步

## ⚠️ 先读这一条：5070 Ti 的 CUDA 版本要求

台式机的 **RTX 5070 Ti 是 Blackwell 架构（sm_120）**，老版本 PyTorch 不认识这张卡。
装了 cu121 / cu124 等旧构建会报：

```
CUDA error: no kernel image is available for execution on the device
```

这个错误有迷惑性 —— `torch.cuda.is_available()` 可能返回 True，但一跑实际计算就崩。

**必须装 CUDA 12.8 或更新版本的 PyTorch 构建。**
到 https://pytorch.org/get-started/locally/ 手动选 CUDA 12.8+，不要用页面默认选中的那个。

笔记本的 RTX 3060（Ampere, sm_86）没有这个限制，但两台机器装同一个版本最省心。

---

## 一、Python 环境（两台机器都要做）

**版本选择：本项目统一 3.13（2026-09-21 起，两台机器都已切到 3.13.15）。**
阶段 1、2 任何版本都行。定 3.13 的理由是阶段 3 的 `bitsandbytes` / `flash-attn` 需要编译，
预编译 wheel 对**最新**版本（3.14）滞后明显，而 3.13 的覆盖已经够用。
**→ 原计划的「阶段 3 前另建 3.12 环境」作废，不需要了。**

检测记录（最新在上）：

| 机器 | Python | 位置 | 备注 |
|---|---|---|---|
| 笔记本 | **3.13.15** | `D:\Program\Python\Python313` | 2026-09-20 装好，venv 已建 |
| 台式机 | **3.13.15** | `G:\Program\Python\Python313` | 2026-09-21 实测；`.venv` 已在其上重建 |
| 台式机（旧） | 3.14.3 | `D:\PYTHON` | **仍是系统默认**（`python` / `py` 都指向它），但项目不再用 |

两边 venv 的包一致：numpy 2.5.3 / scipy 1.18.1 / matplotlib 3.11.1 / scikit-learn 1.9.0。
**torch 两台都未装** —— M1、M2 不需要，到 M3 再装。
届时注意：驱动 610.62（CUDA UMD 13.3），**cu128 与 cu130 构建都满足 5070 Ti 的 sm_120 要求**。

**台式机实测硬件（2026-09-07，硬件未变）：** Ryzen 7 9700X 8C/16T ｜ 内存 **16GB（15.2GB 可用）** ｜ RTX 5070 Ti 16303MiB ｜ Win11 26200
> ⚠️ 内存只有 16GB，和显存一样大。阶段 3 加载 7B 模型时不能用默认加载路径，必须 `low_cpu_mem_usage=True` + `device_map`，否则内存 OOM。

### 1. 装 Miniconda
到 https://docs.conda.io/en/latest/miniconda.html 下载 Windows 64-bit 安装包。
安装时**勾选**「Add Miniconda3 to my PATH environment variable」（虽然官方不推荐，但对你后面用 Git Bash 方便很多）。

### 2. 创建虚拟环境（二选一）

**A. 已经装了 Python（台式机的情况）—— 直接用 venv，不必再装 conda：**

```bash
python -m venv .venv
.venv\Scripts\activate        # PowerShell / CMD
source .venv/Scripts/activate  # Git Bash
```

若系统 Python 是 3.13 而你想用 3.12，用 py launcher 指定：`py -3.12 -m venv .venv`

**B. 没装 Python（笔记本的情况）—— 用 conda：**

```bash
conda create -n nlp python=3.12 -y
conda activate nlp
```

`.venv/` 已在 `.gitignore` 里，不会被同步 —— 两台机器各自建各自的。

### 3. 装 PyTorch（注意两台机器的 CUDA 版本可能不同）

先看显卡驱动支持的 CUDA 版本：

```bash
nvidia-smi
```

然后到 https://pytorch.org/get-started/locally/ 选对应版本。装完务必验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

必须打印出 `True` 和你的显卡型号。**打印 False 就是装错了版本，别将就着往下走。**

**台式机还要额外做一步实际计算验证**（`is_available()` 返回 True 不代表能跑）：

```bash
python -c "import torch; x=torch.randn(1000,1000,device='cuda'); print((x).sum().item()); print(torch.cuda.get_device_capability())"
```

能打印出数值且 capability 显示 `(12, 0)` 才算真正装对。报 `no kernel image` 就是 CUDA 版本太旧，重装 cu128+。

### 4. 装其余依赖

```bash
pip install -r requirements.txt
```

---

## 二、两台机器的分工

| 任务 | 笔记本 RTX 3060 (6GB) | 台式机 9700X + 5070 Ti (16GB) |
|---|---|---|
| 阶段 1 全部内容 | ✅ 完全够用 | ✅ |
| 阶段 2 小规模验证（~10M 参数） | ✅ | ✅ |
| 阶段 2 主训练（~50M 参数 GPT） | ⚠️ 勉强，需减小 batch | ✅ **在这里做** |
| 阶段 3 QLoRA 微调 7B | ❌ 显存不够 | ✅ **在这里做** |
| 阶段 3 全参数微调 1B 以下小模型 | ❌ | ✅ |
| 读讲义、写代码、做题、跑 NumPy 练习 | ✅ **主要用途** | ✅ |

**结论：** 笔记本负责学习、写代码、小规模验证；台式机负责真正的训练任务。代码在两边都能跑，靠 Git 同步。

---

## 三、跨设备同步（关键）

### 为什么必须做

Claude Code 的**对话历史存在本机**（`C:\Users\<用户名>\.claude\projects\`），换设备读不到。
但只要这个项目文件夹同步了，`CLAUDE.md` 和 `PROGRESS.md` 会在新会话里被自动读取，辅导可以无缝接续。

**规则：所有重要状态必须写进仓库文件，不能只存在对话里。**

### 步骤

**第一次（在当前这台机器）：**

```bash
git init
git add .
git commit -m "init: NLP 学习系统骨架"
```

然后在 GitHub 建一个**私有仓库**（名字随意，比如 `nlp-learning`），把它作为远端：

```bash
git remote add origin https://github.com/<你的用户名>/nlp-learning.git
git branch -M main
git push -u origin main
```

**在另一台机器：**

```bash
git clone https://github.com/<你的用户名>/nlp-learning.git
```

**日常使用：**

```bash
git pull      # 开始学习前
# ...学习、做题、写代码...
git add -A && git commit -m "M1: 完成反向传播推导练习" && git push    # 结束时
```

### 换设备后怎么接续对话

在新机器上：
1. `git pull`
2. 在项目目录里打开 Claude Code
3. 说一句「**继续学习**」

我会自动读 `CLAUDE.md` + `PROGRESS.md`，知道你学到哪、哪里弱、下一步该干什么。**你不需要重新解释任何背景。**

### 每次会话结束时

跟我说一句「**更新档案**」，我会把本次的进展、测评结果、错题写进 `PROGRESS.md` 和 `CLAUDE.md`，然后你 commit + push。这是保证不断线的唯一动作。

---

## 四、不要提交进 Git 的东西

`.gitignore` 已配置排除：数据集、模型权重、checkpoint、虚拟环境、缓存。
这些文件很大且可重新生成，两台机器各自维护即可。

如果某个数据集处理脚本很重要，**提交脚本，不提交数据**。
