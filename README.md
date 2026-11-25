# LeNet-5 重访：哪些设计选择才真正重要？

> 对 LeCun、Bottou、Bengio 和 Haffner（1998）关于 MNIST 的忠实性审计。

**状态：E01–E05 与两项对照全部完成。**

本仓库用 PyTorch 从零重建 LeNet-5，而不是调用 `torchvision.models` 里的“LeNet”：
网络结构、输入归一化、稀疏连接表、固定 RBF 码字、损失函数和 SDL 优化器都按论文实现，
并且用单元测试把 60,000 个参数、340,908 条连接这些论文数字钉死。

---

## 问题

1998 年的 LeNet-5 论文在 MNIST 上报出 **0.95%** 的测试误差。它建立在一组与今天教程很不一样的设计选择上：

1. 稀疏的 C3 连接（Table I，1,516 个参数，而不是稠密的 2,416 个）；
2. 可训练的 2×2 求和子采样（一个系数加一个偏置，再接压缩函数）；
3. 固定的 +1/−1 RBF 输出码字，配合只惩罚正确类距离的 MSE（Eq. 8）；
4. 对角 Gauss–Newton 近似的随机 Levenberg–Marquardt（SDL，附录 C）。

现代“LeNet”教程几乎把这些全部换成稠密卷积、MaxPool、Softmax + 交叉熵和 Adam，却仍能得到约 1% 的误差。
**这些选择中，哪些在 1998 年是承重的，哪些只是历史文物？**

## 实验计划

| # | 问题 | 现状 | 结果 |
| --- | --- | :---: | --- |
| E01 | 忠实的架构能否复制 0.95% 的效果？ | 已完成 | 1.72% |
| E02 | 稀疏的 C3 有必要吗？ | 已完成 | 稠密 1.47% vs 稀疏 1.72% |
| E03 | 可训练的求和池与 MaxPool 有什么区别？ | 已完成 | 求和 1.03% vs MaxPool 1.44% |
| E04 | 固定 RBF + MSE 与 Softmax + CrossEntropy 孰优？ | 已完成 | RBF 1.03% vs Softmax 1.22% |
| E05 | SDL 与 SGD：它真的收敛得更快吗？ | 已完成 | SDL 1.72% vs SGD 90.05% |

E02/E05 与 E01 共用 SDL 和 batch 16；E03/E04 因为 MaxPool/Softmax 改用 Adam，
E04 另设一个同 Adam、同 batch 的 RBF 对照，避免把“换优化器”误读成“换输出头”。

| 实验 | 唯一变量 | 对照 |
| --- | --- | --- |
| E02 | `sparse_c3: true → false` | 1,516 vs 2,416 个 C3 参数 |
| E03 | `subsampling: trainable → max` | 44 个可训练参数 vs 0；SDL → Adam |
| E04 | `head: rbf → softmax` | Eq. 8 的距离 vs 交叉熵 |
| E05 | `optim.name: sdl → sgd` | 每参数步长 vs 全局步长 |

## 复刻表

<!-- BEGIN:REPLICATION -->
| 模型 | 测试错误率 (%) | 参数 | 来源 |
| --- | ---: | ---: | --- |
| 纸（无扭曲，20 遍） | 0.95 | 60,000 | LeCun 1998, Fig. 9 |
| 纸（扭曲训练） | 0.80 | 60,000 | LeCun 1998, Fig. 9 |
| 我们的（忠实架构，SDL） | 1.72 | 60,000 | `results/runs/e01-faithful` |
| 我们的（现代 LeNet） | 0.83 | 61,706 | `results/runs/modern-lenet` |
| 我们的（稠密 C3） | 1.47 | 60,900 | `results/runs/e02-dense-c3` |
| 我们的（MaxPool） | 1.44 | 59,956 | `results/runs/e03-maxpool` |
| 我们的（RBF + Adam） | 1.03 | 60,000 | `results/runs/e04-rbf-adam` |
| 我们的（Softmax + Adam） | 1.22 | 60,850 | `results/runs/e04-softmax` |
| 我们的（SGD） | 90.05 | 60,000 | `results/runs/e05-sgd` |
<!-- END:REPLICATION -->

> 该表由 `python scripts/make_tables.py` 从 `results/runs/*/metrics.json` 重新生成，
> 请不要手工编辑表格内容。

<!-- BEGIN:RESULTS -->
## 结果

全部实验都在同一台 16 核 CPU 上完成：60,000 张训练图、10,000 张测试图、20 遍。
每个 run 的配置、逐 epoch 曲线和指标都在 `results/runs/<run-name>/`。

### E01：忠实架构 + SDL

| 指标 | 本文 | 论文 |
| --- | ---: | ---: |
| 最终测试错误率 | **1.72%** | 0.95% |
| 训练错误率（第 20 遍） | 1.74% | 0.35%（第 19 遍） |
| 最好的一遍（测试） | 1.69% | — |
| 参数量 / 连接数 | 60,000 / 340,908 | 60,000 / 340,908 |
| 墙钟时间 | 862 s | — |

![E01 误差曲线](results/runs/e01-faithful/curves.png)

三点观察：

1. **收敛形态与论文 Fig. 5 一致**：第 2 遍测试误差从 44.8% 掉到 3.5%，第 5 遍 2.1%，
   第 8 遍 1.82%，此后以 5e-5、1e-5 的步长缓慢磨到 1.7%。论文说“10 到 12 遍基本收敛”，
   本文也是 10 遍后进入平台。
2. **没有过训练**：训练误差 1.74% 与测试误差 1.72% 几乎相同；论文是训练 0.35%/测试 0.95%。
   本文是“欠拟合”，限制因素不是容量，而是优化预算。
3. **差距主要来自更新次数**：本文每遍 3,750 次更新（batch 16），20 遍 75,000 次；
   论文按单个模式更新，20 遍 1,200,000 次，正好 16 倍。把 batch 设成 1 可以严格对齐，
   但在 CPU 上需要约 5–10 小时，超出本项目的复现预算。

### 四个设计选择，各值多少？

| 实验 | 被检验变量 | 测试误差 | 对照 | 读法 |
| --- | --- | ---: | ---: | --- |
| E02 | 稀疏 C3 → 稠密 6×5×5 | 1.47% | E01 1.72% | 稀疏表省 900 个参数，但精度反而低 0.25pp；**不是承重设计** |
| E03 | 可训练求和 → MaxPool | 1.44%（最好 1.30%） | E04-RBF 1.03% | MaxPool 每遍更新次数还多一倍，仍更差；**求和池是承重的** |
| E04 | RBF+MSE → Softmax+CE | 1.22% | RBF 1.03% | 固定码字 RBF 不劣于可学习 Softmax；**不是历史文物** |
| E05 | SDL → SGD（momentum 0.9） | 90.05% | E01 1.72% | 普通 SGD 在这个架构/初始化下逃不出平台；**SDL 的每参数步长是承重的** |
| — | 现代 LeNet（稠密 C3 + MaxPool + ReLU + Softmax + Adam） | 0.83% | E04-Softmax 1.22% | 现代组合最好，但一次改了多个变量 |

两点方法学说明：

- E03（batch 16）与 E04-RBF（batch 32）的 batch 不同，E03 的更新次数是 E04 的两倍，
  仍然更差，所以“求和池优于 MaxPool”的结论是保守的；
- E05 的 SGD 用了 0.01 → 0.0002 的阶梯（按 1/μ 与 E01 对齐）；在另外试过的 5e-4 全局步长下
  同样停在 90%，所以问题更像是全局步长处理不了这个网络的 ill-conditioning，而不是单点调参失误。

### 这些设计选择，哪些承重？

| 设计 | 结论 | 证据 |
| --- | --- | --- |
| 卷积骨架（局部感受野、权重共享、逐层下采样） | **承重** | 60,000 参数、无数据增强到 1.03%（E04-RBF） |
| 稀疏 C3 连接 | 历史文物 | E02 稠密版更好（1.47% vs 1.72%），还少一层连接表维护 |
| 可训练 2×2 求和子采样 | **承重** | E04-RBF 1.03% vs E03 MaxPool 1.44%，且 E03 更新更多 |
| 固定 +1/−1 RBF + Eq. 8 | **在本预算下仍具竞争力** | E04-RBF 1.03% vs E04-Softmax 1.22%，同 Adam、同 batch |
| SDL（对角 Gauss–Newton） | 对 SGD 承重，但被 Adam 超越 | E05 SGD 90.05% vs E01 SDL 1.72% vs E04-RBF Adam 1.03% |

一句话：**1998 年的选择不是一整套“历史文物”，也不是一整套“必要组件”**。
真正承重的是卷积骨架、可训练求和池和固定 RBF；稀疏 C3 可以去掉；
SDL 在 1998 年是让这个网络可训练的关键，但今天的 Adam 用更少的二阶信息做了同样的事。

### 现代对照

同一 20 遍预算、同样的 32×32 / −0.1 / 1.175 输入，换成
稠密 C3 + MaxPool + ReLU + Softmax + CrossEntropy + Adam（61,706 参数）：

| | 忠实 LeNet-5（SDL） | 现代 LeNet（Adam） |
| --- | ---: | ---: |
| 测试错误率 | 1.72% | **0.83%** |
| 训练错误率 | 1.74% | 0.01% |
| 每遍更新次数 | 3,750 | 1,875 |
| 训练/测试关系 | 欠拟合 | 明显过拟合 |

现代对照用一半的更新次数拿到更低的测试误差，也低于论文的 0.95%。
它证明“1998 年的四项选择不是达到约 1% 的必要条件”；但 E02–E05 又说明，
其中三项在单变量对照下仍然有价值，不能简单归为历史文物。
<!-- END:RESULTS -->


---

## 一、项目做什么

- **不是** 现代 LeNet 教程的又一个复读：默认配置严格对应论文 Figure 2 的七层结构与 Section II.B 的数字；
- **是** 一个可做消融实验的对象：E02–E05 只改一个设计维度（必要时配套换优化器），
  逐个回答“稀疏 C3 / 求和池 / RBF / SDL 各自值多少”；
- **是** 一套把论文公式落成代码的实现：Eq. 6 的缩放 tanh、Eq. 8 的损失、Eq. 21 的步长、
  Eq. 27 的 Gauss–Newton 对角递推，都能在源码里逐行对上。

核心数字（由 `tests/test_model.py` 与 `tests/test_connect.py` 锁定）：

| 层 | 输出 | 参数 | 连接 |
| --- | --- | ---: | ---: |
| 输入 | 1@32×32 | 0 | 0 |
| C1 | 6@28×28 | 156 | 122,304 |
| S2 | 6@14×14 | 12 | 5,880 |
| C3 | 16@10×10 | 1,516 | 151,600 |
| S4 | 16@5×5 | 32 | 2,000 |
| C5 | 120 | 48,120 | 48,120 |
| F6 | 84 | 10,164 | 10,164 |
| 输出（RBF） | 10 | 0 | 840 |
| **合计** | | **60,000** | **340,908** |

## 二、面向对象的设计

### 2.1 网络：`LeNet5`

`src/lenet5/models/lenet5.py`

| 属性 | 含义 |
| --- | --- |
| `config` | `ModelConfig`，记录稀疏 C3、子采样方式、输出头、激活、初始化等开关 |
| `blocks` | 按前向顺序排列的七个 `Module`：C1、S2、C3、S4、C5、flatten、F6 |
| `head` | `RBFHead`（固定码字）或 `SoftmaxHead` |

| 方法 | 作用 |
| --- | --- |
| `forward(x)` | 前向传播，返回 10 个距离（RBF）或 10 个 logits（softmax） |
| `embed(x)` | 返回 F6 的 84 维编码，可用于可视化/检索 |
| `parameter_count()` | 有效参数量（C3 只数被 mask 选中的连接），默认 60,000 |
| `connection_count()` | 按论文口径统计连接数（含偏置），默认 340,908 |
| `describe()` | 返回逐层表格，`describe_table()` 可打印 Table I |
| `predict(x)` | RBF 取 `argmin`，softmax 取 `argmax` |
| `LeNet5.faithful()` / `LeNet5.modern()` | 两个预设，后者是复刻表中的现代对照 |

### 2.2 层：每个 block 都会回答两个问题

`src/lenet5/models/blocks.py`

SDL 需要的不只是梯度，还需要“损失对该层权重的二阶导”。因此每个 block 都统一实现：

| 方法 | 回答的问题 |
| --- | --- |
| `forward(x)` | 前向值，同时缓存输入 `x` 与预激活 `a` |
| `activation_deriv_sq()` | `f'(a)^2`，供 Eq. 27 的递推使用 |
| `input_d2(d2)` | 把上游的 `d2` 传回输入端：`(W^2)^T d2`（卷积用转置卷积） |
| `weight_hessian(d2)` | 本层权重的 `h_w = Σ d2·x^2` 与偏置的 `h_b = Σ d2` |
| `effective_numel()` | 有效参数量；`MaskedConv2d` 只数 mask 为 1 的连接 |

具体类：

- `MaskedConv2d`：C1/C3/C5 的 5×5 卷积；C3 用 Table I 的 0/1 mask，缺失连接永远为 0；
- `ConvBlock`：卷积 + 缩放 tanh，C3 的 `weight_hessian` 也会再乘一次 mask；
- `SubsampleBlock`：`sigmoid`/缩放 tanh 作用于 `a·Σ2x2 + b`，C 个系数 + C 个偏置；
- `LinearBlock`：F6 的全连接 + 缩放 tanh；
- `RBFHead`：固定 `+1/−1` 码字，输出 `Σ_j (x_j − w_ij)^2`；
- `FlattenBlock`：把 `(B,120,1,1)` 变成 `(B,120)`，同时为 Hessian 递推保留形状变换。

### 2.3 训练器：`Trainer`

`src/lenet5/runner/trainer.py`

| 属性 | 含义 |
| --- | --- |
| `model` / `data` / `config` | 注入的三个依赖 |
| `criterion` | `RBFLoss`（Eq. 8）或 `CrossEntropyLoss` |
| `optimizer` | `SDL` / `SGD` / `Adam` |
| `hessian` | `GaussNewtonHessian`，仅 SDL 时存在 |
| `history` | 每个 epoch 的 loss、error、学习率、耗时 |
| `run_dir` | `config.yaml`、`history.csv`、`curves.png`、`metrics.json`、checkpoint 的落盘位置 |

| 方法 | 作用 |
| --- | --- |
| `fit()` | 完整实验：按 schedule 设 η、估计 h、训练、验证、存 checkpoint、收尾 |
| `train_epoch(loader)` | 单遍训练，返回平均损失与错误率 |
| `validate(loader)` / `evaluate()` | 验证/测试，返回 loss 与 error |
| `predict`（经 `model.predict`） | 单批预测 |
| `save_checkpoint` / `load_checkpoint` | 断点续训 |
| `metrics()` | 汇总可写入 `metrics.json` 的字典 |
| `plot_curves()` | 画 train/eval 误差曲线 |

### 2.4 优化器：`StochasticDiagonalLevenbergMarquardt`

`src/lenet5/optim/sdl.py`

- 继承 `torch.optim.Optimizer`，因此 `state_dict()`、`zero_grad()`、学习率调度都能直接用；
- `step()` 执行 Eq. 21：`p -= lr * g / (h + mu)`；
- `set_hessian(mapping)` 在每个 epoch 开始前注入固定的 `h`；
- `learning_rates()` 返回每个参数的有效步长，便于检查 SDL 是否病态。

`src/lenet5/optim/hessian.py` 的 `GaussNewtonHessian.estimate()` 在 500 个样本上做一次前向 +
反向的 `d2` 递推（Eq. 27），再按 Eq. 23–25 累加 `h_w`、`h_b`。

### 2.5 数据与配置

- `MNISTData`：`setup()` 读取 IDX、补齐到 32×32、按 `background=-0.1 / foreground=1.175` 归一化；
  `train_loader()`、`val_loader()`、`test_loader()` 统一管理 batch、shuffle 与随机种子；
- `ExperimentConfig`：`from_yaml()` / `dump()` / `validate()`，把网络、数据、优化、训练四组参数分开；
- `lr_for_epoch()`：把论文的 `0.0005 → 0.0002 → 0.0001 → 0.00005 → 0.00001` 阶梯表变成函数。

## 三、忠实性清单

### 3.1 与论文一致的部分

| 论文位置 | 实现 |
| --- | --- |
| Fig. 2 / Section II.B：C1 6@28×28、S2 6@14×14、C3 16@10×10、S4 16@5×5、C5 120、F6 84、输出 10 | `LeNet5.faithful()`，`test_parameter_and_connection_counts` |
| Table I：C3 的 16×6 稀疏连接 | `C3_INPUTS`，`test_mask_matches_table`；权重参数 1,516 |
| 60,000 参数 / 340,908 连接 | `parameter_count()` / `connection_count()`，测试锁定 |
| Eq. 6：`f(a)=1.7159·tanh(2a/3)` | `ScaledTanh` |
| Eq. 7：`y_i=Σ_j (x_j-w_ij)^2`，码字 +1/−1 | `RBFHead` |
| Eq. 8：`E=(1/P)Σ y_{D_p}` | `RBFLoss` 只 gather 正确类 |
| Eq. 21 + Eq. 23–27：`ε=η/(μ+h)`，Gauss–Newton 对角 | `SDL` + `GaussNewtonHessian` |
| Section III.B：20 遍；η 阶梯；μ=0.02；h 每遍前在 500 样本上重估 | `e01_faithful.yaml`、`Trainer.fit()` |
| Section III.A：32×32；背景 −0.1、前景 1.175；中心化 | `pad_to_32` + `normalise("faithful")` |
| Fig. 9：0.95%（无扭曲）、0.80%（扭曲训练） | 复刻表的“纸”行 |

### 3.2 已知偏差（以及为什么）

1. **初始化**：附录 A 规定 `U(−2.4/F_i, 2.4/F_i)`。在 float32 + 本文的数据管线下，这个规则让
   C1 输出标准差只有约 0.11，逐层衰减后 F6 只剩约 0.005，SDL 的有效步长把 F6 直接推到饱和，
   训练停滞在 90% 错误率。E01 因此使用保持方差的 `U(−√(6/F_i), √(6/F_i))`（`init: variance`）；
   想跑严格版可以 `--init paper`，但需要预期它不收敛。这是本项目最重要的偏差，复刻表因此不是“逐位复现”。
2. **子采样压缩函数**：论文正文写的是 sigmoid；本文使用与 Eq. 6 相同的缩放 tanh。理由同上：
   logistic sigmoid 把每层均值抬到 0.5，且导数最大只有 0.25，逐层把方差压到约 1/16，C3 的输入几乎变成常数。
3. **批大小**：论文按单个模式更新（Appendix B: “one single training sample (or a small number of samples)”）。
   本文按 16 张图求平均梯度，因此每个 epoch 只有 3,750 次更新，而论文是 60,000 次。这会让收敛慢一个数量级，
   也是 0.95% 与本文结果之间的主要差距来源。配置里的 `batch_size: 1` 可以严格复现，但 CPU 上不可行。
4. **未做扭曲增强**：0.80% 那一行需要仿射/弹性扭曲的 540,000 个额外样本，本项目没有实现。
5. **未实现 MAP 准则（Eq. 9）**：MNIST 单字符实验用 Eq. 8；Eq. 9 是给带拒识的文档识别准备的。

## 四、运行

```bash
# 1) 安装依赖（CPU 即可）
pip install -e .
pip install pytest matplotlib

# 2) 下载 MNIST（自动校验 MD5）
python scripts/download_mnist.py

# 3) 复现 E01（20 遍，CPU 约 15 分钟）
python scripts/train.py --config configs/e01_faithful.yaml

# 4) 跑测试（21 项）
python -m pytest -q

# 5) 重新生成复刻表并回写 README
python scripts/make_tables.py
```

快速冒烟：

```bash
python scripts/train.py --config configs/e01_faithful.yaml --epochs 1 --limit-train 2000 --limit-eval 500
```

其它实验：

```bash
python scripts/train.py --config configs/e02_dense_c3.yaml   # E02 稠密 C3
python scripts/train.py --config configs/e03_maxpool.yaml    # E03 MaxPool
python scripts/train.py --config configs/e04_softmax.yaml    # E04 Softmax
python scripts/train.py --config configs/e04_rbf_adam.yaml   # E04 RBF 对照
python scripts/train.py --config configs/e05_sgd.yaml        # E05 SGD
python scripts/train.py --config configs/modern_lenet.yaml   # 现代对照
```

每次运行写入 `results/runs/<run-name>/`：

```
config.yaml        # 本次实验的完整配置（可追溯）
history.csv        # 每个 epoch 的 train/eval loss、error、lr、耗时
history.json
curves.png         # train/eval 误差曲线
metrics.json       # 汇总指标，make_tables.py 的数据源
checkpoints/{last,best}.pt
```

## 五、目录结构

```
configs/                 E01–E05 与现代对照的 YAML
src/lenet5/
  config.py              ExperimentConfig / ModelConfig / LR schedule
  data/mnist.py          IDX 解析、32×32 padding、−0.1/1.175 归一化、DataLoader
  models/
    connect.py           Table I（16×6 稀疏连接表）
    activations.py       Eq. 6 的缩放 tanh、sigmoid、ReLU
    blocks.py            带二阶递推的卷积/子采样/全连接/RBF block
    lenet5.py            七层网络与参数/连接统计
  losses/                Eq. 8 的 RBFLoss
  optim/
    sdl.py               Eq. 21 的 SDL
    hessian.py           Eq. 23–27 的 Gauss–Newton 对角估计
  runner/
    trainer.py           fit / train_epoch / validate / checkpoint
    history.py           每个 epoch 的指标
  analysis/tables.py     复刻表生成
scripts/                 train.py / make_tables.py / download_mnist.py
tests/                   结构、数值、数据、端到端测试
results/                 runs / tables / figures
paper/lecun-98.pdf       原论文
```

## 六、参考

- Y. LeCun, L. Bottou, Y. Bengio, P. Haffner. *Gradient-Based Learning Applied to Document Recognition.*
  Proceedings of the IEEE, 86(11):2278–2324, 1998. 本仓库 `paper/lecun-98.pdf`。
- Y. LeCun, L. Bottou, G. Orr, K.-R. Müller. *Efficient BackProp.* 1998（初始化与二阶方法的背景）。
- MNIST：http://yann.lecun.com/exdb/mnist/（本仓库只使用官方四个 IDX 文件）。

## 七、许可

MIT，见 [LICENSE](LICENSE)。
