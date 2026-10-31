# [48] CaML

CaML 同时读取一个人的病史和候选药物的知识图表示，预测该药物任务的目标值。训练时先复制网络，让副本在一种药物任务上学习，再用副本的参数更新共同网络；换药物重复，得到能接收新药物任务的模型。

药物表示来自作者公开的 StarGraph 文件，含 744 种药物，每种 256 个数字。病史按作者的七个时间窗组织：当天、前一天、此前 2 至 7 天、8 至 30 天、31 至 90 天、91 至 365 天及更早。小例子用六个占位编码，分别表示两种诊断、两种操作和两种药物，因此每人有 42 项输入；每项以 25% 概率取 1。

原保险理赔数据和处理效应目标未公开。这里生成病史，取作者文件中的前三种药物，并生成目标：

```text
目标 = tanh(0.3 × 当天诊断0 - 0.2 × 当天诊断1 + 0.1 × 药物表示的第一个值)
```

两个诊断值为 0 或 1，药物值来自作者映射函数，三个系数为手动设置，tanh 把结果压到 -1 至 1。输出对应这个人工目标。

## 运行

在本目录执行：

```sh
pixi run --locked smoke
```

程序自动下载作者代码和药物向量。每个任务有 32 行，24 行组成训练池、8 行验证；每次实际用第一个八行批次更新一次。两个药物任务各做一次任务内更新和共同参数更新，第三个任务直接预测。输入、参数和结果在 `outputs/`，汇总为 `outputs/result.json`。

## 代码与改动

代码来自 [snap-stanford/caml](https://github.com/snap-stanford/caml/tree/88e2693ffa6996f8768b1eaa2b96f2a600643328)。作者 `src/models/xlearner.py` 定义预测网络，`metalearn/caml.py` 负责任务内训练，`metalearn/train_caml.py` 提供任务间更新，`task_utils.py` 读取药物表示，`causal_singletask/consts_and_utils.py` 创建优化器。

`reproduce.py` 为手动添加，生成输入、加载作者原类和更新片段，组织短任务并保存结果；`sources.json` 固定来源。网络采用作者正式训练使用的 `MLPLateConcatModel`，任务内为 AdamW，任务间为 SGD。

作者 `TaskEmbeddingMapper` 计算平均药物表示时会直接改写字典中的第一个药物向量。运行沿用此函数并保存实际输入。当前结果见 [运行记录](verification/result.json)。

原文：[48] Nilforoshan, H., Moor, M., Roohani, Y., et al. (2023). Zero-shot causal learning. Advances in Neural Information Processing Systems, 36, 6862–6901. [论文](https://arxiv.org/abs/2301.12292)
