# [67] R-GCN 2018

本目录使用 2018 年原作者的 Keras/Theano 实现。输入是 AIFB 机构知识图和部分人员所属研究组的标签。图中包含人员、文献、研究组等对象，已经去除直接暴露研究组答案的关系。

模型按关系种类分别汇总邻居，经过两层计算，为每个人输出四个研究组的分数。公开图有 8,285 个节点，140 人用于训练，36 人用于测试。

## 运行

在本目录执行：

```sh
pixi run --locked smoke
```

程序下载作者代码和 AIFB 数据，执行原数据整理及两轮训练。预测在 `outputs/predictions.npy`，参数在 `outputs/weights.npz`，汇总在 `outputs/result.json`。运行 50 轮可用 `pixi run --locked python reproduce.py --epochs 50`。

## 代码与改动

代码来自 [tkipf/relational-gcn](https://github.com/tkipf/relational-gcn/tree/4bec1341dd46b72bf482f7ed26c2dca4533577f6)，图文件使用作者给出的 [AIFB 地址](https://www.dropbox.com/s/fkvgvkygo2gf28k/aifb_stripped.nt.gz?dl=1)。作者的 `rgcn/layers/` 定义图卷积，`data_utils.py` 和 `prepare_dataset.py` 读取和整理关系，`train.py` 建立模型并训练。

`reproduce.py` 为手动添加，负责下载、转换旧语法、调用作者程序及保存结果。Python 2 的字典、迭代器、整数除法等写法已转换为 Python 3 对应写法。

本机采用 Theano 的 Python 执行方式。默认计算图优化会把很多关系的加法合并成一个无法执行的 81 输入运算，因此设置 `optimizer=None`，逐项执行原加法。修改记录在 `verification/python3.patch`，当前节点分类结果见 [运行记录](verification/result.json)。

原文：[67] Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., Welling, M. (2018). Modeling Relational Data with Graph Convolutional Networks. The Semantic Web: ESWC 2018, Lecture Notes in Computer Science 10843, 593–607. [论文](https://arxiv.org/abs/1703.06103)
