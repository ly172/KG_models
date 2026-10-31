# [49] KG-TREAT

用四名人工患者的病史和成对小图运行作者分类网络，完成两次更新。网络包含 12 层病史处理、5 层图处理，以及治疗和未治疗两种情形的结局预测。

## 运行

```sh
pixi run --locked smoke
```

结果在 `run/result.json`，包含输出、参数更新和重载检查。

## 输入和输出

每名患者有六项病史，每项包括医疗编号、项目种类、距参考点的天数和所属 30 天组。治疗关注图与结局关注图各有五个节点、八条有向边。治疗和结局均取 0 或 1，四种组合各出现一次。

32 个概念的初始向量各有 1,024 个数，由固定种子 20261029 按均值 0、标准差 0.1 生成。病史编号、图和标签均为人工输入；概念宽度采用作者指定的 BioLinkBERT-large 规格。

模型让病史与图交换信息，返回治疗、未治疗两种情形的结局分数、按实际治疗组取出的分数及总损失。本例从初始化状态训练两步；论文在 MarketScan 人群上进行的预训练另需原数据和任务设置。

## 代码和改动

作者代码来自 [KG-TREAT](https://github.com/ruoqi-liu/KG-TREAT)，保存在 `upstream/`。`model/` 定义病史网络、图网络、预测和损失，`utils/` 提供配置。

运行副本调整已迁移的 Transformers 导入位置。已确认的作者行为包括：结局损失先对整批求平均再乘治疗标记，使另一治疗分支也收到梯度；构造后的初始化会重新写入概念表。两项行为均保留在当前计算中。

手动添加的文件：

- `reproduce.py`：生成人工输入，调用作者网络并安排两次更新。
- `prepare.py`：生成兼容副本。
- `verify_initialization.py`：比较初始化后的概念表与输入向量。

固定来源在 `source.json`，已完成的运行记录在 `verification/`。

原文：[49] Liu, R., Wu, L., Zhang, P. (2024). KG-TREAT: Pre-training for Treatment Effect Estimation by Synergizing Patient Data with Knowledge Graphs. Proceedings of the AAAI Conference on Artificial Intelligence, 38. [论文](https://arxiv.org/abs/2403.03791)
