# [37] Graph Neural Prompting

模型读取题目、四个选项和相关概念图，先让图中的概念交换信息，再结合题目选取图中的内容。选出的内容变成一小段数字提示，放在题目前面，交给 FLAN-T5-XL 生成答案。训练更新图网络及其投影部分，语言模型参数固定。

## 运行

在本目录执行：

```sh
pixi run --locked smoke
```

首次运行下载约 11.4 GB 的语言模型参数，请准备至少 15 GB 可用磁盘空间。运行使用 CPU 和 float32。输入文件随代码提供，约 1.9 MB。

结果在 `outputs/result.json`，训练后的图网络参数在 `outputs/adapter.pt`。程序完成两次更新后，会重新创建图网络、加载参数，并生成一次答案。当前运行记录见 `verification/result.json`。

## 输入和输出

输入来自 OpenBookQA 训练集的题目 `7-980`，询问太阳与哪些现象有关。题干和四个选项共 43 个词片，配套四个 ConceptNet 子图。图中保存概念、关系、题干与选项对应的节点，以及每个概念原有的 1024 维向量。输出是答案文字，以及训练损失和恢复检查结果。

四个原图有 161、236、46、86 个概念。作者程序最多保留 200 个节点，其中一个位置用于上下文节点；第二个图因此保留前 199 个概念。小输入保留实际需要的 442 行原概念向量，并同时调整概念编号和向量行号。

数据取自 [作者公开预处理包](https://drive.google.com/file/d/1w_0x9BxQ2xvdl-PkBP5AU5VusEFQecTu/view)。`data/provenance.json` 记录原文件和所取的子集。本例用同一道训练题检查参数更新及生成过程。

## 代码

| 文件 | 用途 |
| --- | --- |
| `reproduce.py` | 手动添加。调用作者图处理和模型，组织两次更新、保存、恢复及生成答案。 |
| `prepare.py` | 手动添加。下载固定版本源码和语言模型参数。 |
| `build_public_example.py` | 手动添加。从原公开数据包提取这一题、四个图和对应向量。 |
| `sources.json` | 手动添加。保存代码与参数文件的下载地址。 |
| `data/example.json`、`data/concept_embeddings.npy` | 从公开包提取的题目、图和原概念向量。 |
| `data/provenance.json` | 手动添加。记录原数据文件与所取子集。 |
| `cache/author/llm.py`、`gnn.py` | 作者的图模型、提示投影、语言模型连接和关系损失。 |
| `cache/author/utils/data_utils.py` | 作者的图读取、上下文节点、反向边和关系采样。 |

需要重新提取小输入时，运行 `pixi run --locked python build_public_example.py --data-root "公开包解压后的data目录"`。

## 改动

使用作者参数：`scaled_distmult=False`、`link_negative_adversarial_sampling=True`、`p_dropout_gnn=0.2`。前两项使关系评分采用原数值尺度，并按作者方法给负样本加权。训练损失由回答损失加 0.1 倍关系损失组成。

网络使用作者 OpenBookQA 脚本的三层、1024 维图表示和一层文字与图的注意力。小测试固定随机种子 12024，每次一题，以 `0.0001` 的固定学习率更新两次；作者完整脚本使用每批八题、50 轮训练和 500 步学习率预热。小测试的训练循环为手动添加。

设备从作者写死的 CUDA 改为 CPU，计算精度从 bfloat16 改为 float32。运行时的源码差异保存在 `outputs/cpu_precision.patch`。

源码来自 [amazon-science/GNP](https://github.com/amazon-science/GNP/tree/a8b1a7376a513d739324ae8bbe30f5f3758f2f8b)，语言模型来自 [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl/tree/7d6315df2c2fb742f0f5b556879d730926ca9001)。

原文：[37] Tian, Y., Song, H., Wang, Z., et al. (2024). Graph Neural Prompting with Large Language Models. Proceedings of the AAAI Conference on Artificial Intelligence, 38(17), 19080–19088. [论文](https://arxiv.org/abs/2309.15427)
