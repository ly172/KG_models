# [36] GreaseLM

GreaseLM 输入一道医学题、全部备选答案，以及每个选项对应的医学概念图。SapBERT 读取题目文字，图网络读取概念和关系，两路信息反复交换，最后给各选项打分。

数据来自 [作者公开的 MedQA 预处理包](https://github.com/snap-stanford/GreaseLM#data)。默认取两道训练题、一道验证题和一道测试题，保留所有选项文字，每次处理一道题。首次下载约 540 MB，建议准备 4 GiB 可用内存。

## 运行

在本目录执行：

```sh
pixi run --locked smoke
```

结果和日志在 `outputs/`，汇总为 `outputs/result.json`。程序训练一轮，保存模型，再重新加载并比较预测。

## 代码与改动

`author/greaselm.py` 是作者训练入口，`author/modeling/` 定义文字与图网络，`author/utils/` 处理数据、参数和辅助计算。`reproduce.py` 为手动添加，负责下载、选题、调用作者训练和检查保存结果。固定来源见 `author_source.json` 和 `sapbert_source.json`。

作者保存函数会删去冻结的概念表，造成严格重载缺项。运行副本把训练时实际使用的表一并保存，重载后预测一致。

作者初始化会把传入的 SapBERT 概念表重新写成随机数，随后冻结。因此实际训练使用固定随机概念表；这与论文附录 B.2 所述的 SapBERT 节点向量不同。小测试沿用这个初始化过程，当前记录见 [运行结果](verification/smoke.json)。

原文：[36] Zhang, X., Bosselut, A., Yasunaga, M., Ren, H., Liang, P., Manning, C. D., Leskovec, J. (2022). GreaseLM: Graph REASoning Enhanced Language Models. International Conference on Learning Representations. [论文](https://arxiv.org/abs/2201.08860)
