# [29] KAT-GNN

为作者 GCN 分支准备人工诊断图和检验图，运行三轮训练。模型分别读取两张图和就诊间隔，再合并信息，输出每名患者两个类别的分数。

## 运行

```sh
pixi run --locked smoke
```

结果和日志在 `outputs/`，汇总见 `outputs/result.json`。

## 输入

生成 48 名人工患者，每人两至四次就诊，训练、验证、测试分别使用 32、8、8 人。诊断图用 276 项表示诊断类别，再加入年龄和性别；检验图用 80 个检验项目各十档表示结果，同样加入年龄和性别。

就诊节点连接当次概念和相邻就诊。病史连接权重为 1，人工知识连接权重为 0.8，两个方向都加入图。标签按患者序号交替取 0、1。作者原入口使用私人病史、检验表及未公开的数据模块，这里按其输入格式生成两张人工图。

## 代码和改动

代码来自 [KAT-GNN 作者仓库](https://github.com/DHLab-TSENG/kat-gnn-ehr-risk)。`author/models/` 定义双图网络和关注权重，`author/train/loop.py` 完成训练与测试，`author/private_all.py` 提供外层循环。

调用作者外层训练循环，按验证 AUPRC 保存最佳权重，指标未提高时加载已保存权重。三轮测试的后两轮执行了权重回退。

手动添加的文件：

- `reproduce.py`：生成人工输入，调用原循环并检查更新和重载结果。
- `fetch_author.py`：按 `source_manifest.json` 下载固定版本的作者文件。

已完成的运行记录在 `verification/smoke.json`。

原文：[29] Lin, K.-W., Kuo, Y.-C., Wang, H.-Y., Tseng, Y.-J. (2025). KAT-GNN: A Knowledge-Augmented Temporal Graph Neural Network for Risk Prediction in Electronic Health Records. arXiv:2511.01249. [论文](https://arxiv.org/abs/2511.01249)
