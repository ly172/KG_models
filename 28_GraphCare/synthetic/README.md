# [28] GraphCare：人工病史和图

用人工就诊、概念图和向量运行作者 GraphCare 网络。模型沿医学关系读取邻居信息，再结合各次就诊出现的概念，输出每名患者的二分类分数。

## 运行

```sh
pixi run --locked smoke
```

结果在 `outputs/result.json`，包括训练、测试和保存后重载结果。公开医学图例子见 [public_umls](public_umls/README.md)。

## 输入

固定种子生成 256 名患者、32 个概念、5 种关系，以及概念和关系各 1,536 维的向量。病史、关系、向量和标签均为人工数据，训练、验证和测试分别使用 128、64、64 人。数据生成规则在 `simulate_inputs.py`。

模型使用三层 BAT、128 维内部表示及边关注。作者 `single_run` 实际将节点关注 `alpha` 和就诊关注 `beta` 都关闭，本例采用相同设置。论文的 BAT 包含这两项关注计算，阅读结果时需结合这一差异。

## 代码和改动

模型、损失、训练和评估函数来自 [GraphCare](https://github.com/pat-jj/GraphCare)，保存在 `upstream/`，固定版本见 `source.json`。运行副本移除未使用的依赖和在线日志调用，使用本地输出。

手动添加的文件：

- `simulate_inputs.py`：生成人工图、就诊、向量和标签。
- `prepare_author.py`：建立运行副本。
- `reproduce.py`：调用作者训练、评估并检查保存后预测。
- `verify_compatibility.py`：检查原入口的导入错误和兼容处理。

已完成的运行记录在 `verification/smoke.json`，公开输入来源见 [input_resources.md](input_resources.md)。
