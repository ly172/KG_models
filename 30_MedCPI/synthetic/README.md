# [30] MedCPI：人工病史

用人工就诊和患者图运行作者 Integrate 网络。模型先编码病史，再用病史给图中各节点分配关注权重，汇总后与病史表示拼接，输出每名患者的一个分数。

## 运行

```sh
pixi run --locked smoke
```

默认训练三轮，按验证 AUROC 选择权重并测试，结果在 `outputs/result.json`。接入公开医学知识可运行：

```sh
pixi run --locked public-ddb
```

对应结果在 `outputs/public_ddb/result.json`。

## 输入

默认生成 48 名患者，每人两至四次就诊、九个概念；训练、验证、测试分别使用 32、8、8 人。概念从 32 个人工编号中循环选择，每次就诊取三个，分别代表诊断、操作和药物。相邻就诊间隔 60 天。患者与就诊、就诊与概念、相邻概念分别相连，六种关系表示三类连接的两个方向。标签交替取 0、1。

公开知识例子使用 GreaseLM 作者发布的 DDB 医学名称和关系，选取 96 个概念，并调用原配 SapBERT 处理文字。患者、就诊和标签继续按上述方式生成。来源地址在 `public_inputs_manifest.json`。

## 代码和改动

模型来自 [MedCPI](https://github.com/hw146/MedCPI)。`author/scripts/integrate_model.py` 定义网络，`integrate_dataset.py` 整理输入，`train_integrate_formal.py` 提供正式训练函数，`sapbert_map_mentions.py` 提供原文字编码函数。

两个入口都调用作者 `run_single_seed`，包含原训练、AUROC 选模、停止规则和重载测试。作者结束报告中的两个重复字段引用了未定义的 `seed_results`，会在训练完成后报错；运行副本删除这两个重复字段，使用同一报告已有的局部参数计数。

手动添加的文件：

- `reproduce.py`：生成人工病史，调用作者正式训练并检查重载结果。
- `fetch_author.py`：按固定清单下载作者文件。
- `reproduce_ddb.py`：准备公开 DDB、SapBERT 和人工病史，调用同一训练入口。

已完成的两个入口记录在 `verification/`。
