# [28] GraphCare

GraphCare 把住院病史中的疾病、操作和药物接到医学知识图上，预测下一次住院是否死亡。这里运行作者的预测网络和训练程序，病史采用公开 MIMIC-IV Demo，知识输入由公开 UMLS 子集及本地 GTE 向量组成。

## 怎么跑

在本目录执行：

```sh
pixi run --locked smoke
```

首次自动下载约 7.2 GB 的编码器和医学资料，另需环境及运行文件空间。默认在 CPU 上使用 float32，建议准备 16 GB 可用内存。医学资源统一缓存到仓库根目录 `.run/resources/graphcare/<版本>/`，其他 GraphCare 公开图入口会复用它。

程序整理病史、编码概念名称，训练一轮并进行验证、测试和保存后重载。结果在 `smoke_result.json`，训练详情在 `real_verification/result.json`，模型和名称向量在 `.private/`。当前例子有 27 名患者、35 个任务实例，完成 25 次参数更新，重载后的测试预测与保存前相同。

## 输入和输出

病史取自 [MIMIC-IV Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/) 的住院、诊断、操作及处方表。诊断和操作转为 CCS 类别，药物转为 ATC 类别，再按患者的住院时间排列。

公开医学对应表把病史类别连接到 UMLS 概念，作者公开的关系文件提供概念间的边。未找到关系的类别保留为独立节点。[GTE 编码器](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct/tree/a9af15a6372d7d6b25e9fb07c2ccb9e1fe645644)读取概念与关系的名称，生成模型使用的数字向量。当前图有 564 个节点、305 条关系，共编码 602 个名称。

模型沿医学关系传递信息，同时读取各次住院中实际出现的节点，把图和病史汇总后输出下次住院的死亡概率。标签由作者 `mortality_prediction_mimic4_fn` 生成。

这套公开关系和 GTE 向量替代了论文中的完整知识输入及 OpenAI 概念向量。作者总入口还将节点与就诊注意力 `alpha`、`beta` 都关闭；这里沿用该设置，关系边的注意力继续运行。因此结果对应作者公开入口的配置，与论文描述的完整 BAT 有差别。

## 本地数据和缓存

```sh
pixi run --locked reproduce --mimic-root "D:/my-mimic-iv" --patients 512 --epochs 1 --batch-size 16
```

需要 `hosp/admissions`、`diagnoses_icd`、`procedures_icd`、`prescriptions`，支持 CSV 及压缩 CSV。程序选择有 2 至 4 次完整住院的患者。作者读取器会丢弃不足一个批次的尾部样本，Demo 因此使用批大小 1；小队列也需相应减小 `--batch-size`。

已有下载资源可传 `--asset-cache "目录"`，其中应含 `model/` 和 `umls/`。已有名称向量可传 `--encoded-cache "目录"`，其中需有 `public_name_vectors.npz` 和 `name_vectors_manifest.json`。

`pixi run --locked smoke --precision bfloat16` 可降低编码器权重占用；速度取决于 CPU 对该精度的支持，实际精度会写入结果。

## 文件

作者的 `upstream/graphcare.py` 提供取图、训练和评估，`upstream/graphcare_/model.py` 与 `gnns.py` 定义网络，`splitter.py` 按患者拆分。`upstream_extra/task_fn.py` 和 `data_prepare.py` 提供任务及住院节点整理函数；其余 PyHealth 文件提供原数据类及读取、映射定义。

| 文件 | 来源和用途 |
| --- | --- |
| `smoke.py`、`reproduce_real.py` | 手动添加。准备小测试，调用作者训练与评估，保存模型并检查重载结果。 |
| `prepare_real.py` | 手动添加。读取病史，连接公开医学关系，调用作者任务函数。 |
| `prepare_author.py` | 手动添加。处理导入兼容，生成运行副本和改动记录。 |
| `vector_cache.py` | 手动添加。复用已计算的名称向量。 |
| `public_umls/fetch_assets.py` | 手动添加。下载固定资源。 |
| `public_umls/encode_names.py` | 手动添加。调用原 GTE 类的 CPU 分支编码名称。 |

## 改动

运行副本去掉当前任务不用的在线日志和额外依赖导入。原文件中写死的在线日志凭据已清除。公开图存在孤立节点，原取图函数可能据关系端点低估节点数；输入准备时把孤立节点排在前面，并同步修改关系和就诊编号，避免越界。

GTE 的自动加载器会要求可选 GPU 库，因此入口直接导入原类，使用其 CPU 分支。详细来源见 `graphcare_resources.json`、`pyhealth_resources.json`、`derived_resources.json` 和 `public_umls/assets.json`。

原文：[28] Jiang, P., Xiao, C., Cross, A., Sun, J. (2024). GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs. International Conference on Learning Representations. [论文](https://arxiv.org/abs/2309.08851)
