# [14] GRAM

GRAM 根据一个人历次住院的诊断，预测下一次住院的诊断类别。这里调用作者的数据整理、分类树和训练程序，默认使用公开 MIMIC-IV Demo。

## 怎么跑

在本目录执行：

```sh
pixi run --locked smoke
```

程序自动下载约 16 MB 的 [Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/)，整理病史并训练一轮，随后计算验证和测试代价。默认使用 CPU。结果在 `smoke_result.json`，模型和日志在 `.private/smoke/`；公开病史缓存在仓库根目录 `.run/mimic_demo_2_2/`。

当前小测试包含 15 名患者、37 次住院和 22 对相邻住院，按患者分为训练 11 人、验证 1 人、测试 3 人。保存后的参数数组通过有限数值检查。

## 输入和输出

输入是按时间排列的住院诊断，以及 [CCS 2015 疾病分类表](https://hcup-us.ahrq.gov/toolssoftware/ccs/Multi_Level_CCS_2015.zip)。诊断来自 `hosp/diagnoses_icd.csv.gz`，住院时间来自 `hosp/admissions.csv.gz`。CCS 告诉模型每个诊断属于哪些上层类别。

每个诊断先从自身及上层类别取信息，一次住院的诊断再合成一个表示。模型依次读取多次住院，输出下次住院各诊断类别的分数。标签沿用作者规则：ICD-9 通常取前三位，E 开头取前四位。

小测试选取有 2 至 4 次完整住院、全部使用 ICD-9 诊断的患者，保留各人的整段病史。字段转换后交给作者原读取器，模型采用默认的 128 维设置。小测试采用作者程序支持的直接初始化，`upstream/` 另提供论文所用的 GloVe 预训练工具。

读取自己的 MIMIC-IV 时可执行：

```sh
pixi run --locked reproduce --mimic-root "D:/my-mimic-iv" --patients 160 --epochs 1
```

需要 `hosp/admissions` 和 `hosp/diagnoses_icd`，支持 CSV 及压缩 CSV。至少需 10 名符合条件的患者，保证训练、验证和测试均有数据；输出在 `.private/current/`。

## 文件

`upstream/process_mimic.py`、`build_trees.py`、`gram.py` 是作者的数据整理、分类树和模型训练代码；`create_glove_comap.py`、`glove.py` 是作者的可选预训练程序。

| 文件 | 来源和用途 |
| --- | --- |
| `smoke.py`、`reproduce.py` | 手动添加。准备公开小测试或本地队列，调用作者程序并保存结果。 |
| `adapt_mimic.py` | 手动添加。选择完整病史，把 MIMIC-IV 字段转换为作者读取器所需的格式。 |
| `prepare_author.py` | 手动添加。生成 Python 3 运行副本并处理空分类层。 |
| `run_author.py` | 手动添加。设置线程和随机种子，执行作者入口。 |
| `verify_compatibility.py` | 手动添加。检查转换后的非空表、空层处理及梯度。 |

## 改动

作者代码使用 Python 2，运行时转换为 Python 3。原 `build_tree` 遇到空分类层会得到一维空数组，随后访问第二维报错；运行副本给空表保留正确列数，已有非空层的数值与原函数一致。

病史从论文的 MIMIC-III 改为 MIMIC-IV。队列检查按作者划分比例确认三部分都有数据，因此公开 Demo 的 15 人队列可以运行。

作者代码固定到 [28cb665](https://github.com/mp2893/gram/tree/28cb66511c0c42629386b96dafe18c2b316c184f)。

原文：[14] Choi, E., Bahadori, M. T., Song, L., Stewart, W. F., Sun, J. (2017). GRAM: Graph-based Attention Model for Healthcare Representation Learning. Proceedings of the 23rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 787–795. [论文](https://arxiv.org/abs/1611.07012)
