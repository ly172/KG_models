# [27] GAMENet

用公开 MIMIC-IV Demo 的住院记录运行作者模型。输入患者的就诊顺序、诊断、操作和既往用药，输出当前就诊各药物组的推荐分数。当前例子保留 23 名患者、68 次就诊。

## 运行

```sh
pixi run --locked smoke
```

首次运行会下载公开数据。结果和日志在 `outputs_demo/`，汇总见 `outputs_demo/result.json`。人工病史例子使用下面的命令，结果在 `outputs/`。

```sh
pixi run --locked smoke-synthetic
```

## 输入

患者记录来自 [MIMIC-IV Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/)。诊断和操作匹配作者 ICD-9 词表，处方经作者的 NDC、RxNorm、ATC 对照表转换成四字符药物组。词表、药物共现图和药物相互作用图均来自 [GAMENet 作者仓库](https://github.com/sjy1203/GAMENet)。

这里把整次住院中匹配到的药物合成训练答案。作者的 [原预处理](https://github.com/sjy1203/GAMENet/blob/da695b4fc9390882f3a681c82115e81291ae6380/data/EDA.ipynb) 为每次 ICU 住院保留最早 `STARTDATE` 对应的处方。因此本例预测整次住院的药物集合，与作者的用药集合定义有差异。MIMIC-IV 处方表按住院号记录，本入口按该住院号汇总。

人工例子从同一词表抽取 36 名患者，每人两至四次就诊，每次四个诊断、两个操作、五个药物，随机种子为 20261029。

## 代码和改动

作者模型与训练程序在 `upstream/code/`，原词表和药物图在 `upstream/data/`。运行副本去掉两个失效且未使用的导入，改为 CPU，并允许设置训练轮数；旧词表转换后逐项核对编号。

手动添加的文件：

- `public_demo.py`：筛选公开病史、匹配编号并按住院时间排序。
- `simulate_inputs.py`：生成人工病史。
- `read_legacy_vocabulary.py`：读取旧词表并核对双向编号。
- `prepare_author.py`：生成运行副本和修改记录。
- `reproduce.py`：调用作者训练与测试，汇总结果。
- `verify_compatibility.py`：检查导入、词表和运行副本。

`public_tables.json` 保存映射表来源，已完成的运行记录在 `verification/`。

原文：[27] Shang, J., Xiao, C., Ma, T., Li, H., Sun, J. (2019). GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination. Proceedings of the AAAI Conference on Artificial Intelligence, 33, 1126–1133. [论文](https://arxiv.org/abs/1809.01852)
