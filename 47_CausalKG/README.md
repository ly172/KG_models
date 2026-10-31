# [47] CausalKG

用人工表运行作者的 R 中介分析程序。输入包括适应证、用药、结局和共同影响因素，输出通过用药产生的间接影响、直接影响、总影响及交互项检验结果。

## 运行

请将仓库放在英文路径中，再执行：

```sh
pixi run --locked smoke
```

结果在 `run/result.json` 和 `run/result.csv`，人工表在 `run/artificial.csv`。

## 输入

固定种子 20261029 生成 2,048 行二元数据。`IND` 表示适应证，`Drug` 表示用药，`SE` 表示结局，`C1` 至 `C4` 表示共同影响因素。四个因素独立抽取，取 1 的概率分别为 45%、35%、50%、30%。适应证概率随 C1、C2 增加，随 C3 减少；用药概率随 IND、C1、C4 增加；结局概率随 IND、Drug、C2、C3 增加。具体概率公式见 `reproduce.py`。

程序先筛选变量，再估计 `IND` 通过 `Drug` 影响 `SE` 的部分。这里运行论文的统计阶段；临床分析还需按知识图定义疾病、药物、结局和患者队列。

## 代码和改动

原 R 程序来自 [CausalKG 作者仓库](https://github.com/bio-ontology-research-group/Mediation-Analysis-using-Causal-Knowledge-Graph)，自动下载到 `run/mediation_analysis.R`，来源在 `source.json`。

作者脚本有两项影响估计的行为：第二轮 LASSO 以 Drug 为响应；交互项未达阈值时虽另拟合无交互模型，最终 `mediate` 仍接收含交互的模型。当前运行执行这份原脚本。

手动添加的文件：

- `reproduce.py`：生成人工表，下载作者程序，汇总结果。
- `run_author.R`：设置随机种子，执行作者原脚本并保存输出。

原文：[47] Toonsi, S., Schofield, P. N., Hoehndorf, R. (2026). Causal knowledge graph analysis identifies adverse drug effects. Bioinformatics, 42(1), btaf661. Published online 12 December 2025. [论文](https://doi.org/10.1093/bioinformatics/btaf661)
