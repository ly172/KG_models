# [46] Malec confounder selection

本目录运行论文的知识图检索部分。输入是“起点、关系、终点”三列关系表，原 SemanticVectors 程序据此建立概念、关系和语义表示，再按作者表达式查询候选共同原因。

例子生成 12 条人工关系，名称以 `syn_` 开头。其中两个概念同时满足“被药物治疗”和“导致目标结局”，另加入只符合一条关系的概念。原 PSI 使用论文的 32,000 位二进制表示及 IDF 权重，查询为：

```text
S(syn_druga)*P(TREATS)+S(syn_outcome)*P(CAUSES-INV)
```

S 是概念的语义表示，P 是关系表示。两项分别寻找药物治疗的对象和结局的原因，合在一起给候选概念评分。输出是五个候选及分数，测试检查前两名是否找回人工设置的共同原因。

## 运行

在本目录执行：

```sh
pixi run --locked smoke
```

程序自动取得 Java、Maven 和原源码，编译后建索引并查询。结果在 `outputs/result.json`，输入在 `outputs/triples.tsv`，日志在同一目录。

## 代码与改动

查询来自 [causalSemantics](https://github.com/kingfish777/causalSemantics/tree/b3d4e90fd318a828fe61dac4d7f536dd17ad7512) 的 `getConfounders.sh`。索引和查询调用 [SemanticVectors 5.9](https://github.com/semanticvectors/semanticvectors/tree/f585f0a66973b63cc6c0952e3fa9610825f285a3) 的 `LuceneIndexFromTriples`、`PSI` 和 `Search`。

`reproduce.py` 为手动添加，下载并编译原库、生成关系表、调用原入口；`sources.json` 记录固定来源。完成的是索引与查询组件。整篇临床分析仍需 SemMedDB 筛选结果、病历概念共现矩阵及因果分析变量。当前组件结果见 [运行记录](verification/result.json)。

原文：[46] Malec, S. A., Wei, P., Bernstam, E. V., Boyce, R. D., Cohen, T. (2021). Using computable knowledge mined from the literature to elucidate confounders for EHR-based pharmacovigilance. Journal of Biomedical Informatics, 117, 103719. [论文](https://doi.org/10.1016/j.jbi.2021.103719)
