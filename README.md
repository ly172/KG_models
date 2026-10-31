# KG models

收录 13 篇文章的可执行程序，目录编号对应《WM_Medication_KG_v2》的引用号。每个模型文件夹都写明输入输出、数据来源、运行方法和具体改动。完整篇目见 [PAPERS.md](PAPERS.md)。

## 运行
例如gram：
执行：（需要git和pixi）

```sh
git clone https://github.com/ly172/KG_models.git
cd KG_models
pixi run --locked smoke --model gram
```

这条命令自动下载公开 MIMIC-IV Demo，完成 GRAM 的数据整理、训练和测试。换模型时替换下面表格中的入口名称，比如说gram换成gamenet；`pixi run --locked list` 可列出全部输入例子。也可以进入模型目录，运行 `pixi run --locked smoke`。

## 模型

| 引用 | 模型 | 入口名称 |
| --- | --- | --- |
| [14] | [GRAM](14_GRAM/README.md) | `gram` |
| [27] | [GAMENet](27_GAMENet/README.md) | `gamenet` |
| [28] | [GraphCare](28_GraphCare/README.md) | `graphcare` |
| [29] | [KAT-GNN](29_KAT_GNN/README.md) | `kat-gnn` |
| [30] | [MedCPI](30_MedCPI/README.md) | `medcpi` |
| [36] | [GreaseLM](36_GreaseLM/README.md) | `greaselm` |
| [37] | [Graph Neural Prompting](37_Graph_Neural_Prompting/README.md) | `gnp` |
| [41] | [C-SWM](41_C_SWM/README.md) | `c-swm` |
| [46] | [Malec PSI检索组件](46_Malec_Confounders/README.md) | `malec-psi` |
| [47] | [CausalKG](47_CausalKG/README.md) | `causalkg` |
| [48] | [CaML](48_CaML/README.md) | `caml` |
| [49] | [KG-TREAT](49_KG_TREAT/README.md) | `kg-treat` |
| [67] | [R-GCN 2018](67_R_GCN_Original/README.md) | `rgcn-2018` |


`run_models.py` 和 `shared/` 为手动添加，分别负责统一启动和公开资料下载。`paper_status.json` 与 `verification/smoke_runs.json` 保存篇目和运行记录。
