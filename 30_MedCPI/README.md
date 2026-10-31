# [30] MedCPI

MedCPI 把住院病史对应到医学概念，为患者建立知识图，再结合病史和图预测结局。默认例子使用公开 MIMIC-IV Demo，任务是预测下一次住院是否死亡。

## 怎么跑

在本目录执行：

```sh
pixi run --locked smoke
```

程序自动取得 [Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/)、公开医学资料和作者使用的 SapBERT 权重，完成概念映射、构图及一轮训练，然后验证、测试并重新加载模型。首次约下载 1.4 GB 资料与权重，运行使用 CPU。

结果在 `smoke_result.json` 和 `verification/`，模型位于 `.private/outputs/integrate/mimic4/mortality/best_valid_formal_report_fixed_seed0.pt`，日志在 `.private/logs/`。资源保存在 `.assets/`，患者输入及派生文件在 `.private/`。

当前例子有 15 名患者、37 次住院，形成训练 19 个、验证 1 个、测试 2 个实例。验证集只有一种标签，作者程序会改用验证代价选择模型。保存后重载的测试结果相同。

## 输入和输出

病史来自住院、诊断、操作、处方及两张 ICD 名称表。程序选择有 2 至 4 次完整住院、诊断和操作均使用 ICD-9 的患者，按时间保留完整病史。目标住院的死亡结局单独作为标签。

医学概念和关系来自 GraphCare 公开 UMLS 文件、SciSpaCy 的概念名称与类型，以及 NLM 的语义类型表。处方中的 NDC 编号先用 2014、2019、2026 年固定版本的公开 RxNorm 对应到药物名称，再连接 UMLS 概念。SapBERT 把病史名称与候选名称转成数字，供作者映射器选择概念。具体文件见 [公开术语资料](public_resources/README.md)。

作者映射器的全部输出原样进入构图，与公开编号对应不一致的选择另存记录。Construct 从训练病史出发抽取任务关系，Personalize 为各个预测实例建立个人图。Integrate 分别读取病史与图，再用病史给图节点分配权重，汇总图信息并与病史拼接，输出死亡分数。

死亡任务的关系范围直接使用作者已发布的 `relation_schema.json`。公开术语和关系覆盖完整知识库的一部分；缺少概念编号的事件保留在病史文件中，作者汇总概念时跳过这些事件。

## 本地数据

```sh
pixi run --locked reproduce --mimic-root "D:/my-mimic-iv" --patients 256 --epochs 2
```

需要 `hosp/admissions`、`diagnoses_icd`、`procedures_icd`、`prescriptions`、`d_icd_diagnoses` 和 `d_icd_procedures`，支持 CSV 及压缩 CSV。至少需 10 名符合筛选条件的患者。已有文件可用 `--public-cache "医学资料目录"` 和 `--sapbert-cache "SapBERT目录"` 指定。

## 文件

`author/scripts/` 来自 [作者固定提交](https://github.com/hw146/MedCPI/tree/9413431d8741dfb875d12a2a8d516c94643d9748)，运行时自动下载。其中 `sapbert_map_mentions.py` 负责概念映射，`build_concept_mkg.py` 和 `build_patient_pkg.py` 构图，`integrate_model.py` 定义预测网络，`train_integrate_formal.py` 完成训练、选模和测试。

| 文件 | 来源和用途 |
| --- | --- |
| `smoke.py`、`reproduce.py` | 手动添加。准备公开或本地病史，调用作者各阶段并汇总结果。 |
| `fetch_author.py`、`fetch_sapbert.py` | 手动添加。取得固定版本的作者源码和 SapBERT。 |
| `prepare_cohort.py` | 手动添加。选择完整病史，写出作者程序所需的表。 |
| `prepare_public_inputs.py` | 手动添加。把公开编码、名称、类型和关系整理成作者输入。 |
| `run_original_mapper.py` | 手动添加。调用作者映射器，原样传递结果，另存来源差异。 |
| `run_author_stages.py` | 手动添加。设置工作目录并依次调用作者程序。 |
| `train.py` | 手动添加。修复训练末尾的报告字段，比较修复前后的参数及重载结果。 |
| `verify_pipeline.py`、`clean_copy_check.py` | 手动添加。检查患者划分、概念输入，并在新目录重做流程比较结果。 |
| `public_resources/*.py` | 手动添加。下载和整理公开术语，每个文件的作用见[该目录说明](public_resources/README.md)。 |

## 改动

作者训练函数在训练和测试结束后引用未定义的 `seed_results`，写报告时报错。运行副本删除两行重复参数计数，报告继续使用前面已有的正确字段。默认入口分别运行原版和修复版，保存的参数逐项相同；差异见 `verification/trainer_report_only.patch`。

作者映射器的输出直接用于构图。每次映射前生成当前队列的候选文字缓存；名称规范字段由作者函数生成。

原文：[30] Wang, H., Dong, H., Liu, L. (2026). MedCPI: A Construct-Personalize-Integrate Framework for KG-enhanced Clinical Prediction. Findings of the Association for Computational Linguistics: ACL 2026, 24266–24282. [论文](https://github.com/hw146/MedCPI)
