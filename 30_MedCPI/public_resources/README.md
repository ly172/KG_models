# 公开术语资料

这里的程序均为手动添加，用公开文件准备作者所需的概念名称、医学类型和药品编号。MedCPI 默认入口会自动调用它；单独准备可在 MedCPI 目录执行：

```sh
pixi run --locked python public_resources/prepare_public_resources.py --output .assets/public_resources
```

首次下载约 940 MB，解压和索引需要额外空间。已有资料可加 `--cache "目录"` 复用。输出来源记录为 `public_resource_receipt.json`。

## 来源

| 资料 | 用途 |
| --- | --- |
| [GraphCare 公开医学文件](https://github.com/pat-jj/GraphCare/tree/fe92aa67add80b62d4b5108d570b6483ee69a5d4/KG_mapping) | 提供部分 UMLS 概念、关系、ICD-9/ATC 对应及 RxNorm 到 UMLS 的对应。 |
| [NLM RxNorm 可开方子集](https://www.nlm.nih.gov/research/umls/rxnorm/docs/prescribe.html) | 使用 2014-12-01、2019-12-02、2026-10-05 三个版本，取得药品名称、NDC 对应和关系；该子集可公开下载。 |
| [SciSpaCy UMLS 2022AB 子集](https://github.com/allenai/scispacy/blob/main/scispacy/linking_utils.py) | 提供概念名称、别名和原语义类型。 |
| [NLM SemGroups](https://www.nlm.nih.gov/research/umls/knowledge_sources/semantic_network/index.html) | 将医学类型归入疾病、操作、药物等类别。 |

各资料沿用原发布方的使用条件。下载地址保存在 `public_sources.json`。

## 文件

| 文件 | 作用 |
| --- | --- |
| `prepare_public_resources.py` | 下载资料并调用其余整理程序。 |
| `extract_public_terminology.py` | 从作者 RDF 读取明确记录的编码、概念和类型。 |
| `prepare_scispacy_types.py` | 提取所需概念的名称、别名和原类型。 |
| `prepare_rxnorm.py` | 整理 2026 年药物名称、NDC 和关系。 |
| `prepare_historical_rxnorm.py` | 下载、解压并检查 2014 年和 2019 年资料。 |
| `build_historical_rxnorm_indices.py` | 分版本建立 NDC 对应索引，记录多个目标及版本差异。 |

NDC 按字符串保存前导零，匹配使用官方 RXNORM 来源中未抑制的精确 11 位编号。一个编号对应多个目标时，全部目标交给后续映射程序处理。
