# 公开医学关系和名称向量

[作者公开的 UMLS 子集](https://github.com/pat-jj/GraphCare/tree/fe92aa67add80b62d4b5108d570b6483ee69a5d4/KG_mapping/umls) 包含 297,927 个概念、98 种关系和 1,212,586 行关系记录。`concepts.txt` 保存概念编号，`concept_names.txt` 保存名称，`relations.txt` 保存关系名称，`umls.csv` 保存起点、关系、终点及权重。

[公开小图例子](public_umls/README.md) 从这些记录选出 32 个概念和 109 条边，再用 [GTE-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct) 编码概念和关系名称，每个名称得到 1,536 个数。患者病史和标签由程序生成。

作者原概念向量未随仓库提供。这里使用公开 UMLS 子集和 GTE 名称向量，输入与论文的 GPT 图、UMLS 采样及 OpenAI 向量流程有差异。文件地址和版本在 `public_umls/assets.json`。
