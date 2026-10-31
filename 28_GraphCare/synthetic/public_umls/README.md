# [28] GraphCare：公开医学图和人工病史

从作者公开 UMLS 关系中取一张小图，用本地 GTE 编码概念名称，再运行作者 GraphCare 网络。输出是每名人工患者的二分类分数。

## 运行

```sh
pixi run --locked smoke
```

首次下载约 7.17 GB，建议准备 16 GB 可用内存。小图、名称向量和结果写入 `outputs/`。

两个 GraphCare 编码入口共用仓库 `.run/resources/graphcare/` 下的缓存，按编码器和图版本保存。已有文件也可以指定目录：

```sh
pixi run --locked python reproduce.py --asset-cache "目录"
```

## 输入

程序从糖尿病概念 C0011849 出发，按编号取 31 个邻居，保留它们之间的原方向关系，得到 32 个概念、10 种关系和 109 条边。42 个概念及关系名称由 GTE 编码，取最后一个有效文字位置的输出并归一化，每个名称得到 1,536 个数。

固定种子生成 640 名人工患者，每人一至三次就诊，每次包含起始概念和三个随机邻居。标签交替取 0、1，训练、验证、测试分别为 512、64、64 人。作者子图函数把病史接入图，再交给三层 BAT。

医学关系来自 [GraphCare 的 UMLS 子集](https://github.com/pat-jj/GraphCare/tree/fe92aa67add80b62d4b5108d570b6483ee69a5d4/KG_mapping/umls)，向量来自 [GTE-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)。这些是替代知识输入；论文使用 GPT 图、UMLS 采样和 OpenAI 向量。

## 代码和改动

训练调用上级目录的作者 GraphCare 模型和函数。`alpha`、`beta` 按作者实际入口设为 False；论文 BAT 包含这两项关注计算。编码器直接加载 Qwen2Model 类，使用原 CPU 注意力实现，解决通用加载器对可选 GPU 依赖的检查错误。

手动添加的文件：

- `fetch_assets.py`：下载 `assets.json` 中的资源，共用版本缓存。
- `public_inputs.py`：选取公开子图并生成人工就诊。
- `encode_names.py`：用公开 GTE 模型计算名称向量。
- `reproduce.py`：调用作者训练、测试及保存后重载。
- `verify_loading.py`：检查编码器加载过程。

已完成的结果见 `verification/smoke.json`。
