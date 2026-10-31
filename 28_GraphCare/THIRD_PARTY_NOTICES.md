# 第三方来源

GraphCare 代码固定到提交 `fe92aa67add80b62d4b5108d570b6483ee69a5d4`，MIT 许可保存在 `upstream/LICENSE`。`graphcare.py` 中写死的日志凭据已清空。

`upstream_extra/pyhealth_data.py`、`pyhealth_mimic4.py` 和 `pyhealth_cross_map.py` 来自 PyHealth 1.1.6，采用 MIT 许可，版权为 Copyright (c) 2024 SunLab。[原许可](https://github.com/sunlabuiuc/PyHealth/blob/v1.1.6/LICENSE)及完整许可文字随代码保存。

GTE 权重与实现来自 [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct/tree/a9af15a6372d7d6b25e9fb07c2ccb9e1fe645644)，模型页标明 Apache 2.0 许可，固定文件记录在 `public_umls/assets.json`。SciSpaCy 派生对应表的来源及整理规则见 `derived_resources.json`，医学术语沿用各来源词表的使用条件。

默认病史为 [MIMIC-IV Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/)，采用 Open Data Commons Open Database License v1.0。完整 MIMIC-IV 按 [PhysioNet](https://physionet.org/content/mimiciv/3.1/) 要求申请访问。运行生成的病史、逐人预测和模型保存在 Git 忽略的 `.private/`。
