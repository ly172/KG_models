# 公开数据下载

`mimic_demo.py` 下载 PhysioNet 的 [MIMIC-IV Clinical Database Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/)，用于医学模型的小规模运行。官方演示包含 100 名患者，约 16 MB 压缩包，可直接取得。

各模型调用 `ensure_demo()` 后得到数据根目录，住院数据位于其中的 `hosp/`。默认缓存是仓库根目录 `.run/mimic_demo_2_2/`。`ensure_demo(output=..., cache=...)` 可以指定输出位置或已有压缩包。版本、来源和许可写入缓存中的 `dataset.json`。

`mimic_demo_files.json` 保存官方文件清单。数据采用 Open Data Commons Open Database License v1.0，完整许可随官方下载包保存。

`downloads.py` 提供各模型共用的下载函数。中断时尝试续传，`cache` 参数可指向已经下载的文件。这些脚本为手动添加，模型计算在各论文目录中。
