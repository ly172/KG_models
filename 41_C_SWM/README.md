# [41] C-SWM

C-SWM 输入一张有五个物体的画面、一个移动动作，以及动作后的画面。模型从图像中提取五份物体表示，让物体交换信息，再结合动作预测下一状态。训练把预测状态拉近真实下一状态，并拉远打乱配对的状态。

画面和动作由作者的 Shapes 环境生成。每幅图为 3×50×50，每个物体用两个数字表示；边界、碰撞和移动规则由原环境处理。输出是物体的下一状态表示，评估检查真实下一状态在候选中的排序。

## 运行

在本目录执行：

```sh
pixi run --locked smoke
```

默认生成 20 局训练数据和 100 局评估数据，训练三轮，共 48 次更新，再做 1、5、10 步预测。结果在 `outputs/quick/result.json`，参数在 `outputs/quick/checkpoints/shapes/model.pt`。完整规模命令为 `pixi run --locked paper-scale`。

小测试每批 128 条，作者完整配置为 1,024 条。原损失从同批样本选对照状态，因此这个调整也缩小了对照范围。评估候选由 10,000 个减为 100 个，指标对应当前规模。

## 代码与改动

代码来自 [tkipf/c-swm](https://github.com/tkipf/c-swm/tree/e944b24bcaa42d9ee847f30163437a50f0237aa0)。`upstream/modules.py` 定义模型和损失，`train.py`、`eval.py` 负责训练与评估，`data_gen/` 和 `envs/` 生成画面，`utils.py` 读取数据。

| 手动添加的文件 | 用途 |
| --- | --- |
| `prepare.py` | 生成兼容副本 |
| `run_author.py` | 设置线程和路径，启动作者脚本 |
| `reproduce.py` | 调用数据生成、训练和评估，保存结果 |
| `verify_compatibility.py` | 比较圆形绘制、动作编号和迭代器接口 |

兼容副本将旧绘图接口 `circle` 换成 `disk`，将迭代器 `.next()` 换成当前写法，并把动作索引转成 64 位整数。Gym 返回的附加信息改为空字典，数据读取放到当前进程，解决 Windows 下重复启动的问题。具体差异在 `compatibility.patch`，当前小测试记录在 [运行结果](verification/smoke.json)。

原文：[41] Kipf, T., van der Pol, E., Welling, M. (2020). Contrastive Learning of Structured World Models. International Conference on Learning Representations. [论文](https://arxiv.org/abs/1911.12247)
