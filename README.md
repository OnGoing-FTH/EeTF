# EeTF

工业场景下的基于分块 Transformer 的边缘检测模型。

## 项目目录

```text
EeTF/
├── data/                         # 数据处理
│   ├── datasets/                 # 数据集定义与数据载入器
│   └── transforms/               # 数据增强与预处理
├── patching/                     # 图像切块与 Patch 拼接
├── models/                       # Transformer 模型
├── losses/                       # 损失函数
├── metrics/                      # 评价指标与可视化
├── engine/                       # 训练器、验证器与检查点管理
├── utils/                        # 日志、随机种子等通用工具
├── tests/                        # 单元测试与流程测试
├── checkpoints/                  # 模型权重
├── outputs/                      # 预测结果与可视化输出
└── logs/                         # 训练日志
```

## 模块职责

| 目录 | 职责 |
|---|---|
| `data/datasets` | 读取图像和边缘标签，构造训练集、验证集和测试集 |
| `data/transforms` | 图像归一化、裁剪、翻转、旋转等预处理与增强 |
| `patching` | 将大图切成固定大小 Patch，并将预测结果拼接回原图 |
| `models` | 实现 Patch Embedding、Transformer Encoder、Decoder 和边缘预测头 |
| `losses` | 实现 BCE、Dice、Focal 及组合损失 |
| `metrics` | 计算 Precision、Recall、F1、IoU 等指标 |
| `engine` | 管理训练、验证、测试、优化器、学习率调度和模型保存 |
| `utils` | 提供日志、配置读取、随机种子等通用功能 |
| `tests` | 验证数据、切块、模型、损失和训练流程 |

## 推荐数据流

```text
图像/边缘标签
    -> Dataset/DataLoader
    -> 数据增强与归一化
    -> Image Patcher
    -> Patch Embedding
    -> Transformer Encoder
    -> Decoder
    -> 边缘预测图
    -> Patch Merge
    -> Loss 与 Metrics
```
