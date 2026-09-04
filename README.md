# EeTF

基于 PyTorch 的工业场景稀疏边缘像素分割项目。模型将输入图像分为 `64 x 32` 的无重叠 Patch，使用 CNN、块级统计特征、MLP、2D-RoPE Attention 和 DynamicViT 路由器选择重点块，并通过 selected/remaining 双分支恢复完整像素级边缘掩码。

## 环境

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
cd /home/fth/EdTF/EeTF
```

主要依赖：PyTorch、torchvision、Pillow、NumPy、OpenCV。

## 目录

```text
EeTF/
├── data/
│   ├── images/                  # 完整原始图像数据集
│   ├── edge_maps/               # 与 images 同名的边缘标签
│   ├── datasets/
│   │   ├── edge_dataset.py      # 图像/标签配对、读取和 Dataset
│   │   └── split.py             # 固定随机种子的训练/验证划分
│   └── transforms/
│       ├── edge_augment.py      # 训练期同步几何与图像专属增强
│       └── letterbox.py         # 验证/推理期等比例缩放、填充与还原
├── patching/                    # 64 x 32 Patch 切分
├── models/                      # CNN、MLP、融合、RoPE、路由和双分支解码
├── losses/                      # 稀疏像素分割损失和动态保留率损失
├── metrics/                     # 前景主导的分割指标
├── engine/                      # 单 epoch 训练、验证和推理逻辑
├── utils/                       # 标签读取、Patch 标签池化和统计特征
├── checkpoints/                 # latest.pt 和 best.pt
├── outputs/                     # 概率图、掩码、叠加图和多图拼图
├── train.py                     # 训练命令行入口
└── infer.py                     # 推理和可视化命令行入口
```

## 数据集格式

完整数据集直接存放于：

```text
data/images/
data/edge_maps/
```

图像和标签通过相同文件名 stem 严格配对。例如：

```text
data/images/1787823737128.png
data/edge_maps/1787823737128.png
```

支持 `.bmp`、`.jpg`、`.jpeg`、`.png`、`.tif`、`.tiff` 和 `.webp`。标签读取为单通道 `8-bit` 灰度图，归一化到 `[0, 1]`；因此可保留渐变线条强度。Patch 路由监督使用 `label > 0` 的硬占用标签。

Dataset 会在构建时检查以下问题：

- 图像存在但没有同名标签；
- 标签存在但没有同名图像；
- 图像和标签原始分辨率不一致；
- 目录中没有可用样本。

## 划分和加载顺序

训练集、验证集从完整配对样本中划分，而不是从图像和标签目录分别划分：

```text
1. 按文件名 stem 配对图像与标签
2. 使用 seed 打乱配对样本
3. 按 val_ratio 划分训练集和验证集
4. 训练 DataLoader 每个 epoch shuffle=True
5. 验证 DataLoader shuffle=False
```

默认参数：

```text
val_ratio = 0.2
seed = 42
batch_size = 1
```

`B=1` 是当前模型约束，因为每个样本可能经过增强后落在不同目标分辨率，且 DynamicViT 路由按单样本 Patch 数量选择 Token。相同 `seed` 和相同文件集合会得到相同的训练/验证划分。

## 预处理和增强

支持的模型输入目标分辨率：

```text
768 x 768
512 x 1024
1024 x 512
```

这些尺寸可被 Patch 高度 `64` 和宽度 `32` 整除。

训练阶段：

- 图像和标签同步执行水平翻转、垂直翻转、旋转、缩放和 Shear；
- 图像专属执行 CLAHE、锐化、模糊、噪声和局部光照变化；
- 图像几何变换使用双线性插值；
- 标签几何变换始终使用最近邻插值；
- 最后执行保持长宽比的 Letterbox，填充为背景 `0`。

验证和推理阶段：不使用随机增强，仅选择与原图长宽比最接近的目标分辨率并执行确定性 Letterbox。推理结果会移除填充区域并恢复到原图大小。

## 模型数据流

```text
[B, 3, H, W]
    -> ImagePatchingRect: [B, N, 3, 64, 32]
    -> CNNBase: [B, N, 8192]
    -> BlockFeatureExtractor + MLPBase: [B, N, 768]
    -> FeatureFusion: [B, N, 768]
    -> TokenSelector (DynamicViT)
       -> selected Patch 分支: 原始 Patch + 局部解码
       -> remaining 特征分支: 768 -> 512 -> 32 x 16 + 超分解码
    -> 按原 Patch 索引回填与合并
    -> mask_logits: [B, 1, H, W]
```

关键配置：

```text
Patch size: 64 x 32
Token dimension: 768
Attention heads: 12
Head dimension: 64
2D-RoPE: 前 32 维使用 x，后 32 维使用 y
Remaining branch: 768 -> 512 -> 32 x 16
```

## 训练

使用默认数据目录训练：

```bash
python train.py \
  --epochs 100 \
  --keep-ratio 0.3 \
  --learning-rate 1e-4 \
  --val-ratio 0.2 \
  --seed 42
```

显式指定数据位置：

```bash
python train.py \
  --data-root /home/fth/EdTF/EeTF/data \
  --image-dir images \
  --mask-dir edge_maps \
  --epochs 100 \
  --keep-ratio 0.3 \
  --checkpoint-dir checkpoints
```

常用参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--epochs` | `100` | 训练轮数 |
| `--learning-rate` | `1e-4` | AdamW 学习率 |
| `--weight-decay` | `1e-4` | AdamW 权重衰减 |
| `--keep-ratio` | `0.3` | DynamicViT 保留的 Patch 比例 |
| `--val-ratio` | `0.2` | 验证集比例 |
| `--seed` | `42` | 数据划分和 DataLoader 随机种子 |
| `--ratio-weight` | `2.0` | Token 保留率约束权重 |
| `--router-weight` | `1.0` | Patch 路由监督权重 |
| `--resume` | 无 | 从 `latest.pt` 或其他 checkpoint 恢复 |

每轮训练执行：

```text
SparseEdgeLoss (Weighted BCE + Dice)
+ Patch Keep/Drop 类别均衡路由损失
+ DynamicViT 目标保留率约束
```

checkpoint 输出：

```text
checkpoints/latest.pt    # 最近一个 epoch 的可恢复状态
checkpoints/best.pt      # 验证 foreground_f1 最高的模型
```

## 验证指标

稀疏线条分割中大量背景像素会使普通 Accuracy 虚高。因此选择最佳模型时使用：

```text
foreground_f1
```

同时记录：

```text
foreground_precision
foreground_recall
foreground_iou
dice
balanced_accuracy
weighted_accuracy
raw_accuracy            # 仅作参考，不作为主指标
```

`weighted_accuracy` 会降低真负背景像素的贡献，默认背景权重为 `0.05`。

## 推理和可视化

单张图片推理：

```bash
python infer.py \
  --input data/images/example.png \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs \
  --threshold 0.5
```

对整个图片目录推理：

```bash
python infer.py \
  --input data/images \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs \
  --threshold 0.5
```

每张图片的输出：

```text
outputs/<name>_prob.png      # 8-bit 灰度预测概率图
outputs/<name>_mask.png      # 按阈值生成的二值掩码
outputs/<name>_overlay.png   # 红色半透明掩码叠加到原图
```

目录推理默认还会生成：

```text
outputs/contact_sheet.png
```

拼图每行展示一个样本：

```text
原图 | 掩码叠加图
```

调整拼图缩略图尺寸：

```bash
python infer.py \
  --input data/images \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs \
  --tile-width 400 \
  --tile-height 300
```

不生成拼图：

```bash
python infer.py \
  --input data/images \
  --checkpoint checkpoints/best.pt \
  --output-dir outputs \
  --no-contact-sheet
```

## 已验证内容

已完成以下基础流程验证：

```text
Patch 切分与完整掩码回填
selected/remaining 索引互补性
模型前向与反向传播
SparseEdgeLoss 反向传播
灰度标签读取与 Patch 标签池化
训练增强和 Letterbox 标签对齐
完整数据集固定随机划分
训练单 epoch、验证指标和单图推理
概率图、二值图、叠加图及 contact sheet 保存
```

## 已知限制

- 当前主流程只支持 `batch_size=1`。
- `BlockFeatureExtractor` 当前输出 4 维统计量，主模型零填充到 MLP 所需的 28 维；后续可扩展为完整 28 维统计特征。
- DynamicViT 的动态 `torch.topk` 在部分 ONNX Runtime 版本中对动态 K 的支持有限；部署时可改为固定最大 Token 数并配合有效 Mask，或将 K 作为显式输入。
