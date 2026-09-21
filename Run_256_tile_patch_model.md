# EeTF 256×256 Tile 训练、推理与部署命令

## 1. 当前处理规格

```text
大图输入
  → 保持原始长宽比
  → 高度和宽度 Letterbox 到 256 的整数倍
  → 切分为多个无重叠 256×256 Tile
  → 每个 Tile 切分为 16×16 网格
  → 共 256 个 16×16 Patch
  → 第一阶段执行 Patch Router 选择
  → 第二阶段直接对 16×16 Patch 精细分割
  → 输出 256×256 Tile 掩码
  → 按 Tile 坐标回拼大图
  → 去除 Letterbox Padding
  → 恢复原始图像尺寸
```

固定模型规格：

```text
Tile：256×256
Patch：16×16
Patch Grid：16×16
Patch 数量：256
模型输入：[B,3,256,256]
模型输出：[B,1,256,256]
Checkpoint format_version：6
```

---

## 2. 激活运行环境

所有命令都在项目目录执行：

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
cd /home/fth/EdTF/EeTF
```

检查 CUDA 是否可见：

```bash
nvidia-smi
```

训练和推理程序会自动选择设备：

```text
CUDA 可用 → 使用 cuda
CUDA 不可用 → 使用 cpu
```

---

## 3. 原始数据整理

原始数据根目录：

```text
/home/fth/EdTF/DATA
```

程序会递归查找各级子目录中的 Labelme JSON，并优先使用 JSON 的 `imagePath` 配对图片；如果 `imagePath` 不可用，则使用同目录、同文件名主干的图片。

建议先执行只读预览。该命令只统计配对和计划生成的 Tile 数量，不写入数据：

```bash
python prepare_teed_dataset.py \
  --src /home/fth/EdTF/DATA \
  --out /home/fth/EdTF/EeTF/data_256 \
  --tile-size 256 \
  --seed 2026 \
  --dry-run
```

确认预览无误后，正式生成完整数据集：

```bash
python prepare_teed_dataset.py \
  --src /home/fth/EdTF/DATA \
  --out /home/fth/EdTF/EeTF/data_256 \
  --tile-size 256 \
  --seed 2026 \
  --line-width 1 \
  --soft-labels \
  --soft-sigma 0.5 \
  --soft-radius 1.0
```

如果输出目录已经存在，并确认需要整体替换：

```bash
python prepare_teed_dataset.py \
  --src /home/fth/EdTF/DATA \
  --out /home/fth/EdTF/EeTF/data_256 \
  --tile-size 256 \
  --seed 2026 \
  --line-width 1 \
  --soft-labels \
  --soft-sigma 0.5 \
  --soft-radius 1.0 \
  --overwrite
```

如需生成硬边缘二值标签而不是高斯软标签：

```bash
python prepare_teed_dataset.py \
  --src /home/fth/EdTF/DATA \
  --out /home/fth/EdTF/EeTF/data_256_binary \
  --tile-size 256 \
  --seed 2026 \
  --line-width 1 \
  --no-soft-labels
```

整理规则：

```text
1. 原始图片和 JSON 不移动、不改名、不修改。
2. 标签先在原图尺寸上根据 Labelme shapes 栅格化。
3. 支持 polygon、line、linestrip、polyline、rectangle、circle 和 point。
4. 图像和标签不拉伸、不缩放。
5. 宽度或高度不足 256 整数倍时，仅在右侧和底部补零。
6. 图像和标签使用完全相同的坐标切成无重叠 256×256 Tile。
7. 小于 256×256 的图片补零后形成一个 256×256 Tile。
8. 所有来源的 Tile 使用固定 seed 全局打乱。
9. 输出名称从 000001 开始连续编号。
10. 图片和标签始终使用相同编号。
```

生成目录：

```text
data_256/
├── images/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
├── edge_maps/
│   ├── 000001.png
│   ├── 000002.png
│   └── ...
├── manifest.json
├── pairs.json
└── summary.json
```

文件说明：

```text
manifest.json：每个新编号对应的原图、JSON、Tile 行列、原图坐标和有效区域
pairs.json：按编号排列的 image/label 相对路径对
summary.json：源图数量、孤立图片数量、Tile 数量和前景/背景 Tile 数量
```

`manifest.json` 会保留以下关键追溯信息：

```text
新样本编号
源图片相对路径
源 JSON 相对路径
源图宽高
补齐后宽高
Tile row / col
Tile top / left
有效区域 valid_width / valid_height
该 Tile 是否包含边缘
```

正式训练时使用新数据目录：

```bash
python train.py \
  --data-root data_256 \
  --image-dir images \
  --mask-dir edge_maps \
  --tile-size 256 \
  --patch-size 16 \
  --epochs 100 \
  --routing-epochs 45 \
  --frozen-epochs 35 \
  --batch-size 8 \
  --val-batch-size 8 \
  --num-workers 4 \
  --val-ratio 0.2 \
  --seed 26 \
  --run-dir runs/train
```

> 注意：整理脚本中的 `--seed` 控制 Tile 新编号的打乱顺序；训练命令中的 `--seed` 控制训练/验证划分和训练随机性，两者互不替代。

---

## 4. 数据目录

默认数据结构：

```text
EeTF/
└── data/
    ├── images/
    │   ├── 0001.jpg
    │   ├── 0002.jpg
    │   └── ...
    └── edge_maps/
        ├── 0001.png
        ├── 0002.png
        └── ...
```

图像与标签必须使用相同的文件名主干，例如：

```text
data/images/0001.jpg
data/edge_maps/0001.png
```

程序先按照完整原图划分训练集和验证集，然后分别展开成 `256×256` Tile。同一张原图的 Tile 不会同时进入训练集和验证集。

---

## 5. 启动完整训练

推荐训练命令：

```bash
python train.py \
  --data-root data \
  --image-dir images \
  --mask-dir edge_maps \
  --tile-size 256 \
  --patch-size 16 \
  --epochs 100 \
  --routing-epochs 45 \
  --frozen-epochs 35 \
  --selection-threshold 0.5 \
  --learning-rate 1e-4 \
  --finetune-lr-multiplier 0.1 \
  --weight-decay 1e-4 \
  --router-weight 1.0 \
  --router-boundary-weight 0.2 \
  --router-pairwise-weight 0.1 \
  --router-morphology-weight 0.1 \
  --router-boundary-margin 0.1 \
  --router-pair-margin 0.2 \
  --seg-cldice-weight 0.5 \
  --seg-cldice-iterations 10 \
  --batch-size 8 \
  --val-batch-size 8 \
  --num-workers 4 \
  --val-ratio 0.2 \
  --seed 26 \
  --run-dir runs/train
```

如果显存不足，将 Batch Size 调小：

```bash
python train.py \
  --data-root data \
  --image-dir images \
  --mask-dir edge_maps \
  --tile-size 256 \
  --patch-size 16 \
  --epochs 100 \
  --routing-epochs 45 \
  --frozen-epochs 35 \
  --batch-size 2 \
  --val-batch-size 2 \
  --num-workers 2 \
  --val-ratio 0.2 \
  --seed 26 \
  --run-dir runs/train
```

最低显存配置可以使用：

```bash
python train.py \
  --data-root data \
  --image-dir images \
  --mask-dir edge_maps \
  --tile-size 256 \
  --patch-size 16 \
  --epochs 100 \
  --routing-epochs 45 \
  --frozen-epochs 35 \
  --batch-size 1 \
  --val-batch-size 1 \
  --num-workers 0 \
  --val-ratio 0.2 \
  --seed 26 \
  --run-dir runs/train
```

### 三阶段训练日程

以上推荐命令对应：

```text
Epoch 1–45：Routing 阶段
Epoch 46–80：Segmentation 阶段
Epoch 81–100：Fine-tune 阶段
```

要求：

```text
epochs > routing_epochs + frozen_epochs
```

训练输出目录：

```text
runs/train/<时间编号>/
├── args.json
├── metrics.csv
├── results.png
├── checkpoints/
│   ├── latest.pt
│   ├── best_router.pt
│   └── best.pt
└── validation/
```

Checkpoint 用途：

```text
latest.pt：最近一轮完整状态，用于断点恢复
best_router.pt：第一阶段路由指标最佳，仅用于分析 Router
best.pt：分割指标最佳，用于正式推理和部署
```

---

## 6. 限制大图缩放尺寸后训练

默认不主动缩小原图，只将高度和宽度补齐到 `256` 的整数倍。

如果原始图像非常大，可用 `--max-size` 限制缩放后的最长边。例如限制最长边不超过 `2048`：

```bash
python train.py \
  --data-root data \
  --image-dir images \
  --mask-dir edge_maps \
  --tile-size 256 \
  --patch-size 16 \
  --max-size 2048 \
  --epochs 100 \
  --routing-epochs 45 \
  --frozen-epochs 35 \
  --batch-size 8 \
  --val-batch-size 8 \
  --num-workers 4 \
  --val-ratio 0.2 \
  --seed 26 \
  --run-dir runs/train
```

使用 `--max-size` 时仍然保持非正方形和原始长宽比，缩放后再补齐到 `256` 的倍数。

---

## 7. 断点恢复训练

必须使用格式版本为 `6` 的新 Tile Checkpoint。旧的 `768×768` Checkpoint 与当前网络不兼容。

从最近状态继续训练：

```bash
python train.py \
  --resume runs/train/<时间编号>/checkpoints/latest.pt \
  --epochs 120
```

示例：

```bash
python train.py \
  --resume runs/train/20260915_120000_000000/checkpoints/latest.pt \
  --epochs 120
```

恢复训练时：

- 延续原来的训练集和验证集划分；
- 延续模型、优化器和混合精度状态；
- 延续原来的阶段配置；
- `--epochs` 表示新的总训练轮数，不是额外训练轮数；
- 推荐始终从 `latest.pt` 恢复，不要从旧的 `best.pt` 覆盖后续历史。

---

## 8. 单张大图推理

```bash
python infer.py \
  --input data/test_images/1787823739062.png \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --output-dir outputs \
  --tile-batch-size 16 \
  --threshold 0.5
```

其中：

```text
--tile-batch-size：一次送入 CUDA 的 256×256 Tile 数量
--threshold：最终像素掩码二值化阈值
```

Router 的选块阈值从 Checkpoint 自动读取，`--threshold` 不会改变 Router 阈值。

---

## 9. 目录批量推理

```bash
python infer.py \
  --input test_data \
  --checkpoint runs/train/20260915_152729_759933/checkpoints/best.pt \
  --output-dir outputs \
  --tile-batch-size 16 \
  --threshold 0.5
```

关闭多图汇总拼图：

```bash
python infer.py \
  --input data/test_images \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --output-dir outputs \
  --tile-batch-size 16 \
  --threshold 0.5 \
  --no-contact-sheet
```

调整汇总拼图显示尺寸：

```bash
python infer.py \
  --input data/test_images \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --output-dir outputs \
  --tile-batch-size 16 \
  --threshold 0.5 \
  --tile-width 400 \
  --tile-height 300
```

推理输出：

```text
outputs/<时间编号>/
├── args.json
├── <name>_prob.png
├── <name>_mask.png
├── <name>_overlay.png
└── contact_sheet.png
```

文件说明：

```text
*_prob.png：灰度概率图
*_mask.png：经过 threshold 二值化的掩码
*_overlay.png：原图与预测区域叠加图
contact_sheet.png：批量推理汇总图
```

推理程序会自动执行：

```text
原图
→ Letterbox 到 H/W 为 256 倍数
→ 生成 256×256 Tile
→ 按 tile-batch-size 分批送入 CUDA
→ Tile 掩码按坐标回拼
→ 去除 Padding
→ 恢复原始 H×W
```

---

## 10. 推理速度测试

模型前向速度测试：

```bash
python infer.py \
  --input data/test_images \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --output-dir outputs \
  --benchmark \
  --warmup 10 \
  --iterations 50
```

Benchmark 输出保存在：

```text
outputs/<时间编号>/benchmark.json
```

注意：Benchmark 用于模型前向性能评估，不代表包含图像读取、Letterbox、Tile 回拼和文件保存的完整端到端速度。

---

## 11. 导出 ONNX

使用正式分割权重导出：

```bash
python export_onnx.py \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --output deployment/eetf_tile_256.onnx \
  --batch-size 4 \
  --opset 18
```

⚠️ **重要**：必须使用 `--batch-size > 1` 导出，否则 ONNX 图中的 batch 维度会被固化为 1，导致推理时无法使用更大的 tile-batch-size。推荐使用 `--batch-size 4` 或更大值。

可选的 ONNX 简化（需要 `pip install onnxsim`）：

```bash
python export_onnx.py \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --output deployment/eetf_tile_256.onnx \
  --batch-size 32 \
  --opset 18 \
  --simplify
```

推荐使用 `opset 18`。当前 PyTorch/ONNX 导出器对部分算子降级到较旧 Opset 时可能不完整。

ONNX 接口：

```text
输入：
images       [B,3,256,256]

输出：
mask_logits  [B,1,256,256]
keep_logits  [B,256,2]
keep_probs   [B,256]
```

只有 Batch 维是动态维度，空间尺寸固定为 `256×256`。

部署前向使用：

```text
forward_deploy(images)
```

该路径对全部 `256` 个 Patch 执行两个 Decoder，再通过 Router Mask 进行张量融合，避免动态 Python 列表和动态长度索引进入 ONNX/TensorRT 图。

---

## 12. ONNX Runtime 推理

### 安装依赖

```bash
pip install onnxruntime-gpu  # CUDA
# 或
pip install onnxruntime      # CPU only
```

### 单张图片推理（CUDA）

```bash
python infer_onnx.py \
  --input data/test_images/example.png \
  --onnx deployment/eetf_tile_256.onnx \
  --output-dir outputs_onnx \
  --provider cuda \
  --tile-batch-size 16 \
  --threshold 0.5
```

### 目录批量推理

```bash
python infer_onnx.py \
  --input data/test_images \
  --onnx deployment/eetf_tile_256.onnx \
  --output-dir outputs_onnx \
  --provider cuda \
  --tile-batch-size 16 \
  --threshold 0.5
```

### CPU 推理

```bash
python infer_onnx.py \
  --input data/test_images/example.png \
  --onnx deployment/eetf_tile_256.onnx \
  --output-dir outputs_onnx_cpu \
  --provider cpu \
  --tile-batch-size 4 \
  --threshold 0.5
```

### TensorRT ExecutionProvider 推理

需要安装 TensorRT 和 `onnxruntime-gpu` with TensorRT support。

```bash
python infer_onnx.py \
  --input data/test_images \
  --onnx deployment/eetf_tile_256.onnx \
  --output-dir outputs_trt \
  --provider tensorrt \
  --tile-batch-size 16 \
  --threshold 0.5 \
  --trt-fp16 \
  --trt-engine-cache-dir deployment/trt_cache
```

参数说明：

```text
--provider: auto（自动选择） | cuda | cpu | tensorrt
--tile-batch-size: 每批处理的 256×256 Tile 数量
--threshold: 二值化阈值
--trt-fp16: 启用 TensorRT FP16 模式
--trt-engine-cache-dir: TensorRT Engine 缓存目录（首次推理会构建并保存）
--verbose: 打印每张图片的详细信息
--no-contact-sheet: 跳过目录推理时的 Contact Sheet 生成
```

输出文件：

```text
<name>_prob.png    # 概率图 [0, 255]
<name>_mask.png    # 二值掩码
<name>_overlay.png # 叠加可视化
contact_sheet.png  # 目录推理时的汇总预览
```

ONNX Runtime 推理使用与 PyTorch `infer.py` 完全相同的 Tile 切分、坐标记录、回拼和尺寸恢复逻辑。

### 验证 PyTorch 和 ONNX 输出一致性

对同一张图片分别进行 PyTorch 和 ONNX 推理：

```bash
# PyTorch 推理
python infer.py \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --input data/test_images/example.png \
  --output-dir outputs_pytorch \
  --tile-batch-size 16

# ONNX 推理
python infer_onnx.py \
  --input data/test_images/example.png \
  --onnx deployment/eetf_tile_256.onnx \
  --output-dir outputs_onnx \
  --provider cuda \
  --tile-batch-size 16
```

使用 Python 比较输出差异：

```python
import cv2
import numpy as np

# 加载概率图
pt_prob = cv2.imread('outputs_pytorch/<时间编号>/<name>_prob.png', cv2.IMREAD_GRAYSCALE)
onnx_prob = cv2.imread('outputs_onnx/<name>_prob.png', cv2.IMREAD_GRAYSCALE)

# 计算差异
prob_diff = np.abs(pt_prob.astype(float) - onnx_prob.astype(float))
print(f'Max diff: {prob_diff.max():.2f}')
print(f'Mean diff: {prob_diff.mean():.2f}')
print(f'Pixels with diff > 1: {(prob_diff > 1).mean()*100:.2f}%')

# 比较二值掩码
pt_mask = cv2.imread('outputs_pytorch/<时间编号>/<name>_mask.png', cv2.IMREAD_GRAYSCALE)
onnx_mask = cv2.imread('outputs_onnx/<name>_mask.png', cv2.IMREAD_GRAYSCALE)
mask_diff = (pt_mask != onnx_mask)
print(f'Mask identical: {(~mask_diff).mean()*100:.2f}%')
```

正常情况下：
- 概率图平均差异 < 1/255
- 二值掩码相同像素 > 99.9%
- 差异主要来自浮点运算精度和 ONNX Runtime 优化

---

## 13. 使用 TensorRT 构建 FP16 Engine（可选）

确保系统已安装 TensorRT，并且 `trtexec` 在 PATH 中。

构建动态 Batch FP16 Engine：

```bash
trtexec \
  --onnx=deployment/eetf_tile_256.onnx \
  --saveEngine=deployment/eetf_tile_256_fp16.engine \
  --minShapes=images:1x3x256x256 \
  --optShapes=images:8x3x256x256 \
  --maxShapes=images:32x3x256x256 \
  --fp16
```

如果显存较小：

```bash
trtexec \
  --onnx=deployment/eetf_tile_256.onnx \
  --saveEngine=deployment/eetf_tile_256_fp16.engine \
  --minShapes=images:1x3x256x256 \
  --optShapes=images:4x3x256x256 \
  --maxShapes=images:8x3x256x256 \
  --fp16
```

TensorRT 性能测试：

```bash
trtexec \
  --loadEngine=deployment/eetf_tile_256_fp16.engine \
  --shapes=images:8x3x256x256 \
  --warmUp=1000 \
  --duration=10
```

---

## 14. 常用参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--tile-size` | `256` | 大图 Tile 尺寸，当前必须为 256 |
| `--patch-size` | `16` | Tile 内 Patch 尺寸，当前必须为 16 |
| `--max-size` | 空 | 可选的大图缩放最长边限制 |
| `--batch-size` | `1` | 训练 Tile Batch Size |
| `--val-batch-size` | `1` | 验证 Tile Batch Size |
| `--tile-batch-size` | `16` | 大图推理时的 CUDA Tile Batch Size |
| `--selection-threshold` | `0.5` | Router 选块阈值 |
| `--threshold` | `0.5` | 最终像素掩码阈值 |
| `--routing-epochs` | `20` | 第一阶段训练轮数 |
| `--frozen-epochs` | `20` | 第二阶段冻结路由模块的训练轮数 |
| `--epochs` | `100` | 三个阶段的总训练轮数 |
| `--num-workers` | `0` | DataLoader Worker 数量 |
| `--val-ratio` | `0.2` | 原图级验证集比例 |
| `--seed` | `42` | 数据划分和训练随机种子 |

---

## 15. 推荐执行顺序

```bash
# 1. 激活环境
source /home/fth/miniconda3/bin/activate
conda activate py11
cd /home/fth/EdTF/EeTF

# 2. 整理原始数据（首次）
python prepare_teed_dataset.py \
  --src /home/fth/EdTF/DATA \
  --out data_256 \
  --tile-size 256 \
  --seed 2026 \
  --dry-run  # 先预览

python prepare_teed_dataset.py \
  --src /home/fth/EdTF/DATA \
  --out data_256 \
  --tile-size 256 \
  --seed 2026  # 正式生成

# 3. 启动训练
python train.py \
  --data-root data_256 \
  --image-dir images \
  --mask-dir edge_maps \
  --tile-size 256 \
  --patch-size 16 \
  --epochs 100 \
  --routing-epochs 45 \
  --frozen-epochs 35 \
  --batch-size 8 \
  --val-batch-size 8 \
  --num-workers 4 \
  --val-ratio 0.2 \
  --seed 26 \
  --run-dir runs/train

# 4. PyTorch 推理验证
python infer.py \
  --input data/test_images \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --output-dir outputs_pytorch \
  --tile-batch-size 16 \
  --threshold 0.5

# 5. 导出 ONNX（使用 batch-size > 1）
python export_onnx.py \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --output deployment/eetf_tile_256.onnx \
  --batch-size 4 \
  --opset 18 \
  --simplify

# 6. ONNX Runtime 推理验证
python infer_onnx.py \
  --input data/test_images \
  --onnx deployment/eetf_tile_256.onnx \
  --output-dir outputs_onnx \
  --provider cuda \
  --tile-batch-size 16

# 7. 验证 PyTorch 和 ONNX 输出一致性（对同一张图片）
python infer.py \
  --input data/test_images/example.png \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --output-dir outputs_pytorch_verify \
  --tile-batch-size 16

python infer_onnx.py \
  --input data/test_images/example.png \
  --onnx deployment/eetf_tile_256.onnx \
  --output-dir outputs_onnx_verify \
  --provider cuda \
  --tile-batch-size 16

# 使用 Python 比较差异（参见第 12 节）

# 8. （可选）构建 TensorRT FP16 Engine
trtexec \
  --onnx=deployment/eetf_tile_256.onnx \
  --saveEngine=deployment/eetf_tile_256_fp16.engine \
  --minShapes=images:1x3x256x256 \
  --optShapes=images:8x3x256x256 \
  --maxShapes=images:32x3x256x256 \
  --fp16

# 9. TensorRT ExecutionProvider 推理
python infer_onnx.py \
  --input data/test_images \
  --onnx deployment/eetf_tile_256.onnx \
  --output-dir outputs_trt \
  --provider tensorrt \
  --trt-fp16 \
  --trt-engine-cache-dir deployment/trt_cache
```

---

## 16. 网络结构 SVG

当前 256×256 Tile 特征流、16×16 Patch Router、双分支解码和大图回拼结构图：

```text
tile_feature_flow.svg
```

可直接使用浏览器、VS Code 或支持 SVG 的图像查看器打开。
