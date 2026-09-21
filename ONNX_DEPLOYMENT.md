# EeTF ONNX 部署说明

## 概述

本文档说明 EdgeDynamicViT 模型如何从 PyTorch 动态路由转换为 ONNX/TensorRT 固定形状部署。

---

## 核心设计：固定双分支全计算

### 训练时的动态路由

```python
# 训练阶段
keep_probs = router(features)  # [B, 256]
selected_indices = torch.nonzero(keep_probs >= threshold).squeeze(1)  # 动态长度
remaining_indices = torch.nonzero(keep_probs < threshold).squeeze(1)  # 动态长度

# 只计算选中的 Patch
selected_features = features[:, selected_indices]  # [B, N_sel, C]
selected_logits = selected_decoder(selected_features)

remaining_features = features[:, remaining_indices]  # [B, N_rem, C]
remaining_logits = remaining_decoder(remaining_features)

# 动态索引回填
output[:, selected_indices] = selected_logits
output[:, remaining_indices] = remaining_logits
```

**特点**：

- `selected_indices` 长度动态（0 到 256）
- 每个 Tile 的选中数量不同
- 计算量与路由决策相关
- 需要 `nonzero()`、`gather()`、`scatter()` 等动态操作

**问题**：

- ONNX/TensorRT 对动态长度 Tensor 优化较弱
- `NonZero` 输出形状不确定
- 动态索引操作难以融合
- 可能无法构建 TensorRT Engine

---

### 部署时的固定形状路由

```python
# 部署阶段 forward_deploy()
keep_probs = router(features)  # [B, 256]
keep_mask = keep_probs >= threshold  # [B, 256] bool

# 两个分支都对全部 256 个 Patch 计算
selected_logits = selected_decoder(features)    # [B, 256, H, W]
remaining_logits = remaining_decoder(features)  # [B, 256, H, W]

# 张量选择（关键）
keep_mask_expanded = keep_mask[..., None, None]  # [B, 256, 1, 1]
merged_logits = torch.where(
    keep_mask_expanded,
    selected_logits,
    remaining_logits
)  # [B, 256, H, W]
```

**特点**：

- 全程固定形状
- 两个 Decoder 都执行全部 Patch 计算
- `torch.where` 逐元素硬选择
- 无动态索引、无动态长度

**优势**：

- ONNX 图干净稳定
- TensorRT 可最大化优化
- 支持 Kernel Fusion、FP16、INT8
- 部署结果与训练语义完全一致

---

## 关键问题解答

### Q1: 两个分支都计算，会干扰吗？

**不会**。`torch.where` 是逐元素硬开关：

```python
output[i] = selected_logits[i] if keep_mask[i] else remaining_logits[i]
```

未选中分支的结果**直接丢弃**，不参与最终输出，不存在数值混合或加权融合。

### Q2: Router 还保留吗？

**保留**。部署路径中仍然包含：

- Router 网络
- `keep_logits` 输出
- `keep_probs` 输出
- 阈值选块逻辑
- 两个分支的选择

只是从"动态索引"改为"固定形状 + 张量选择"。

### Q3: 计算量增加了多少？

训练时计算量：

```
256 * (p * F_selected + (1-p) * F_remaining)
```

部署时计算量：

```
256 * (F_selected + F_remaining)
```

当 `p=0.5` 时约 2 倍开销，但考虑到：

- TensorRT Kernel Fusion
- FP16/INT8 加速
- 无动态索引开销
- 256×256 Tile 本身很小

实际推理速度可能优于动态索引方案。

### Q4: 能否保留动态路由？

**可以，但不推荐**。动态路由需要：

- ONNX `NonZero` + `Gather` + `ScatterND`
- TensorRT 对这些算子优化较弱
- 可能无法构建 Engine
- 调试困难

适用场景：

- ONNX Runtime CPU/CUDA EP（不用 TensorRT）
- Router 剪枝收益远大于固定计算开销
- 愿意牺牲 TensorRT 优化

### Q5: 如何验证一致性？

```python
# PyTorch 训练模式
train_out = model(images)

# PyTorch 部署模式
deploy_out = model.forward_deploy(images)

# 验证
assert torch.allclose(train_out['mask_logits'], deploy_out['mask_logits'], rtol=1e-5)
```

ONNX 导出后：

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
onnx_out = session.run(None, {"images": images.numpy()})

# 验证
assert np.allclose(deploy_out['mask_logits'].numpy(), onnx_out[0], rtol=1e-4)
```

---

## ONNX 输入输出

### 输入

```text
images: [batch, 3, 256, 256] float32
```

- Batch 维动态
- 空间尺寸固定为 256×256
- 每个输入是一个 Tile

### 输出

```text
mask_logits: [batch, 1, 256, 256] float32  # 分割 logits
keep_logits: [batch, 256, 2]      float32  # Router 原始输出
keep_probs:  [batch, 256]         float32  # Router 概率（softmax 后的类别1）
```

**用途**：

- `mask_logits`：最终分割预测，用于大图回拼
- `keep_logits` / `keep_probs`：路由决策调试和可视化

### 输入图像格式和预处理约定

ONNX 模型本身只接收已经完成预处理的 Tile，不直接接收 JPEG/PNG 文件。
部署程序负责完成图像读取、颜色转换、数值缩放和维度转换。

#### 1. 原始图像读取格式

`infer_onnx.py` 使用 OpenCV 读取原图：

```python
image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
```

此时格式为：

```text
类型：       numpy.ndarray
布局：       [H, W, 3]，HWC
颜色顺序：   BGR
数据类型：   uint8
取值范围：   [0, 255]
```

如果使用其他图像库读取，也必须在送入预处理前转换为等价的 RGB 图像。模型训练和当前推理代码使用 RGB 顺序，不能直接把 BGR 数组送给模型。

#### 2. BGR 转 RGB、HWC 转 CHW、归一化

当前模型没有额外的 ImageNet mean/std 标准化；只进行 `/255.0` 缩放：

```python
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
```

转换后的单张大图为：

```text
类型：       torch.Tensor
布局：       [3, H, W]，CHW
颜色顺序：   RGB
数据类型：   float32
取值范围：   [0.0, 1.0]
```

#### 3. Letterbox 到 256 的整数倍

模型空间输入固定为 `256×256`，但原始大图可以是任意宽高。预处理通过 `letterbox_to_multiple(image, 256)`：

- 保持原始宽高比；
- 默认不缩放原图，只在边缘补零，使高、宽都成为 256 的整数倍；
- 如果指定 `max_size`，才会先按比例缩小，再补齐到 256 的整数倍；
- 补边位置和缩放信息保存在 `metadata` 中；
- 当前 `letterbox()` 实现采用上下、左右对称补边，不能在外部自行改变补边位置。

输出为：

```text
letterbox_image：torch.float32 [3, padded_height, padded_width]
其中 padded_height % 256 == 0
     padded_width  % 256 == 0
```

例如，原图 `1200×2048` 会变成 `1280×2048`，随后得到 `5×8=40` 个 Tile。

`metadata` 至少包含：

```python
{
    "original_height": ...,
    "original_width": ...,
    "resized_height": ...,
    "resized_width": ...,
    "pad_top": ...,
    "pad_bottom": ...,
    "pad_left": ...,
    "pad_right": ...,
    "scale": ...,
    "target_height": ...,
    "target_width": ...,
}
```

### 切图格式

使用 `split_tiles(letterbox_image, tile_size=256)` 进行无重叠切图。切图顺序是从上到下、从左到右：

```python
tiles, tile_metadata = split_tiles(letterbox_image, 256)
```

输出格式为：

```text
tiles：torch.float32 [N, 3, 256, 256]
```

其中：

```text
N = (padded_height / 256) * (padded_width / 256)
```

每个 Tile 已经是：

```text
布局：       CHW
颜色：       RGB
数据类型：   float32
取值范围：   [0.0, 1.0]
尺寸：       [3, 256, 256]
```

`tile_metadata` 是长度为 `N` 的字典列表，每个 Tile 对应一个坐标记录：

```python
{
    "batch_index": 0,
    "tile_index": 0,
    "row": 0,
    "col": 0,
    "top": 0,
    "left": 0,
    "bottom": 256,
    "right": 256,
}
```

其中 `top/left/bottom/right` 是该 Tile 在 Letterbox 画布上的像素坐标。由于 Tile 不重叠，后续回拼不需要平均或加权融合。

### 送入 ONNX Runtime 的格式

`torch.Tensor` 需要转换为 NumPy 数组后才能传给 ONNX Runtime：

```python
batch_tiles = tiles[start:end].numpy()
mask_logits, keep_logits, keep_probs = session.run(
    None,
    {"images": batch_tiles},
)
```

ONNX 输入的最终格式是：

```text
名称：       images
类型：       numpy.ndarray
数据类型：   float32
布局：       NCHW
形状：       [B, 3, 256, 256]
颜色顺序：   RGB
取值范围：   [0.0, 1.0]
```

这里的 `B` 是当前一次送入 ONNX 的 Tile 数量，由 `--tile-batch-size` 控制。最后一个 Batch 可以小于该值。它不是原图数量，而是 Tile 数量。

例如：

```text
大图共 40 个 Tile
--tile-batch-size 16
实际送入模型的 Batch：16、16、8
```

不要传入以下格式：

```text
错误：uint8 [B, 256, 256, 3]
错误：BGR [B, 3, 256, 256]
错误：单张大图 [1, 3, H, W]（H/W 不是 256）
错误：范围 [0, 255] 的 float32
错误：CHW 但仍然保留 batch 维为 [3, 256, 256]，未组成 [B,3,256,256]
```

### ONNX 输出后处理

`mask_logits` 是 logits，不是概率，需要先做 sigmoid：

```python
mask_probs = 1.0 / (1.0 + np.exp(-mask_logits))
```

输出形状为：

```text
mask_logits：numpy.float32 [B, 1, 256, 256]
mask_probs： numpy.float32 [B, 1, 256, 256]
```

`keep_logits` 和 `keep_probs` 不参与像素掩码拼接：

```text
keep_logits：[B, 256, 2]
keep_probs： [B, 256]
```

它们用于观察每个 Tile 内 256 个 `16×16` Patch 的 Router 选择结果。

### Tile 回拼和恢复原图

所有 ONNX Batch 的输出先按照送入顺序拼接：

```python
all_mask_logits = np.concatenate(all_mask_logits, axis=0)  # [N,1,256,256]
mask_probs = sigmoid(all_mask_logits)                       # [N,1,256,256]
mask_probs_tensor = torch.from_numpy(mask_probs)
```

然后使用切图时保存的同一份 `tile_metadata` 回拼：

```python
letterbox_prob = merge_tiles(
    mask_probs_tensor,
    tile_metadata,
    (target_height, target_width),
    256,
)
```

回拼结果为：

```text
letterbox_prob：torch.float32 [1, 1, padded_height, padded_width]
```

`merge_tiles()` 对每个 Tile 执行：

```python
canvas[0, :, top:bottom, left:right] = tile
```

因此必须保证：

- 输出 Tile 数量和 `tile_metadata` 数量一致；
- Batch 输出的顺序没有改变；
- `tile_metadata` 与原始切图使用的是同一份列表；
- 不要按置信度、Router 结果或其他方式重新排序输出。

最后调用 `unletterbox(letterbox_prob, metadata)`：

1. 根据 `pad_top/left` 和 `resized_height/width` 裁掉补零区域；
2. 如果 Letterbox 阶段发生了缩放，则插值恢复到 `original_height × original_width`；
3. 得到原图尺寸的概率图。

当前 `predict_image_onnx()` 返回的概率图实际形状为：

```text
probability_map：numpy.float32 [1, original_height, original_width]
```

保存 PNG 前需要去掉通道维：

```python
probability_image = (probability_map.squeeze(0) * 255).clip(0, 255).astype(np.uint8)
cv2.imwrite("result_prob.png", probability_image)
```

二值掩码同样基于去掉通道维后的概率图：

```python
binary_mask = (
    probability_map.squeeze(0) >= threshold
).astype(np.uint8) * 255
```

完整的形状变化如下：

```text
原始文件
  JPEG/PNG
  ↓ cv2.imread
BGR uint8 [H,W,3]
  ↓ BGR→RGB、HWC→CHW、/255
RGB float32 [3,H,W]
  ↓ letterbox_to_multiple(256)
RGB float32 [3,Hpad,Wpad]
  ↓ split_tiles(256)
Tile float32 [N,3,256,256]
  ↓ 按 tile-batch-size 分批
ONNX 输入 float32 [B,3,256,256]
  ↓ ONNX Runtime
mask_logits [B,1,256,256]
  ↓ sigmoid
Tile probability [N,1,256,256]
  ↓ 按 row/col 坐标 merge_tiles
Letterbox probability [1,1,Hpad,Wpad]
  ↓ unletterbox
原图 probability [1,H,W]
  ↓ squeeze、阈值化、保存
PNG [H,W]
```

### 阈值处理

Router 阈值在模型内部固化（来自 checkpoint 的 `selection_threshold`）。

如需修改阈值，有两种方案：

**方案 A：重新导出 ONNX**（推荐）

```bash
# 修改阈值后重新导出
python export_onnx.py --checkpoint best.pt --output model_new_threshold.onnx
```

**方案 B：阈值作为 ONNX 输入**

需要修改 `forward_deploy()` 接受阈值参数，导出时增加输入：

```python
def forward_deploy(self, images, threshold):
    keep_mask = keep_probs >= threshold
    ...
```

优点：部署时可动态调整  
缺点：多一个输入，TensorRT 图稍复杂

第一版建议使用方案 A。

---

## 推理流程

### PyTorch 推理（infer.py）

```python
1. 加载大图 [H, W, 3]
2. Letterbox 到 256 的整数倍
3. 切分为 N 个 256×256 Tile
4. 批量推理：model.forward_deploy(tiles)
5. 得到 N 个 [1, 256, 256] mask_logits
6. sigmoid → probability
7. 按 Tile 坐标回拼到 Letterbox 画布
8. 去除 Padding
9. 恢复原图尺寸
```

### ONNX Runtime 推理（infer_onnx.py）

```python
1. 加载大图 [H, W, 3]
2. Letterbox 到 256 的整数倍
3. 切分为 N 个 256×256 Tile
4. 批量推理：session.run(None, {"images": tiles})
5. 得到 N 个 [1, 256, 256] mask_logits
6. sigmoid → probability
7. 按 Tile 坐标回拼到 Letterbox 画布
8. 去除 Padding
9. 恢复原图尺寸
```

**关键**：两者使用完全相同的 Tile 切分、坐标记录和回拼逻辑（来自 `data/transforms/letterbox.py`），确保空间一致性。

---

## 部署路径对比

| 特性 | PyTorch 动态 | PyTorch forward_deploy | ONNX/TensorRT |
|---|---|---|---|
| Router | ✓ | ✓ | ✓ |
| 阈值选块 | ✓ | ✓ | ✓ |
| Selected Decoder | 仅选中 Patch | 全部 Patch | 全部 Patch |
| Remaining Decoder | 仅剩余 Patch | 全部 Patch | 全部 Patch |
| 动态索引 | ✓ | ✗ | ✗ |
| 张量选择 | 索引回填 | `torch.where` | ONNX `Where` |
| 输出一致性 | - | 与动态完全一致 | 与 forward_deploy 一致 |
| 计算量 | 与路由决策相关 | 固定 | 固定 |
| TensorRT 优化 | 差 | 优 | 优 |

---

## 使用示例

### 1. 导出 ONNX

```bash
python export_onnx.py \
  --checkpoint runs/train/20260115_120000/checkpoints/best.pt \
  --output deployment/eetf_tile_256.onnx \
  --opset 18 \
  --simplify
```

### 2. ONNX Runtime CUDA 推理

```bash
python infer_onnx.py \
  --input data/test_images \
  --onnx deployment/eetf_tile_256.onnx \
  --output-dir outputs_onnx \
  --provider cuda \
  --tile-batch-size 16
```

### 3. TensorRT ExecutionProvider

```bash
python infer_onnx.py \
  --input data/test_images \
  --onnx deployment/eetf_tile_256.onnx \
  --output-dir outputs_trt \
  --provider tensorrt \
  --trt-fp16 \
  --trt-engine-cache-dir deployment/trt_cache
```

### 4. 独立构建 TensorRT Engine

```bash
trtexec \
  --onnx=deployment/eetf_tile_256.onnx \
  --saveEngine=deployment/eetf_tile_256_fp16.engine \
  --minShapes=images:1x3x256x256 \
  --optShapes=images:8x3x256x256 \
  --maxShapes=images:32x3x256x256 \
  --fp16
```

---

## 依赖安装

### ONNX Runtime GPU

```bash
pip install onnxruntime-gpu
```

### ONNX 简化（可选）

```bash
pip install onnxsim
```

### TensorRT

参考 NVIDIA 官方文档安装 TensorRT。

---

## 故障排除

### 问题：ONNX 导出失败

**检查**：

- Checkpoint 是否 `format_version=6`
- 是否 `tile_size=256, patch_size=16`
- 是否使用 `best.pt` 或 `final.pt`（拒绝 routing 阶段）

### 问题：ONNX Runtime 报错"CUDAExecutionProvider not available"

**解决**：

```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

验证：

```python
import onnxruntime as ort
print(ort.get_available_providers())
# 应包含 'CUDAExecutionProvider'
```

### 问题：TensorRT 构建失败

**检查**：

- ONNX Opset 是否支持（推荐 18）
- 是否有不支持的算子
- 尝试不使用 `--fp16`

### 问题：ONNX 与 PyTorch 结果不一致

**验证步骤**：

1. 先验证 `forward_deploy()` 与 `forward()` 一致
2. 再验证 ONNX 与 `forward_deploy()` 一致
3. 使用相同的输入 Tile
4. 检查 sigmoid 是否在模型内/外执行

---

## 性能优化

### Tile Batch Size

- CUDA：16-32
- TensorRT FP16：32-64
- CPU：4-8

根据显存/内存调整。

### Provider 选择

优先级：

```text
TensorRT FP16 > CUDA > CPU
```

首次推理：

```bash
--provider cuda
```

验证无误后：

```bash
--provider tensorrt --trt-fp16
```

### TensorRT Engine Cache

首次推理会构建 Engine（耗时），后续推理直接加载缓存：

```bash
--trt-engine-cache-dir deployment/trt_cache
```

---

## 总结

EdgeDynamicViT 的 ONNX 部署采用**固定双分支全计算 + torch.where 张量选择**策略：

✅ 保留 Router 语义  
✅ 输出与训练一致  
✅ ONNX/TensorRT 兼容性强  
✅ 部署稳定性高  
✅ 无动态索引复杂度  

计算量虽然固定，但 TensorRT 优化和 FP16 加速可部分抵消，适合工业边缘检测的实时部署需求。
