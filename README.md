# EeTF

PyTorch 工业稀疏边缘分割系统。当前版本采用两级空间结构：一级动态选择 `256x256` 大块，二级对每个选中大块的全部 `32x32` 小块进行块级监督和像素掩码预测。

## 环境与数据

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
cd /home/fth/EdTF/EeTF
```

数据位于 `data/images` 和 `data/edge_maps`，按文件名 stem 配对。图像和灰度标签同步 Letterbox 到 `2560x2560`；图像使用双线性插值，标签使用最近邻插值。灰度标签保留 `[0,1]` 软像素目标，块级占用标签使用 `label > 0`。

## 两级特征流

```text
输入图像 [B,3,2560,2560]
    |
    +-- 一级无重叠切图: 100 个 [256,256] 大块，网格 10x10
    |       |
    |       +-- CNN + 完整 10x10 Grid 跨块交互 -> [B,100,64,...]
    |       +-- 28D 统计特征 -> MLP -> [B,100,256]
    |       +-- FeatureFusion -> 一级 Router -> [B,100,2]
    |                                      |
    |                                      +-- 未选中大块直接丢弃
    |                                      +-- 选中大块 packed 为 [M1,3,256,256]
    |
    +-- 每个选中大块切成 8x8=64 个 [32,32] 小块
            |
            +-- 全部保留，不进行第二次动态丢弃
            +-- 二级块级 logits: [M1,64,2]
            +-- 二级像素 logits: [M1*64,1,32,32]
            +-- 先按 8x8 回填大块，再按 10x10 回填整图

最终 mask_logits: [B,1,2560,2560]
```

`M1 = sum(N1(b))`，其中 `N1(b)` 是第 `b` 个样本被一级 Router 选中的大块数。不同样本可以有不同的 `N1`，模型使用 packed 表示和逐样本索引回填，不做 padding。

## 模块

```text
main.py                         两级 EdgeDynamicViT
patching/patching.py            unfold 无重叠矩形切图
utils/block_feature_extractor.py 4+16+8=28D 统计与邻域特征
models/cnn_base.py              一级大块 CNN 与二维 Grid 交互
models/mlp_base.py              28 -> 32 -> 64 -> 128 -> 256
models/feature_fusion.py        64D CNN 与 256D 统计特征融合
models/dynamic_vit.py           一级概率阈值 Router 与 packed 分组
models/patch_decoders.py        二级块级 logits 与像素 Decoder
engine/train.py                 三阶段损失和单轮训练
engine/validate.py              阶段验证
train.py                        训练、恢复和 format_version=6 checkpoint
infer.py                        2560x2560 层级模型推理
```

## 训练阶段

### 阶段一：一级路由与二级块级监督

一级 Router 监督 `256x256` 大块占用目标：

```text
macro_targets: [B,100]
macro_logits:  [B,100,2]
```

对选中的大块继续生成 `32x32` 小块占用目标：

```text
sub_targets:      [M1,64]
sub_block_logits: [M1,64,2]
```

阶段一损失为：

```text
L_stage1 = L_macro_router + sub_block_weight * L_sub_block
```

其中 Router 保留带邻域关系的 weighted CE、pairwise、boundary 和 morphology 项。

### 阶段二：像素掩码

一级路由模块冻结，选中大块的全部二级小块进入像素 Decoder：

```text
sub_mask_logits: [M1*64,1,32,32]
```

二级小块先回拼为选中大块，再回拼为完整图像。阶段二损失为：

```text
L_stage2 = Weighted BCE + Dice + lambda_clDice * Soft-clDice
```

Soft-clDice 作用于完整 `[B,1,2560,2560]` 输出。

### 阶段三：联合微调

所有模块解冻，一级路由损失以较低学习率参与联合微调：

```text
L_stage3 = L_stage2 + router_weight * L_router
```

## 参数与运行

```bash
python train.py \
  --data-root data \
  --batch-size 1 \
  --val-batch-size 1 \
  --routing-epochs 20 \
  --frozen-epochs 20 \
  --sub-block-weight 1.0
```

推理只接受 `format_version=6` 的 `2560x2560` hierarchical checkpoint：

```bash
python infer.py --input data/images \
  --checkpoint runs/train/<run>/checkpoints/best.pt \
  --output-dir runs/infer
```

未选中的一级大块在最终输出中保持背景值。推理 benchmark 只统计 `model(inputs)` 的纯前向时间，并同时返回 `batch_fps` 和 `image_fps`。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
git diff --check
```

测试覆盖一级 `10x10` 路由、二级 `8x8` 小块、块级监督、多 Batch packed 索引、两级回填、空选中分支、损失梯度和阶段调度。

输入尺寸、Patch 尺寸和 checkpoint 格式已经升级；旧的 `768x768`、单级 `64x64`、Selected/Remaining 双分支 checkpoint 不兼容当前版本。
