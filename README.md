# EeTF

PyTorch 工业稀疏边缘分割：28 维统计特征与 CNN 融合，概率阈值选块，双分支像素解码。

## 环境和数据

```bash
source /home/fth/miniconda3/bin/activate
conda activate py11
cd /home/fth/EdTF/EeTF
```

依赖 PyTorch、torchvision、NumPy、Pillow、OpenCV、Matplotlib（Agg 无窗口绘图）。
完整数据存放于 `data/images` 和 `data/edge_maps`，按相同文件名 stem 配对。
灰度标签除以 255 保留软像素目标，块级标签通过 `label > 0` 后最大池化获得。
默认固定 seed=42、验证比例 0.2；训练打乱，验证不打乱。当前只支持 B=1。
训练同步几何增强，图像使用双线性插值，标签使用最近邻；最后 Letterbox 到
`768x768`、`512x1024`、`1024x512`。验证只做确定性 Letterbox。

## 模块与特征流

```text
main.py                         EdgeDynamicViT 模型
patching/patching.py             64x32 无重叠分块
utils/block_feature_extractor.py 4+16+8 维统计及四邻域结构特征
models/cnn_base.py               Patch CNN → [B,N,8192]
models/mlp_base.py               28→32→64→128→256→512→768
models/feature_fusion.py         CNN 投影 + MLP → [B,N,768]
models/dynamic_vit.py            阈值路由、2D-RoPE Attention
models/patch_decoders.py         Selected/Remaining 双分支
losses/sparse_segmentation_loss.py  加权 BCE + Dice
engine/train.py                  分阶段损失和单轮优化
engine/validate.py               分阶段验证
train.py                        阶段调度、checkpoint 保存和恢复
infer.py / engine/infer.py       掩码、叠加图与拼图
```

```text
图像 [B,3,H,W] → Patch [B*N,3,64,32]
  ├─ CNN → [B,N,8192] → 投影 768
  └─ 28 维统计 → MLP → [B,N,768]
               ↓ 融合
         Router → Keep 概率 [B,N]
               ↓ 阈值划分
  ├─ selected: RoPE 特征投影 768→128 + 原始 Patch CNN
  │            → 块内 Transformer → [N1,1,64,32]
  └─ remaining: [B,N2,768] → 512 → [B*N2,1,32,16]
                → 上采样解码 → [B,N2,1,64,32]
               ↓ 原索引回填
         mask_logits [B,1,H,W]
```

28 维为 4 维基础统计、16 维四邻域基础差异、8 维边界强度/法向梯度差异，不补零。
只有缺失邻居的对应列为零。RGB 输入应在 `[0,1]`，统计使用 FP32；结构响应除以 32、
标准差乘以 2、熵除以 `log2(num_levels)`、边界法向梯度差除以 2。
外部 `block_features` 必须使用相同列顺序和尺度。Sobel 使用 replicate 填充。

## 阈值选块

不使用 top-k、Gumbel 采样或固定保留率惩罚：

```text
keep_probability >= selection_threshold → selected
keep_probability <  selection_threshold → remaining
```

数量随图像变化；支持全部选中和全部不选，空分支不执行解码器。
阈值仍是离散决策，不是可微软门控。像素损失不会通过分组索引训练 Router；
Router 使用独立选块监督，CNN/MLP/融合可经像素特征路径更新。

## 三阶段训练

| 阶段 | 默认轮次 | 可训练模块 | 损失 |
|---|---|---|---|
| routing | 1–20 | CNN、统计 MLP、融合、Router | 加权 CE + 相邻关系 + 结构边界 + 孤立结构损失 |
| segmentation | 21–40 | RoPE、两个解码器 | Weighted BCE + Dice |
| finetune | 41–100 | 全部模块 | 像素损失 + router_weight × 选块损失 |

选块预训练不执行 RoPE/解码器。Router Loss 为：
`L_router = L_weighted_CE + λ_pair L_pairwise + λ_boundary L_boundary + λ_morph L_morphology`。
其中相邻 Patch 同标签时约束概率接近、异标签时约束概率分离；边界项将概率推离选块阈值；
形态学项只惩罚预测类别与标签不一致且四邻域孤立的 Patch。所有结构项作用于连续 keep 概率，
因此可以反向传播到 Router。冻结阶段将选块模块设为 eval 并关闭梯度，
包括冻结 BatchNorm 运行统计；解冻微调的选块模块学习率默认是解码器的 0.1 倍。
阶段顺序连续衔接上一轮状态，不自动回滚到 best_router.pt。
像素 BCE 的前景权重由非零标签支持区域计数决定，Dice 保留灰度软目标。
`losses/dynamic_loss.py` 仅保留为旧工具，新流程不调用它。

```bash
python train.py \
  --data-root data --image-dir images --mask-dir edge_maps \
  --epochs 100 --routing-epochs 45 --frozen-epochs 54 \
  --selection-threshold 0.5 --learning-rate 1e-4 \
  --finetune-lr-multiplier 0.1 --router-weight 1.0 \
  --router-boundary-weight 0.2 --router-pairwise-weight 0.1 \
  --router-morphology-weight 0.1 --router-boundary-margin 0.1 \
  --router-pair-margin 0.2 \
  --val-ratio 0.2 --seed 26 --run-dir runs/train
```

`--epochs` 是三个阶段的总轮数，必须大于 routing-epochs + frozen-epochs。
原 `--keep-ratio` 和 `--ratio-weight` 参数已移除。

保存文件：
- `latest.pt`：最近轮次完整状态，包括阶段、阈值、日程、优化器、随机状态和数据划分。
- `best_router.pt`：选块阶段验证集 Patch F1 最高的 checkpoint，不可直接用于像素分割推理。
- `best.pt`：分割阶段与微调阶段中前景像素 F1 最高的 checkpoint。

```bash
python train.py --resume runs/train/<时间编号>/checkpoints/latest.pt --epochs 120
```

恢复时使用 checkpoint 的日程、阈值及数据参数，拒绝变化的数据划分；
可指定新的总轮数及 worker 数。新格式恢复沿用 checkpoint 所在 run，忽略新的 run-dir；旧三阶段 checkpoint 没有曲线历史时创建新 run，不补造历史指标。只支持格式版本 2 的三阶段 checkpoint。
原固定 top-k 训练状态不支持直接恢复。

验证阶段同时报告 Patch Precision/Recall/F1 和实际选中比例；
分割与微调阶段还报告 foreground_f1、IoU、Precision、Recall、Dice 和降背景权重准确率。
普通 Accuracy 仅作参考。像素指标沿用逐图平均，Patch 指标累计混淆计数。

## 推理可视化

```bash
python infer.py --input data/images --checkpoint runs/train/20260908_091004_581768/checkpoints/best.pt \
  --output-dir outputs --threshold 0.5 --tile-width 400 --tile-height 300
```

`--input` 可为单张图或目录。选块阈值自动从 checkpoint 读取；
`--threshold` 仅控制最终像素掩码阈值，不是选块阈值。
去除 Letterbox 填充后恢复原始尺寸，输出：

```text
<name>_prob.png       灰度概率图
<name>_mask.png       二值掩码
<name>_overlay.png    红色半透明叠加图
contact_sheet.png    多图两列：原图 | 叠加图
```

用 `--no-contact-sheet` 关闭拼图。

## 纯模型推理 FPS

```bash
python infer.py --input data/images \
  --checkpoint runs/train/<时间编号>/checkpoints/best.pt \
  --benchmark --warmup 10 --iterations 50 --output-dir outputs
```

`--benchmark` 仅测速，不保存掩码或执行后处理；每张图先完成读取、Letterbox 和设备传输，
然后预热 10 次，测量 50 次 `model(inputs)`。CUDA 同步计时，使用 B=1、FP32 和 inference_mode。
计时包含模型内部切块、统计特征、动态路由和解码，排除读取、预处理、设备传输、输出 sigmoid、
掩码还原、可视化与文件保存。同步墙钟时间包含模型内部 CPU 调度，不是端到端处理 FPS。

输出在新的时间目录 `outputs/<时间编号>/benchmark.json`，包含每图输入尺寸、平均延迟与 FPS，
以及设备、选块阈值和汇总结果。汇总 FPS = 总测量帧数 / 总前向耗时，不直接平均逐图 FPS。
阈值路由计算量依赖图像内容，建议使用真实图片目录测量；预热耗时不计入结果。

## 运行目录与曲线

每次新训练和推理使用本地时间 `YYYYMMDD_HHMMSS_ffffff`（微秒）自动编号，避免覆盖。
`--run-dir` 默认 `runs/train`；旧参数 `--checkpoint-dir` 是它的别名，现在同样表示 run 父目录。
`--output-dir` 默认 `outputs`，表示推理输出父目录。推理必须显式指定 checkpoint。

```text
runs/train/<时间编号>/
├── args.json
├── metrics.csv
├── results.png
├── checkpoints/{latest,best_router,best}.pt
└── validation/
    └── epoch_0001_routing/
        ├── metrics.json
        ├── confusion_matrix.png
        └── confusion_matrix_normalized.png
outputs/<时间编号>/
├── args.json
├── *_prob.png / *_mask.png / *_overlay.png
└── contact_sheet.png
```

每轮验证以 epoch 和阶段编号归属于本次训练 run。第一阶段每轮输出 Patch 二分类混淆矩阵，并在训练日志中分别记录 CE、相邻关系、结构边界和孤立结构四项 Router 子损失：
纵轴是真实类别，横轴是预测类别，类别顺序 Drop/Keep，计数布局 `[[TN,FP],[FN,TP]]`。
另存按真实类别逐行归一化的矩阵，缺失类别显示为 0。

`results.png` 每轮更新六个子图：总 Loss、选块 Loss、像素 Loss、Patch F1、Pixel F1、实际选块比例。
训练/验证各一条曲线；背景标记 Routing、Frozen、Fine-tune，epoch 分界处有虚线。
总 Loss 在阶段之间断开，避免误认为损失组成不变。第一阶段像素指标、冻结阶段选块 Loss
在 CSV 留空，绘图不伪装成 0。训练指标来自带增强的训练前向，验证来自 eval 前向，二者并非同一测量条件。

恢复时以 checkpoint 内 history 重建曲线和 CSV，避免重复 epoch。
默认继续 latest.pt；如果 best checkpoint 早于当前 run 的 latest，拒绝覆盖较新的历史。

## 回归测试与限制

```bash
python -m unittest discover -s tests -v
```

覆盖统计特征、邻域关系、三种尺寸、RoPE 梯度、阈值分组、空分支、阶段冻结及模型/优化器重载。
阈值选块可能选择所有 Patch，峰值显存不再受固定比例约束。
ONNX 动态 nonzero 索引及空分支导出需要单独验证，当前不保证可导出。
改变统计语义和 Selected 上下文解码器后，旧权重不再严格兼容，建议重新训练。


## 推理速度

Overall: 68.04 FPS, 14.696 ms/frame