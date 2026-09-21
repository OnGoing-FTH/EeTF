# ONNX 模型导出与部署校验清单

## 1. 当前问题

在 `Tool_GeneralEdgeExtraction` 插件中载入：

```text
D:\work_code\ORT\新工具文档\best_tile_256.onnx
```

插件显示：

```text
ONNX 的空间尺寸 H/W 必须固定，仅 batch 维允许动态
```

同时程序生成了 dump 文件：

```text
D:\work_code\RelWithDebInfo\Dumps\dump_2026_09_18_18_18_43_519.dmp
```

当前判断：这个提示发生在模型载入阶段，早于 ROI 构造和 tile 切分。不能用“插件已经把图片切成 256x256”来解决，因为插件是在 `Engine::create()` 和 `getInputDims()` 检查通过后才开始切图。

需要区分以下两个概念：

```text
实际送入推理的图片尺寸：由插件切图逻辑决定
ONNX 输入 Tensor 的声明尺寸：由导出模型的 graph.input 决定
```

## 2. 已确认的导出脚本行为

`export_onnx.py` 使用：

```python
sample = torch.zeros(args.batch_size, 3, 256, 256)
```

并且只声明 batch 维动态：

```python
dynamic_axes={
    'images': {0: 'batch'},
    'mask_logits': {0: 'batch'},
    'keep_logits': {0: 'batch'},
    'keep_probs': {0: 'batch'},
}
```

如果最终 ONNX 文件确实由该脚本导出，并且导出器没有额外改写 shape，那么预期签名应为：

```text
输入：
images       [batch, 3, 256, 256] float32

输出：
mask_logits  [batch, 1, 256, 256] float32
keep_logits  [batch, 256, 2]       float32
keep_probs   [batch, 256]         float32
```

其中只有 `batch` 是动态维度。允许的典型表示为：

```text
[-1, 3, 256, 256]
```

或：

```text
['batch', 3, 256, 256]
```

不允许的输入表示为：

```text
[-1, 3, -1, -1]
['batch', 3, 'height', 'width']
```

## 3. 必须从实际 ONNX 文件核验的内容

不要只看导出脚本最后的打印信息。脚本中的：

```python
print("Input: images [batch, 3, 256, 256]")
```

只是预期摘要，不是对最终文件的重新读取和验证。

请使用 Netron、ONNX Python API 或其他 ONNX 检查工具，直接打开 `best_tile_256.onnx`，记录以下结果。

### 3.1 输入签名

必须记录：

```text
输入名称：
输入数量：
输入类型：
输入 shape：
输入 layout：
```

期望结果：

```text
名称：images
数量：1
类型：float32
shape：[batch, 3, 256, 256]
layout：NCHW
```

重点确认：

- 是否只有一个输入。
- 名称是否为 `images`。
- 第 1 维是否为 `3`。
- 第 2 维是否固定为 `256`。
- 第 3 维是否固定为 `256`。
- 是否只有第 0 维是动态 batch。
- 是否存在 `height`、`width`、`sequence` 等额外动态符号。
- 是否意外导出了 NHWC：`[batch, 256, 256, 3]`。
- 是否意外导出了固定 batch=1：`[1, 3, 256, 256]`。

固定 batch=1 不一定是模型错误，但当前插件会尝试按多个 tile 组 batch 推理，需要特别确认插件是否应改为逐 tile 推理，或者重新导出动态 batch 模型。

### 3.2 输出签名

必须记录所有输出，不要只看第一个输出：

```text
输出名称：
输出类型：
输出 shape：
输出顺序：
```

期望结果：

```text
mask_logits  float32 [batch, 1, 256, 256]
keep_logits  float32 [batch, 256, 2]
keep_probs   float32 [batch, 256]
```

重点确认：

- 第一个输出是否确实是 `mask_logits`。
- `mask_logits` 是否为 logits，而不是已经 sigmoid 后的概率。
- `mask_logits` 的通道数是否为 `1`。
- `mask_logits` 的空间尺寸是否固定为 `256x256`。
- 输出顺序是否和 `output_names` 一致。
- 是否存在额外输出或输出顺序被导出器重排。
- `keep_logits`、`keep_probs` 是否确实存在；插件边缘提取只使用 `mask_logits`，但引擎需要正确处理多输出。

### 3.3 模型元数据和 opset

请记录：

```text
IR version：
opset version：
domain：
producer name/version：
metadata_props：
```

重点确认：

- opset 是否为脚本默认的 `18`。
- 是否有 `task=dense_map` 等元数据影响引擎默认行为。
- 是否有 `imgsz`、`names`、`author` 等元数据。
- 是否有不应存在的旧模型 metadata。
- 是否经过 `onnxsim` 简化，以及简化前后 shape 是否一致。

当前插件会显式设置自定义前处理和自定义后处理，因此模型不能依赖引擎自动选择错误的 ImageNet Normalize 或 LetterBox 流程。

## 4. ONNX Runtime 独立验证

在不经过插件和 TensorRT 的情况下，使用 ONNX Runtime 验证模型。

建议测试：

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    r"D:\work_code\ORT\新工具文档\best_tile_256.onnx",
    providers=["CPUExecutionProvider"],
)

for item in session.get_inputs():
    print("input:", item.name, item.shape, item.type)
for item in session.get_outputs():
    print("output:", item.name, item.shape, item.type)

x = np.zeros((1, 3, 256, 256), dtype=np.float32)
y = session.run(None, {"images": x})
for index, item in enumerate(y):
    print(index, item.shape, item.dtype, np.nanmin(item), np.nanmax(item))
```

必须确认：

- `session.get_inputs()` 能正常创建。
- `[1, 3, 256, 256]` 输入可以正常推理。
- 如果 batch 动态，测试 `[2, 3, 256, 256]` 是否也能正常推理。
- 输出数量、顺序、shape 和数值范围正确。
- 输出中没有 NaN 或 Inf。
- `mask_logits` 在 sigmoid 前后数值合理。

如果 ONNX Runtime CPU 也无法创建 session 或推理失败，优先修复模型导出；如果 ONNX Runtime 正常而 TensorRT 失败，则重点检查 TensorRT 算子支持、opset、动态 profile 和 CUDA/TensorRT 版本。

## 5. TensorRT 侧重点

当前 `OrtInfer` 实际使用 TensorRT 后端。模型载入路径大致为：

```text
ONNX
  -> TensorRT parser
  -> TensorRT network
  -> optimization profile
  -> build/deserialize ICudaEngine
  -> getTensorShape
  -> getInputDims
```

需要检查 TensorRT 日志中是否出现：

```text
Data parsing error
Model compilation failed
Failed to create ICudaEngine
Failed to deserialize ICudaEngine
setDimensions failed
Failed to set MAX input shape
```

### 5.1 动态 profile

如果输入是动态 batch，TensorRT profile 应类似：

```text
min: 1 x 3 x 256 x 256
opt: N x 3 x 256 x 256
max: N x 3 x 256 x 256
```

其中 `N` 由插件配置的最大 batch 决定。

如果输入是固定 batch=1，则 profile 必须使用：

```text
min/opt/max: 1 x 3 x 256 x 256
```

不能对固定 batch=1 的模型强行使用 `N>1` 的 profile。

### 5.2 引擎缓存

TensorRT 可能生成与模型同目录的 `.engine` 文件。需要确认：

- 是否存在旧的 `.engine` 缓存。
- 缓存是否由当前 CUDA/TensorRT 版本生成。
- 缓存是否对应当前 ONNX 文件。
- 删除旧缓存后是否重新构建。
- 当前引擎缓存文件名是否包含 batch、TensorRT、CUDA 和 cache version 信息。

如果 ONNX 已更换但仍复用旧 engine，可能出现 shape、输出数量或反序列化异常。

## 6. 当前引擎实现中的关键诊断点

`OrtInfer` 的 `getInputDims()` 最终来自 TensorRT engine 的输入 Tensor shape，而不是直接读取 Python 导出脚本。

动态 batch 的正常结果应类似：

```text
[-1, 3, 256, 256]
```

如果得到：

```text
[-1, -1, -1, -1]
```

需要确认 TensorRT parser 看到的 ONNX shape 是否真的如此。

如果得到空维度、未初始化维度或 H/W 小于等于 0，则不能直接断定 ONNX H/W 动态，也可能是 TensorRT 初始化失败后的无效状态。

当前 `Engine::create()` 流程需要重点核验：

```text
engine->init() 的返回值是否检查
init() 失败时是否仍然返回 Engine 对象
getInputDims() 是否可能在 init 失败后被调用
```

如果 `init()` 失败但仍返回对象，插件可能把真正的 TensorRT 初始化错误误显示为：

```text
ONNX 的空间尺寸 H/W 必须固定，仅 batch 维允许动态
```

因此需要同时记录：

```text
Engine::create 是否成功
Engine::init 返回值
getInputDims 实际返回数组
getOutputDims 实际返回数组
TensorRT parser/build 日志
```

## 7. 插件实际预处理契约

模型确认无误后，插件期望的每个 tile 是：

```text
原始 ROI
  -> 右侧/下侧补零到 256 的整数倍
  -> 无重叠切成 256x256
  -> BGR/HWC/uint8 cv::Mat
  -> Engine 的 ToTensor
  -> RGB/CHW/float32/[0,1]
  -> ONNX
```

插件不会对模型输入 tile 再做：

- ImageNet mean/std Normalize。
- LetterBox 缩放。
- 正方形拉伸。
- BGR 直接送入模型。
- 把整张非 256 图像直接送入模型。

需要和训练代码核对：

- 训练是否使用 RGB。
- 训练是否只除以 `255`。
- 训练是否没有 ImageNet Normalize。
- 训练是否采用 256x256 tile。
- 原图非 256 倍数时 padding 是右/下补零还是居中补零。
- mask 回拼时是否按同样的 tile 顺序。
- 模型输出是否为 logits，需要插件外部 sigmoid。

## 8. 建议外部检测后回填的信息

请将以下结果填入此表，便于和插件实现逐项对齐：

| 项目 | 检测结果 | 期望值 | 是否一致 |
|---|---|---|---|
| ONNX 输入名称 |  | `images` |  |
| ONNX 输入数量 |  | `1` |  |
| ONNX 输入类型 |  | `float32` |  |
| ONNX 输入 shape |  | `[batch,3,256,256]` |  |
| ONNX 输入 layout |  | `NCHW` |  |
| mask 输出名称 |  | `mask_logits` |  |
| mask 输出 shape |  | `[batch,1,256,256]` |  |
| mask 输出类型 |  | `float32` |  |
| 输出数量 |  | `3` |  |
| keep_logits shape |  | `[batch,256,2]` |  |
| keep_probs shape |  | `[batch,256]` |  |
| ONNX opset |  | `18` |  |
| ONNX Runtime batch=1 |  | 成功 |  |
| ONNX Runtime batch>1 |  | 动态模型应成功 |  |
| ONNX Runtime 输出 NaN/Inf |  | 无 |  |
| TensorRT parser |  | 成功 |  |
| TensorRT profile |  | H/W 固定 256 |  |
| TensorRT engine build |  | 成功 |  |
| `getInputDims()` |  | `[-1,3,256,256]` 或 `[1,3,256,256]` |  |
| `getOutputDims()` |  | 与 ONNX 输出一致 |  |
| 运行时 OrtInfer.dll |  | 与最新编译产物一致 |  |
| 运行时 OrtInfer.lib |  | 与最新编译产物一致 |  |
| 旧 `.engine` 缓存 |  | 已清理或确认匹配 |  |

## 9. 结果判定

### 情况 A：ONNX 输入就是动态 H/W

例如：

```text
[batch, 3, height, width]
```

说明导出结果与导出脚本预期不一致。优先检查：

- 实际执行的是否是这份 `export_onnx.py`。
- `torch.onnx.export` 版本和参数。
- 是否有其他导出脚本覆盖输出文件。
- 是否经过某个二次转换工具。
- 是否使用了错误的 checkpoint/model forward。

### 情况 B：ONNX 输入固定 H/W，但 ONNX Runtime 正常，TensorRT 失败

优先检查：

- TensorRT parser 日志。
- opset 兼容性。
- TensorRT 是否支持模型中的算子。
- CUDA/TensorRT/ONNX Runtime 版本组合。
- 动态 batch profile。
- 旧 engine 缓存。
- `Engine::init()` 返回值是否被正确处理。

### 情况 C：ONNX 和 TensorRT 都显示正确，但插件仍提示 H/W 不固定

优先检查：

- 程序实际加载的 `OrtInfer.dll` 路径。
- 插件编译时链接的 `OrtInfer.lib` 路径。
- DLL 与 LIB 是否来自同一次构建。
- 程序是否加载了安装目录或其他目录下的同名 DLL。
- `getInputDims()` 的实际返回值。
- 插件运行时加载的 ONNX 路径是否确实是当前 `best_tile_256.onnx`。

### 情况 D：加载失败并生成 dump

使用 Visual Studio 打开 `.dmp`，记录：

```text
Exception Code：
Faulting Module：
Faulting Function：
Call Stack：
```

重点判断故障模块属于：

```text
Tool_GeneralEdgeExtraction.dll
OrtInfer.dll
nvinfer.dll
nvonnxparser.dll
onnxruntime.dll
CUDA DLL
```

如果故障发生在 `OrtInfer.dll` 或 TensorRT DLL 中，应先解决引擎初始化/ABI/运行库问题；如果发生在插件 DLL 中，再检查插件的模型状态和 batch/output 处理。

## 10. 当前结论

根据 `export_onnx.py`，正确导出的模型应当固定空间尺寸 `256x256`，只允许 batch 动态。

因此当前报错不能直接说明 tile 没有切好。模型载入检查发生在切 tile 之前。

当前需要优先确认：

1. 实际 ONNX 文件的真实输入输出 shape。
2. ONNX Runtime 是否可以独立加载和推理。
3. TensorRT parser/build 是否成功。
4. `Engine::init()` 是否失败但被忽略。
5. `getInputDims()` 的真实返回值。
6. 程序是否实际加载了最新的 `OrtInfer.dll` 和匹配的 TensorRT/CUDA 运行库。
7. 是否存在不匹配的旧 `.engine` 缓存。

在这些信息确认前，不建议仅仅放宽插件的 H/W 检查，因为那可能掩盖 TensorRT 初始化失败或模型签名不一致问题。
