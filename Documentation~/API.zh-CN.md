# willkyu ONNX Runtime Inference API 参考

命名空间：`OnnxRuntimeInference`

## 基本约定

- 输入 tensor 为 NCHW `float[]`，语义是 `[1, 3, height, width]`。
- 普通路径：`FrameOnnxRunner.TryBeginRun(CapturedFrame)` 在 runner 内部执行 CPU resize 和 CPU tensor 转换。
- 快路径：`PreparedFrameOnnxInputBuffer` 让捕获线程预处理原始帧，主线程通过 `TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease)` 直接推理已经写好的 tensor。
- DirectML 仅在 Windows Editor/Player 下可用；初始化失败会回退 CPU，并写入 `InitializationWarning`。

## 主要类

### `DetectorRuntimeKind`

选择 ONNX Runtime 后端。

| 值 | 说明 |
| --- | --- |
| `OnnxRuntimeCpu` | 使用 CPU Execution Provider。 |
| `OnnxRuntimeDirectML` | Windows 下优先使用 DirectML Execution Provider；失败时回退 CPU。 |

### `ColorOrder`

颜色通道顺序。

| 值 | 说明 |
| --- | --- |
| `Rgb` | R、G、B 顺序。 |
| `Bgr` | B、G、R 顺序。 |

### `InferencePreprocessBackend`

一次推理实际使用的预处理链路。

| 值 | 说明 |
| --- | --- |
| `CpuResizeCpuTensor` | runner 内部执行 CPU resize，并在 CPU 写 NCHW tensor。 |
| `PreparedCpuTensor` | 捕获线程已经准备好模型输入，runner 只执行 ONNX Runtime 与 decode。 |

### `DetectorInputSpec`

描述模型输入。

```csharp
new DetectorInputSpec(
    int width,
    int height,
    ColorOrder tensorColorOrder,
    bool normalizeToUnitRange)
```

参数：`width`、`height` 必须大于 0；`tensorColorOrder` 决定写入 NCHW 时的通道顺序；`normalizeToUnitRange=true` 时将 0-255 转为 0-1。

属性：`Width`、`Height`、`TensorColorOrder`、`NormalizeToUnitRange`。

### `DetectorClass`

检测类别配置。

```csharp
new DetectorClass(int id, string label, float threshold)
```

参数：`id` 是类别 ID；`label` 是显示名称；`threshold` 是该类别的置信度阈值。

属性：`Id`、`Label`、`Threshold`。

### `DetectorModelProfile`

描述一个检测模型配置。

```csharp
new DetectorModelProfile(
    string detectorId,
    string displayName,
    string onnxModelName,
    DetectorInputSpec inputSpec,
    IReadOnlyList<DetectorClass> classes)
```

参数：`detectorId` 是稳定 ID；`displayName` 是显示名称；`onnxModelName` 是模型文件名或相对路径；`inputSpec` 不能为空；`classes` 可为 `null`，会转为空列表。

属性：`DetectorId`、`DisplayName`、`OnnxModelName`、`InputSpec`、`Classes`。

### `DetectionResult`

单个检测框。

```csharp
new DetectionResult(
    int classId,
    string label,
    float confidence,
    float x1,
    float y1,
    float x2,
    float y2)
```

属性同构造参数：`ClassId`、`Label`、`Confidence`、`X1`、`Y1`、`X2`、`Y2`。坐标为原始帧坐标。

### `DetectionBatch`

一次推理的检测结果集合。

```csharp
new DetectionBatch(
    IReadOnlyList<DetectionResult> detections,
    IReadOnlyDictionary<string, float> classScores)
```

属性：`Detections` 返回检测框列表；`ClassScores` 返回每个类别的最高置信度。

### `DetectorJsonConfigLoader`

读取 `detectors.json`。

```csharp
DetectorModelProfile LoadFirst(string json)
DetectorModelProfile LoadFirstFromFile(string path)
IReadOnlyList<DetectorModelProfile> LoadAll(string json)
IReadOnlyList<DetectorModelProfile> LoadAllFromFile(string path)
```

参数：`json` 是配置文本，不能为空；`path` 是配置文件路径，不能为空。

返回值：`LoadFirst...` 返回第一个模型配置；`LoadAll...` 返回全部模型配置。

### `IOnnxDetectorSession`

推理会话抽象。

```csharp
float[] Run(float[] nchwInput, int width, int height)
```

参数：`nchwInput` 是 `[1,3,height,width]` 展平数组；`width` 和 `height` 是输入 tensor 尺寸。

返回值：ONNX Runtime 输出 tensor 的展平 `float[]`。具体布局由模型决定。

### `OnnxRuntimeDetectorSession`

基于 `Microsoft.ML.OnnxRuntime.InferenceSession` 的会话实现。

```csharp
new OnnxRuntimeDetectorSession(
    string modelPath,
    DetectorRuntimeKind runtimeKind,
    string preferredInputName = "images",
    string preferredOutputName = "output0")
```

参数：`modelPath` 是 ONNX 文件路径；`runtimeKind` 是首选后端；`preferredInputName` 和 `preferredOutputName` 是首选输入/输出名，不存在时使用模型中的第一个输入/输出。

属性：`RequestedRuntimeKind`、`ActualRuntimeKind`、`InitializationWarning`、`InputName`、`OutputName`。

`Run(...)` 参数和返回值同 `IOnnxDetectorSession.Run(...)`。

### `TensorPreprocessor`

把捕获帧或像素数组写为 NCHW tensor。

```csharp
float[] ToNchw(
    CapturedFrame frame,
    DetectorInputSpec inputSpec,
    ColorOrder sourceColorOrderOverride = ColorOrder.Rgb)
```

参数：`frame` 是捕获帧；`inputSpec` 是模型输入规格；`sourceColorOrderOverride` 可在 RGB/RGBA 源数据上强制按 BGR 解读。

返回值：新的 NCHW `float[]`。

```csharp
float[] ToNchw(
    byte[] pixels,
    int width,
    int height,
    FramePixelFormat format,
    bool rowsBottomUp,
    DetectorInputSpec inputSpec,
    ColorOrder sourceColorOrderOverride = ColorOrder.Rgb)
```

参数：`pixels` 是原始像素；`format` 支持 `Rgba32`、`Bgra32`、`Rgb24`、`Bgr24`；`rowsBottomUp=true` 时写入 tensor 前按 top-down 语义翻转。

返回值：新的 NCHW `float[]`。

```csharp
void WriteNchw(
    byte[] pixels,
    int width,
    int height,
    FramePixelFormat format,
    bool rowsBottomUp,
    DetectorInputSpec inputSpec,
    ColorOrder sourceColorOrderOverride,
    float[] tensor)
```

参数同上，`tensor` 是调用方提供的输出数组，长度至少为 `width * height * 3`。

返回值：无，直接写入 `tensor`。

### `PreparedFrameOnnxInputBuffer`

双缓冲模型输入。用于捕获线程预处理快路径：从 top-down RGBA32 原始帧一次写入模型尺寸 preview bytes 和 NCHW tensor。

```csharp
new PreparedFrameOnnxInputBuffer(
    DetectorInputSpec inputSpec,
    FrameResizeAlgorithm resizeAlgorithm = FrameResizeAlgorithm.Nearest)
```

参数：`inputSpec` 是模型输入规格；`resizeAlgorithm` 是从原始帧缩放到模型输入尺寸的采样算法，默认 `Nearest`，与原项目默认一致。

属性：`Width`、`Height` 返回模型输入尺寸；`ResizeAlgorithm` 返回构造时指定的采样算法。

```csharp
bool TryAcquireWrite(out PreparedFrameOnnxInputBuffer.WriteLease lease)
```

参数：`lease` 接收写入租约。

返回值：`true` 表示有空闲缓冲可写；`false` 表示两个缓冲都正在被读取或等待消费。

```csharp
bool TryAcquireLatest(out PreparedFrameOnnxInputBuffer.ReadLease lease)
```

参数：`lease` 接收最新可读模型输入。若随后传入 `FrameOnnxRunner.TryBeginRun(lease)` 且返回 `true`，runner 会持有并释放该 lease；若返回 `false`，调用方应自行释放。

返回值：`true` 表示取得 latest 输入；`false` 表示当前没有可读输入。

```csharp
bool TryCopyLatestPreviewPixels(
    byte[] destination,
    out int width,
    out int height,
    out int originalWidth,
    out int originalHeight,
    out long frameId,
    out DateTime timestampUtc)
```

参数：`destination` 是调用方提供的 RGBA32 目标数组；其它 `out` 参数返回模型输入尺寸、原始帧尺寸、帧号和时间戳。

返回值：`true` 表示已复制 latest preview；`false` 表示暂无可复制帧。

`WriteLease.TryPrepare(CapturedFrame sourceFrame)`：参数 `sourceFrame` 必须是 top-down `Rgba32`；返回 `true` 表示已完成 resize、preview 和 tensor 写入并发布。

`ReadLease` 属性：`Width`、`Height` 是模型输入尺寸；`OriginalWidth`、`OriginalHeight` 是捕获原始尺寸；`FrameId`、`TimestampUtc` 是原始帧信息；`PreviewPixels` 是模型尺寸 RGBA32 preview；`Tensor` 是 NCHW `float[]`。`CreatePreviewFrame()` 返回包装 preview 的 `CapturedFrame`，用于少量调试或适配旧接口。

### `YoloEnd2EndDecoder`

解码 YOLO end2end `[1,300,6]` 输出，行语义为 `[x1, y1, x2, y2, confidence, classId]`。

```csharp
DetectionBatch Decode(
    float[] output,
    DetectorModelProfile profile,
    int originalWidth,
    int originalHeight,
    bool applyClassNms = false,
    float nmsIouThreshold = 0.5f)
```

```csharp
DetectionBatch Decode(
    float[] output,
    DetectorInputSpec inputSpec,
    IReadOnlyList<DetectorClass> classes,
    int originalWidth,
    int originalHeight,
    bool applyClassNms = false,
    float nmsIouThreshold = 0.5f)
```

参数：`output` 是模型输出；`profile` 或 `inputSpec/classes` 提供输入尺寸和类别阈值；`originalWidth`、`originalHeight` 用于把检测框缩放回原始帧；`applyClassNms` 控制是否按类别执行 NMS；`nmsIouThreshold` 是 NMS 阈值。

返回值：`DetectionBatch`，包含过滤后的检测框和各类别最高分。

### `FrameOnnxRunnerOptions`

`FrameOnnxRunner` 配置。

| 属性 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `ResizeAlgorithm` | `FrameResizeAlgorithm` | `Bilinear` | `TryBeginRun(CapturedFrame)` 路径的 CPU resize 采样算法。 |
| `ApplyClassNms` | `bool` | `false` | 是否按类别执行 NMS。 |
| `NmsIouThreshold` | `float` | `0.5` | NMS IoU 阈值。 |
| `DisposeSession` | `bool` | `false` | 释放 runner 时是否一并释放传入 session。 |

### `FrameOnnxRunner`

非阻塞帧推理入口。每次只运行一帧；上一帧仍在推理时，新的 `TryBeginRun` 返回 `false`。

```csharp
new FrameOnnxRunner(
    IOnnxDetectorSession session,
    DetectorModelProfile profile,
    FrameResizeAlgorithm resizeAlgorithm = FrameResizeAlgorithm.Bilinear,
    bool applyClassNms = false,
    float nmsIouThreshold = 0.5f,
    bool disposeSession = false)
```

```csharp
new FrameOnnxRunner(
    IOnnxDetectorSession session,
    DetectorModelProfile profile,
    FrameOnnxRunnerOptions options)
```

参数：`session` 是推理会话；`profile` 是模型配置；`options` 可为 `null`，此时使用默认配置。

```csharp
bool TryBeginRun(CapturedFrame sourceFrame)
```

参数：`sourceFrame` 是当前捕获帧，当前 runner 要求 `FramePixelFormat.Rgba32`。如果 `sourceFrame.RowsBottomUp=true`，runner 会在内部转成 top-down 后再 resize。

返回值：`true` 表示本次推理已启动；`false` 表示上一帧仍在运行。

```csharp
bool TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease preparedInput)
```

参数：`preparedInput` 是 `PreparedFrameOnnxInputBuffer.TryAcquireLatest` 返回的 latest 输入 lease。返回 `true` 后 runner 会持有该 lease 直到推理结束；返回 `false` 时调用方仍负责释放。

返回值：`true` 表示本次推理已启动；`false` 表示上一帧仍在运行。

```csharp
bool TryGetResult(out FrameOnnxInferenceResult result)
```

参数：`result` 接收完成结果。

返回值：`true` 表示取到一条结果；`false` 表示尚无完成结果。

属性：`IsRunning` 表示是否正在推理；`Profile` 返回构造时传入的模型配置。

### `FrameOnnxInferenceResult`

`FrameOnnxRunner` 的完成结果。

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `Batch` | `DetectionBatch` | 解码后的检测结果。 |
| `RawOutput` | `float[]` | ONNX Runtime 原始输出。 |
| `OriginalWidth` / `OriginalHeight` | `int` | 原始帧尺寸。 |
| `InputWidth` / `InputHeight` | `int` | 模型输入尺寸。 |
| `ResizeDuration` | `TimeSpan` | CPU resize 耗时；prepared 路径为 0。 |
| `TensorDuration` | `TimeSpan` | RGBA 到 NCHW 耗时；prepared 路径为 0。 |
| `InferenceDuration` | `TimeSpan` | ONNX Runtime `Run` 耗时。 |
| `DecodeDuration` | `TimeSpan` | YOLO decode 耗时。 |
| `PreprocessDuration` | `TimeSpan` | `ResizeDuration + TensorDuration`。 |
| `TotalActiveDuration` | `TimeSpan` | resize、tensor、ORT、decode 总耗时。 |
| `ActiveFps` | `double` | `1 / TotalActiveDuration`，不包含限频等待。 |
| `PreprocessBackend` | `InferencePreprocessBackend` | 实际预处理链路。 |
| `Succeeded` | `bool` | 是否成功。 |
| `ErrorMessage` | `string` | 失败原因；成功时为空。 |

### `OrtNativeLibraryPreloader`

Windows 下按绝对路径预加载 ONNX Runtime 原生 DLL。

```csharp
void EnsureLoaded()
```

返回值：无。非 Windows 平台无操作；Windows 下找不到或加载失败时抛出异常。

属性：`LoadedPath` 返回已加载的 `onnxruntime.dll` 路径；`LoadWarning` 返回预加载阶段的警告文本。
