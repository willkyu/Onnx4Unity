# ONNX Runtime Inference API

命名空间：`OnnxRuntimeInference`。

## 输入约定

- 帧输入是 RGBA32 byte 数据，通过 `RgbaFrameInput` 传入。
- Tensor 输入是 NCHW `float[]`，语义为 `[1, 3, height, width]`。
- Prepared 快路径把 resize/tensor 工作放在推理调用之前完成，然后把 `PreparedFrameOnnxInputBuffer.ReadLease` 直接交给 `FrameOnnxRunner`。

## 核心类型

### `RgbaFrameInput`

```csharp
new RgbaFrameInput(
    byte[] pixels,
    int width,
    int height,
    bool rowsBottomUp = false,
    long frameId = 0,
    DateTime timestampUtc = default)
```

表示一帧 RGBA32 图像。`pixels` 至少需要 `width * height * 4` 字节。`rowsBottomUp=true` 表示内存中的第 0 行是底部行，runner 会在 resize/tensor 前翻转。属性与构造参数一致。

### `OnnxResizeAlgorithm`

`Nearest` 或 `Bilinear`。

### `DetectorRuntimeKind`

`OnnxRuntimeCpu` 使用 CPU。`OnnxRuntimeDirectML` 在 Windows 下使用 DirectML，初始化失败时回退 CPU。

### `DetectorInputSpec`

```csharp
new DetectorInputSpec(int width, int height, ColorOrder tensorColorOrder, bool normalizeToUnitRange)
```

描述模型输入尺寸、RGB/BGR tensor 通道顺序，以及是否把 0-255 归一化到 0-1。

### `DetectorClass`、`DetectorModelProfile`、`DetectionResult`、`DetectionBatch`

这些是模型元数据与检测结果的数据类型。

- `DetectorClass(int id, string label, float threshold)`
- `DetectorModelProfile(string detectorId, string displayName, string onnxModelName, DetectorInputSpec inputSpec, IReadOnlyList<DetectorClass> classes)`
- `DetectionResult(int classId, string label, float confidence, float x1, float y1, float x2, float y2)`
- `DetectionBatch(IReadOnlyList<DetectionResult> detections, IReadOnlyDictionary<string, float> classScores)`

### `DetectorJsonConfigLoader`

```csharp
DetectorModelProfile LoadFirst(string json)
DetectorModelProfile LoadFirstFromFile(string path)
IReadOnlyList<DetectorModelProfile> LoadAll(string json)
IReadOnlyList<DetectorModelProfile> LoadAllFromFile(string path)
```

从 `detectors.json` 使用的 JSON 格式读取检测模型配置。

## 预处理

### `TensorPreprocessor`

```csharp
float[] ToNchw(RgbaFrameInput frame, DetectorInputSpec inputSpec)
float[] ToNchw(byte[] rgba32, int width, int height, bool rowsBottomUp, DetectorInputSpec inputSpec)
void WriteNchw(byte[] rgba32, int width, int height, bool rowsBottomUp, DetectorInputSpec inputSpec, float[] tensor)
```

把 RGBA32 输入写成 NCHW float tensor。`WriteNchw` 写入调用方提供的数组，不分配新 tensor。

### `PreparedFrameOnnxInputBuffer`

```csharp
new PreparedFrameOnnxInputBuffer(DetectorInputSpec inputSpec, OnnxResizeAlgorithm resizeAlgorithm = OnnxResizeAlgorithm.Nearest)
```

双缓冲模型输入。写入端把 top-down RGBA32 原始帧转换为模型尺寸 preview bytes 和 NCHW tensor。

```csharp
bool TryAcquireWrite(out PreparedFrameOnnxInputBuffer.WriteLease lease)
bool TryAcquireLatest(out PreparedFrameOnnxInputBuffer.ReadLease lease)
bool TryCopyLatestPreviewPixels(byte[] destination, out int width, out int height, out int originalWidth, out int originalHeight, out long frameId, out DateTime timestampUtc)
```

`WriteLease.TryPrepare(RgbaFrameInput sourceFrame)` 在输入为 top-down RGBA32 时返回 `true` 并发布结果。bottom-up 帧返回 `false`；需要先由调用方翻转。

`ReadLease` 暴露 `Width`、`Height`、`OriginalWidth`、`OriginalHeight`、`FrameId`、`TimestampUtc`、`PreviewPixels`、`Tensor`。`CreatePreviewFrame()` 返回包装 preview bytes 的 `RgbaFrameInput`。

## 推理

### `IOnnxDetectorSession`

```csharp
float[] Run(float[] nchwInput, int width, int height)
```

执行一次模型输入并返回展平输出。

### `OnnxRuntimeDetectorSession`

```csharp
new OnnxRuntimeDetectorSession(
    string modelPath,
    DetectorRuntimeKind runtimeKind,
    string preferredInputName = "images",
    string preferredOutputName = "output0")
```

封装 `Microsoft.ML.OnnxRuntime.InferenceSession`。属性：`RequestedRuntimeKind`、`ActualRuntimeKind`、`InitializationWarning`、`InputName`、`OutputName`。

### `FrameOnnxRunnerOptions`

| 属性 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `ResizeAlgorithm` | `OnnxResizeAlgorithm` | `Bilinear` | `TryBeginRun(RgbaFrameInput)` 的 resize 算法。 |
| `ApplyClassNms` | `bool` | `false` | 是否启用按类别 NMS。 |
| `NmsIouThreshold` | `float` | `0.5` | NMS IoU 阈值。 |
| `DisposeSession` | `bool` | `false` | 释放 runner 时是否释放 session。 |

### `FrameOnnxRunner`

```csharp
new FrameOnnxRunner(IOnnxDetectorSession session, DetectorModelProfile profile, OnnxResizeAlgorithm resizeAlgorithm = OnnxResizeAlgorithm.Bilinear, bool applyClassNms = false, float nmsIouThreshold = 0.5f, bool disposeSession = false)
new FrameOnnxRunner(IOnnxDetectorSession session, DetectorModelProfile profile, FrameOnnxRunnerOptions options)
bool TryBeginRun(RgbaFrameInput sourceFrame)
bool TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease preparedInput)
bool TryGetResult(out FrameOnnxInferenceResult result)
```

同一时间最多运行一帧。上一帧仍在运行时，`TryBeginRun` 返回 `false`。`TryBeginRun(ReadLease)` 返回 `true` 后，runner 持有并释放该 lease。

### `FrameOnnxInferenceResult`

主要属性：`Batch`、`RawOutput`、`OriginalWidth`、`OriginalHeight`、`InputWidth`、`InputHeight`、`SourceFrameId`、`SourceTimestampUtc`、`ResizeDuration`、`TensorDuration`、`InferenceDuration`、`DecodeDuration`、`PreprocessDuration`、`TotalActiveDuration`、`ActiveFps`、`PreprocessBackend`、`Succeeded`、`ErrorMessage`。

`ActiveFps` 不包含调用方限频或等待时间。

### `YoloEnd2EndDecoder`

```csharp
DetectionBatch Decode(float[] output, DetectorModelProfile profile, int originalWidth, int originalHeight, bool applyClassNms = false, float nmsIouThreshold = 0.5f)
DetectionBatch Decode(float[] output, DetectorInputSpec inputSpec, IReadOnlyList<DetectorClass> classes, int originalWidth, int originalHeight, bool applyClassNms = false, float nmsIouThreshold = 0.5f)
```

解码 YOLO end2end 输出行 `[x1, y1, x2, y2, confidence, classId]`。

### `OrtNativeLibraryPreloader`

```csharp
void EnsureLoaded()
```

Windows 下预加载随包放置的 ONNX Runtime 原生 DLL。`LoadedPath` 和 `LoadWarning` 提供诊断信息。
