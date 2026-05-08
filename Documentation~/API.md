# willkyu ONNX Runtime Inference API Reference

Namespace: `OnnxRuntimeInference`.

This document covers the main public APIs only. Types that implement `IDisposable` should be disposed using normal C# ownership rules; common dispose methods are not expanded separately.

## Conventions

- Input tensors are NCHW `float[]` arrays with `[1, 3, height, width]` semantics.
- Normal path: `FrameOnnxRunner.TryBeginRun(CapturedFrame)` performs CPU resize and CPU tensor conversion inside the runner.
- Fast path: `PreparedFrameOnnxInputBuffer` lets a capture thread preprocess the original frame, then the main thread calls `TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease)` to run inference directly from the prepared tensor.
- DirectML is available only in Windows Editor/Player. Initialization failure falls back to CPU and records `InitializationWarning`.

## Main Types

### `DetectorRuntimeKind`

Selects the ONNX Runtime backend.

| Value | Description |
| --- | --- |
| `OnnxRuntimeCpu` | Uses the CPU Execution Provider. |
| `OnnxRuntimeDirectML` | Prefers the DirectML Execution Provider on Windows and falls back to CPU on failure. |

### `ColorOrder`

Color channel order.

| Value | Description |
| --- | --- |
| `Rgb` | R, G, B order. |
| `Bgr` | B, G, R order. |

### `InferencePreprocessBackend`

The preprocessing path used by an inference result.

| Value | Description |
| --- | --- |
| `CpuResizeCpuTensor` | The runner performed CPU resize and wrote NCHW tensor on CPU. |
| `PreparedCpuTensor` | Model input was already prepared by the capture thread; the runner only executed ONNX Runtime and decode. |

### `DetectorInputSpec`

Describes model input.

```csharp
new DetectorInputSpec(
    int width,
    int height,
    ColorOrder tensorColorOrder,
    bool normalizeToUnitRange)
```

Parameters: `width` and `height` must be positive; `tensorColorOrder` controls channel order when writing NCHW; `normalizeToUnitRange=true` converts 0-255 values to 0-1.

Properties: `Width`, `Height`, `TensorColorOrder`, `NormalizeToUnitRange`.

### `DetectorClass`

Detection class configuration.

```csharp
new DetectorClass(int id, string label, float threshold)
```

Parameters: `id` is class id; `label` is display name; `threshold` is the confidence threshold for that class.

Properties: `Id`, `Label`, `Threshold`.

### `DetectorModelProfile`

Describes one detector model configuration.

```csharp
new DetectorModelProfile(
    string detectorId,
    string displayName,
    string onnxModelName,
    DetectorInputSpec inputSpec,
    IReadOnlyList<DetectorClass> classes)
```

Parameters: `detectorId` is a stable id; `displayName` is a display label; `onnxModelName` is a model file name or relative path; `inputSpec` cannot be null; `classes` may be null and is converted to an empty list.

Properties: `DetectorId`, `DisplayName`, `OnnxModelName`, `InputSpec`, `Classes`.

### `DetectionResult`

One detection box.

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

Properties match constructor parameters: `ClassId`, `Label`, `Confidence`, `X1`, `Y1`, `X2`, `Y2`. Coordinates are in original-frame space.

### `DetectionBatch`

A batch of detection results from one inference run.

```csharp
new DetectionBatch(
    IReadOnlyList<DetectionResult> detections,
    IReadOnlyDictionary<string, float> classScores)
```

Properties: `Detections` returns detection boxes; `ClassScores` returns the highest confidence for each class label.

### `DetectorJsonConfigLoader`

Reads `detectors.json`.

```csharp
DetectorModelProfile LoadFirst(string json)
DetectorModelProfile LoadFirstFromFile(string path)
IReadOnlyList<DetectorModelProfile> LoadAll(string json)
IReadOnlyList<DetectorModelProfile> LoadAllFromFile(string path)
```

Parameters: `json` is config text and cannot be empty; `path` is config file path and cannot be empty.

Return value: `LoadFirst...` returns the first model profile; `LoadAll...` returns every model profile.

### `IOnnxDetectorSession`

Inference session abstraction.

```csharp
float[] Run(float[] nchwInput, int width, int height)
```

Parameters: `nchwInput` is a flattened `[1,3,height,width]` array; `width` and `height` are input tensor dimensions.

Return value: flattened ONNX Runtime output tensor. Layout depends on the model.

### `OnnxRuntimeDetectorSession`

Session implementation based on `Microsoft.ML.OnnxRuntime.InferenceSession`.

```csharp
new OnnxRuntimeDetectorSession(
    string modelPath,
    DetectorRuntimeKind runtimeKind,
    string preferredInputName = "images",
    string preferredOutputName = "output0")
```

Parameters: `modelPath` is the ONNX file path; `runtimeKind` is the preferred backend; `preferredInputName` and `preferredOutputName` are preferred input/output names, falling back to the model's first input/output when not found.

Properties: `RequestedRuntimeKind`, `ActualRuntimeKind`, `InitializationWarning`, `InputName`, `OutputName`.

`Run(...)` parameters and return value match `IOnnxDetectorSession.Run(...)`.

### `TensorPreprocessor`

Writes captured frames or pixel arrays as NCHW tensors.

```csharp
float[] ToNchw(
    CapturedFrame frame,
    DetectorInputSpec inputSpec,
    ColorOrder sourceColorOrderOverride = ColorOrder.Rgb)
```

Parameters: `frame` is a captured frame; `inputSpec` is model input spec; `sourceColorOrderOverride` can force RGB/RGBA source data to be interpreted as BGR.

Return value: a new NCHW `float[]`.

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

Parameters: `pixels` is raw pixel data; `format` supports `Rgba32`, `Bgra32`, `Rgb24`, and `Bgr24`; `rowsBottomUp=true` flips to top-down semantics while writing tensor.

Return value: a new NCHW `float[]`.

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

Parameters are the same as above, with `tensor` provided by the caller and at least `width * height * 3` floats long.

Return value: none; writes directly into `tensor`.

### `PreparedFrameOnnxInputBuffer`

Double-buffered model input. Used by the capture-thread preprocessing fast path: one pass from a top-down RGBA32 original frame writes model-sized preview bytes and NCHW tensor.

```csharp
new PreparedFrameOnnxInputBuffer(
    DetectorInputSpec inputSpec,
    FrameResizeAlgorithm resizeAlgorithm = FrameResizeAlgorithm.Nearest)
```

Parameters: `inputSpec` is model input spec; `resizeAlgorithm` is the sampling algorithm from original frame to model input size and defaults to `Nearest`, matching the original project default.

Properties: `Width` and `Height` return model input size; `ResizeAlgorithm` returns the constructor resize algorithm.

```csharp
bool TryAcquireWrite(out PreparedFrameOnnxInputBuffer.WriteLease lease)
```

Parameter: `lease` receives the write lease.

Return value: `true` when a free buffer can be written; `false` when both buffers are being read or waiting to be consumed.

```csharp
bool TryAcquireLatest(out PreparedFrameOnnxInputBuffer.ReadLease lease)
```

Parameter: `lease` receives the latest readable model input. If it is passed to `FrameOnnxRunner.TryBeginRun(lease)` and the call returns `true`, the runner owns and releases the lease. If the call returns `false`, the caller must dispose it.

Return value: `true` when latest input is acquired; `false` when no readable input is available.

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

Parameters: `destination` is a caller-provided RGBA32 destination array; other `out` parameters return model input size, original frame size, frame id, and timestamp.

Return value: `true` when latest preview was copied; `false` when no frame is available.

`WriteLease.TryPrepare(CapturedFrame sourceFrame)`: `sourceFrame` must be top-down `Rgba32`; returns `true` after resize, preview, and tensor are written and published.

`ReadLease` properties: `Width` and `Height` are model input size; `OriginalWidth` and `OriginalHeight` are original capture size; `FrameId` and `TimestampUtc` are original frame metadata; `PreviewPixels` is model-sized RGBA32 preview; `Tensor` is NCHW `float[]`. `CreatePreviewFrame()` wraps preview bytes in a `CapturedFrame` for small debugging or legacy adapter cases.

### `YoloEnd2EndDecoder`

Decodes YOLO end2end `[1,300,6]` output with rows `[x1, y1, x2, y2, confidence, classId]`.

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

Parameters: `output` is model output; `profile` or `inputSpec/classes` provide input size and class thresholds; `originalWidth` and `originalHeight` scale detection boxes back to original frame coordinates; `applyClassNms` controls per-class NMS; `nmsIouThreshold` is the NMS threshold.

Return value: `DetectionBatch` containing filtered detection boxes and highest score per class.

### `FrameOnnxRunnerOptions`

`FrameOnnxRunner` configuration.

| Property | Type | Default | Description |
| --- | --- | --- | --- |
| `ResizeAlgorithm` | `FrameResizeAlgorithm` | `Bilinear` | CPU resize sampling algorithm for `TryBeginRun(CapturedFrame)`. |
| `ApplyClassNms` | `bool` | `false` | Whether to run per-class NMS. |
| `NmsIouThreshold` | `float` | `0.5` | NMS IoU threshold. |
| `DisposeSession` | `bool` | `false` | Whether disposing the runner also disposes the passed session. |

### `FrameOnnxRunner`

Non-blocking frame inference entry point. Only one frame runs at a time; while the previous frame is still running, new `TryBeginRun` calls return `false`.

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

Parameters: `session` is the inference session; `profile` is model configuration; `options` may be `null`, which uses default options.

```csharp
bool TryBeginRun(CapturedFrame sourceFrame)
```

Parameter: `sourceFrame` is the current captured frame. The runner currently requires `FramePixelFormat.Rgba32`. If `sourceFrame.RowsBottomUp=true`, it is converted to top-down internally before resize.

Return value: `true` when inference started; `false` when the previous frame is still running.

```csharp
bool TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease preparedInput)
```

Parameter: `preparedInput` is the latest input lease returned by `PreparedFrameOnnxInputBuffer.TryAcquireLatest`. After this method returns `true`, the runner holds the lease until inference finishes. If it returns `false`, the caller is still responsible for disposing the lease.

Return value: `true` when inference started; `false` when the previous frame is still running.

```csharp
bool TryGetResult(out FrameOnnxInferenceResult result)
```

Parameter: `result` receives the completed result.

Return value: `true` when a result was available; `false` when no completed result exists yet.

Properties: `IsRunning` indicates whether inference is active; `Profile` returns the model profile passed to the constructor.

### `FrameOnnxInferenceResult`

Completed result from `FrameOnnxRunner`.

| Property | Type | Description |
| --- | --- | --- |
| `Batch` | `DetectionBatch` | Decoded detection results. |
| `RawOutput` | `float[]` | Raw ONNX Runtime output. |
| `OriginalWidth` / `OriginalHeight` | `int` | Original frame size. |
| `InputWidth` / `InputHeight` | `int` | Model input size. |
| `ResizeDuration` | `TimeSpan` | CPU resize duration; zero on prepared path. |
| `TensorDuration` | `TimeSpan` | RGBA to NCHW duration; zero on prepared path. |
| `InferenceDuration` | `TimeSpan` | ONNX Runtime `Run` duration. |
| `DecodeDuration` | `TimeSpan` | YOLO decode duration. |
| `PreprocessDuration` | `TimeSpan` | `ResizeDuration + TensorDuration`. |
| `TotalActiveDuration` | `TimeSpan` | Total active duration of resize, tensor, ORT, and decode. |
| `ActiveFps` | `double` | `1 / TotalActiveDuration`; excludes rate-limit waiting. |
| `PreprocessBackend` | `InferencePreprocessBackend` | Actual preprocessing path. |
| `Succeeded` | `bool` | Whether the run succeeded. |
| `ErrorMessage` | `string` | Failure reason, empty on success. |

### `OrtNativeLibraryPreloader`

Preloads ONNX Runtime native DLLs by absolute path on Windows.

```csharp
void EnsureLoaded()
```

Return value: none. No-op on non-Windows platforms; throws on Windows if native libraries cannot be found or loaded.

Properties: `LoadedPath` returns the loaded `onnxruntime.dll` path; `LoadWarning` returns warning text from the preload stage.
