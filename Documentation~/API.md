# willkyu ONNX Runtime Inference API Reference

Namespace: `OnnxRuntimeInference`.

## Conventions

- Input tensors are NCHW `float[]` arrays with `[1, 3, height, width]` semantics.
- The core package does not depend on WindowCapture. The normal path uses `OnnxInputFrame`: `FrameOnnxRunner.TryBeginRun(OnnxInputFrame)`.
- The fast path uses `PreparedFrameOnnxInputBuffer` for capture-thread preprocessing, then the main thread calls `TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease)` to run inference directly from the prepared tensor.
- When `com.willkyu.window-capture` is installed, the bridge assembly additionally provides `CapturedFrame` extension methods.
- DirectML is available only in Windows Editor/Player. Initialization failure falls back to CPU and records `InitializationWarning`.

## Enums

### `DetectorRuntimeKind`

| Value | Description |
| --- | --- |
| `OnnxRuntimeCpu` | Uses the CPU Execution Provider. |
| `OnnxRuntimeDirectML` | Prefers the DirectML Execution Provider on Windows and falls back to CPU on failure. |

### `ColorOrder`

| Value | Description |
| --- | --- |
| `Rgb` | R, G, B order. |
| `Bgr` | B, G, R order. |

### `InferencePreprocessBackend`

| Value | Description |
| --- | --- |
| `CpuResizeCpuTensor` | The runner performed CPU resize and wrote NCHW tensor on CPU. |
| `PreparedCpuTensor` | Model input was already prepared; the runner only executed ONNX Runtime and decode. |

### `OnnxFramePixelFormat`

| Value | Description |
| --- | --- |
| `Rgba32` | 4 bytes per pixel, R, G, B, A order. |
| `Bgra32` | 4 bytes per pixel, B, G, R, A order. |
| `Rgb24` | 3 bytes per pixel, R, G, B order. |
| `Bgr24` | 3 bytes per pixel, B, G, R order. |

`OnnxFramePixelFormatUtility.GetBytesPerPixel(format)` returns bytes per pixel. `GetByteCount(width, height, format)` returns the full frame byte count.

### `OnnxResizeAlgorithm`

| Value | Description |
| --- | --- |
| `Nearest` | Nearest-neighbor sampling. Fast and suitable for recognition input. |
| `Bilinear` | Bilinear sampling. Smoother visual output. |

## Data Models

### `OnnxInputFrame`

The ONNX package's own CPU input frame type.

```csharp
new OnnxInputFrame(
    byte[] pixels,
    int width,
    int height,
    OnnxFramePixelFormat format,
    bool rowsBottomUp,
    long frameId,
    DateTime timestampUtc,
    Action<byte[]> releasePixels = null)
```

Parameters: `pixels` is raw pixel data; `width` and `height` are image dimensions; `format` is pixel format; `rowsBottomUp=true` means the first data row is the image bottom row; `frameId` and `timestampUtc` are caller-provided frame metadata; `releasePixels` is optional external-buffer cleanup.

Properties: `Pixels`, `Width`, `Height`, `Format`, `RowsBottomUp`, `FrameId`, `TimestampUtc`.

### `DetectorInputSpec`

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

```csharp
new DetectorClass(int id, string label, float threshold)
```

Parameters: `id` is class id; `label` is display name; `threshold` is the confidence threshold for that class.

Properties: `Id`, `Label`, `Threshold`.

### `DetectorModelProfile`

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

```csharp
new DetectionBatch(
    IReadOnlyList<DetectionResult> detections,
    IReadOnlyDictionary<string, float> classScores)
```

Properties: `Detections` returns detection boxes; `ClassScores` returns the highest confidence for each class label.

## Configuration And Sessions

### `DetectorJsonConfigLoader`

```csharp
DetectorModelProfile LoadFirst(string json)
DetectorModelProfile LoadFirstFromFile(string path)
IReadOnlyList<DetectorModelProfile> LoadAll(string json)
IReadOnlyList<DetectorModelProfile> LoadAllFromFile(string path)
```

Parameters: `json` is config text and cannot be empty; `path` is config file path and cannot be empty.

Return value: `LoadFirst...` returns the first model profile; `LoadAll...` returns every model profile.

### `IOnnxDetectorSession`

```csharp
float[] Run(float[] nchwInput, int width, int height)
```

Parameters: `nchwInput` is a flattened `[1,3,height,width]` array; `width` and `height` are input tensor dimensions.

Return value: flattened ONNX Runtime output tensor. Layout depends on the model.

### `OnnxRuntimeDetectorSession`

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

## Preprocessing

### `TensorPreprocessor`

```csharp
float[] ToNchw(
    OnnxInputFrame frame,
    DetectorInputSpec inputSpec,
    ColorOrder sourceColorOrderOverride = ColorOrder.Rgb)
```

Parameters: `frame` is an ONNX input frame; `inputSpec` is model input spec; `sourceColorOrderOverride` can force RGB/RGBA source data to be interpreted as BGR.

Return value: a new NCHW `float[]`.

```csharp
float[] ToNchw(
    byte[] pixels,
    int width,
    int height,
    OnnxFramePixelFormat format,
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
    OnnxFramePixelFormat format,
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
    OnnxResizeAlgorithm resizeAlgorithm = OnnxResizeAlgorithm.Nearest)
```

Parameters: `inputSpec` is model input spec; `resizeAlgorithm` is the sampling algorithm from original frame to model input size and defaults to `Nearest`.

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

`WriteLease.TryPrepare(OnnxInputFrame sourceFrame)`: `sourceFrame` must be top-down `Rgba32`; returns `true` after resize, preview, and tensor are written and published.

`ReadLease` properties: `Width` and `Height` are model input size; `OriginalWidth` and `OriginalHeight` are original size; `FrameId` and `TimestampUtc` are frame metadata; `PreviewPixels` is model-sized RGBA32 preview; `Tensor` is NCHW `float[]`. `CreatePreviewInputFrame()` wraps preview bytes in an `OnnxInputFrame`.

`ClearReadyFrames()` drops latest inputs that have not been acquired yet, while keeping slots currently held by a `ReadLease`. Use it when switching capture sources so a new source cannot first consume stale prepared tensors from the previous source.

## Inference And Decode

### `FrameOnnxRunnerOptions`

| Property | Type | Default | Description |
| --- | --- | --- | --- |
| `ResizeAlgorithm` | `OnnxResizeAlgorithm` | `Bilinear` | CPU resize sampling algorithm for `TryBeginRun(OnnxInputFrame)`. |
| `ApplyClassNms` | `bool` | `false` | Whether to run per-class NMS. |
| `NmsIouThreshold` | `float` | `0.5` | NMS IoU threshold. |
| `DisposeSession` | `bool` | `false` | Whether disposing the runner also disposes the passed session. |

### `FrameOnnxRunner`

Non-blocking frame inference entry point. Only one frame runs at a time; while the previous frame is still running, new `TryBeginRun` calls return `false`.

```csharp
new FrameOnnxRunner(
    IOnnxDetectorSession session,
    DetectorModelProfile profile,
    OnnxResizeAlgorithm resizeAlgorithm = OnnxResizeAlgorithm.Bilinear,
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

Parameters: `session` is the inference session; `profile` is model configuration; `resizeAlgorithm` or `options.ResizeAlgorithm` controls CPU resize; `applyClassNms` and `nmsIouThreshold` control decode-time NMS; `disposeSession=true` disposes the session when the runner is disposed.

```csharp
bool TryBeginRun(OnnxInputFrame sourceFrame)
```

Parameter: `sourceFrame` is the current input frame. The runner currently requires `OnnxFramePixelFormat.Rgba32`. If `sourceFrame.RowsBottomUp=true`, it is converted to top-down internally before resize.

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

## WindowCapture Bridge Extensions

The bridge assembly is in `Runtime/WindowCaptureBridge`. It is enabled only when `com.willkyu.window-capture` is installed and does not make the core `OnnxRuntimeInference` asmdef depend on WindowCapture.

### `WindowCaptureOnnxExtensions`

The common call shape is `TryBeginRun(CapturedFrame sourceFrame)`, `TryPrepare(CapturedFrame sourceFrame)`, and `CreatePreviewFrame()`. They are implemented as extension methods in the bridge assembly.

```csharp
OnnxInputFrame ToOnnxInputFrame(this CapturedFrame frame)
```

Parameter: `frame` is a WindowCapture `CapturedFrame`.

Return value: an `OnnxInputFrame` wrapper over the same pixel array.

```csharp
bool TryBeginRun(this FrameOnnxRunner runner, CapturedFrame sourceFrame)
```

Parameters: `runner` is the ONNX runner; `sourceFrame` is the captured frame.

Return value: same as `FrameOnnxRunner.TryBeginRun(OnnxInputFrame)`.

```csharp
bool TryPrepare(this PreparedFrameOnnxInputBuffer.WriteLease lease, CapturedFrame sourceFrame)
```

Parameters: `lease` is the write lease; `sourceFrame` must be top-down `Rgba32`.

Return value: same as `WriteLease.TryPrepare(OnnxInputFrame)`.

```csharp
CapturedFrame CreatePreviewFrame(this PreparedFrameOnnxInputBuffer.ReadLease lease)
```

Parameter: `lease` is the latest read lease.

Return value: a `CapturedFrame` wrapper around the prepared preview for debugging or legacy compatibility.

## Native Library Preload

### `OrtNativeLibraryPreloader`

Preloads ONNX Runtime native DLLs by absolute path on Windows.

```csharp
void EnsureLoaded()
```

Return value: none. No-op on non-Windows platforms; throws on Windows if native libraries cannot be found or loaded.

Properties: `LoadedPath` returns the loaded `onnxruntime.dll` path; `DirectMlLoadedPath` returns the loaded `DirectML.dll` path; `LoadWarning` returns warning text from the preload stage.
