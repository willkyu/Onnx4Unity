# ONNX Runtime Inference API

Namespace: `OnnxRuntimeInference`.

## Input Conventions

- Frame input is RGBA32 byte data wrapped by `RgbaFrameInput`.
- Tensor input is flattened NCHW `float[]` with `[1, 3, height, width]` semantics.
- The prepared path keeps resize/tensor work outside the inference call and passes `PreparedFrameOnnxInputBuffer.ReadLease` directly to `FrameOnnxRunner`.

## Core Types

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

Represents one RGBA32 frame. `pixels` must contain at least `width * height * 4` bytes. `rowsBottomUp=true` means row 0 in memory is the bottom row; the runner flips it before resize/tensor conversion. Properties mirror the constructor parameters.

### `OnnxResizeAlgorithm`

`Nearest` or `Bilinear`.

### `DetectorRuntimeKind`

`OnnxRuntimeCpu` uses CPU. `OnnxRuntimeDirectML` uses DirectML on Windows and falls back to CPU if initialization fails.

### `DetectorInputSpec`

```csharp
new DetectorInputSpec(int width, int height, ColorOrder tensorColorOrder, bool normalizeToUnitRange)
```

Defines model input size, RGB/BGR tensor order, and whether byte values are normalized to `0..1`.

### `DetectorClass`, `DetectorModelProfile`, `DetectionResult`, `DetectionBatch`

These are simple immutable data contracts for model metadata and decoded detections.

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

Loads detector profiles from the package JSON format used by `detectors.json`.

## Preprocessing

### `TensorPreprocessor`

```csharp
float[] ToNchw(RgbaFrameInput frame, DetectorInputSpec inputSpec)
float[] ToNchw(byte[] rgba32, int width, int height, bool rowsBottomUp, DetectorInputSpec inputSpec)
void WriteNchw(byte[] rgba32, int width, int height, bool rowsBottomUp, DetectorInputSpec inputSpec, float[] tensor)
```

Writes RGBA32 input to NCHW float tensor. `WriteNchw` writes into a caller-provided buffer and avoids allocation.

### `PreparedFrameOnnxInputBuffer`

```csharp
new PreparedFrameOnnxInputBuffer(DetectorInputSpec inputSpec, OnnxResizeAlgorithm resizeAlgorithm = OnnxResizeAlgorithm.Nearest)
```

Double-buffered prepared model input. A writer turns a top-down RGBA32 source frame into both model-sized preview bytes and NCHW tensor.

```csharp
bool TryAcquireWrite(out PreparedFrameOnnxInputBuffer.WriteLease lease)
bool TryAcquireLatest(out PreparedFrameOnnxInputBuffer.ReadLease lease)
bool TryCopyLatestPreviewPixels(byte[] destination, out int width, out int height, out int originalWidth, out int originalHeight, out long frameId, out DateTime timestampUtc)
```

`WriteLease.TryPrepare(RgbaFrameInput sourceFrame)` returns `true` when the frame was top-down RGBA32 and was published. Bottom-up frames return `false`; flip before writing if needed.

`ReadLease` exposes `Width`, `Height`, `OriginalWidth`, `OriginalHeight`, `FrameId`, `TimestampUtc`, `PreviewPixels`, and `Tensor`. `CreatePreviewFrame()` returns an `RgbaFrameInput` over the preview bytes.

## Inference

### `IOnnxDetectorSession`

```csharp
float[] Run(float[] nchwInput, int width, int height)
```

Runs one model input and returns flattened output.

### `OnnxRuntimeDetectorSession`

```csharp
new OnnxRuntimeDetectorSession(
    string modelPath,
    DetectorRuntimeKind runtimeKind,
    string preferredInputName = "images",
    string preferredOutputName = "output0")
```

Wraps `Microsoft.ML.OnnxRuntime.InferenceSession`. Properties: `RequestedRuntimeKind`, `ActualRuntimeKind`, `InitializationWarning`, `InputName`, `OutputName`.

### `FrameOnnxRunnerOptions`

| Property | Type | Default | Description |
| --- | --- | --- | --- |
| `ResizeAlgorithm` | `OnnxResizeAlgorithm` | `Bilinear` | Resize algorithm for `TryBeginRun(RgbaFrameInput)`. |
| `ApplyClassNms` | `bool` | `false` | Enables per-class NMS. |
| `NmsIouThreshold` | `float` | `0.5` | NMS IoU threshold. |
| `DisposeSession` | `bool` | `false` | Disposes the session when the runner is disposed. |

### `FrameOnnxRunner`

```csharp
new FrameOnnxRunner(IOnnxDetectorSession session, DetectorModelProfile profile, OnnxResizeAlgorithm resizeAlgorithm = OnnxResizeAlgorithm.Bilinear, bool applyClassNms = false, float nmsIouThreshold = 0.5f, bool disposeSession = false)
new FrameOnnxRunner(IOnnxDetectorSession session, DetectorModelProfile profile, FrameOnnxRunnerOptions options)
bool TryBeginRun(RgbaFrameInput sourceFrame)
bool TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease preparedInput)
bool TryGetResult(out FrameOnnxInferenceResult result)
```

Runs at most one frame at a time. `TryBeginRun` returns `false` while another frame is in flight. If `TryBeginRun(ReadLease)` returns `true`, the runner owns and disposes that lease.

### `FrameOnnxInferenceResult`

Important properties: `Batch`, `RawOutput`, `OriginalWidth`, `OriginalHeight`, `InputWidth`, `InputHeight`, `SourceFrameId`, `SourceTimestampUtc`, `ResizeDuration`, `TensorDuration`, `InferenceDuration`, `DecodeDuration`, `PreprocessDuration`, `TotalActiveDuration`, `ActiveFps`, `PreprocessBackend`, `Succeeded`, `ErrorMessage`.

`ActiveFps` excludes caller-side rate limiting or waiting.

### `YoloEnd2EndDecoder`

```csharp
DetectionBatch Decode(float[] output, DetectorModelProfile profile, int originalWidth, int originalHeight, bool applyClassNms = false, float nmsIouThreshold = 0.5f)
DetectionBatch Decode(float[] output, DetectorInputSpec inputSpec, IReadOnlyList<DetectorClass> classes, int originalWidth, int originalHeight, bool applyClassNms = false, float nmsIouThreshold = 0.5f)
```

Decodes YOLO end2end output rows `[x1, y1, x2, y2, confidence, classId]`.

### `OrtNativeLibraryPreloader`

```csharp
void EnsureLoaded()
```

Preloads bundled ONNX Runtime native DLLs on Windows. `LoadedPath` and `LoadWarning` expose diagnostics.
