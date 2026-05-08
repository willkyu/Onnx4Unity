# willkyu ONNX Runtime Inference

This is a lightweight Unity UPM package for sending frames captured by `com.willkyu.window-capture` into ONNX Runtime detector models, with the help of Codex.

## Scope

- Runs ONNX models through `Microsoft.ML.OnnxRuntime.dll`.
- Supports `DetectorRuntimeKind.OnnxRuntimeDirectML` on Windows and falls back to CPU when DirectML initialization fails.
- Bundles x64 `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `DirectML.dll`, and managed `Microsoft.ML.OnnxRuntime.dll`.
- Uses `FrameOnnxRunner` for the main `CapturedFrame` to detection-result path: CPU resize, NCHW tensor conversion, ONNX Runtime, and YOLO end2end decode.
- Uses `PreparedFrameOnnxInputBuffer` for the original project style fast path: a capture thread preprocesses the original frame and writes model-sized preview bytes plus NCHW tensor in one pass; the main thread only takes the latest tensor lease and calls `TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease)`.
- Uses `DetectorJsonConfigLoader` to read `detectors.json` model input size, tensor color order, normalization, and class thresholds.

This package does not include model files, business UI, or window/device capture implementations. Capture and original/resized frame APIs are provided by `com.willkyu.window-capture`.

## Install

In Unity, open `Window > Package Manager`, click `+`, choose `Add package from git URL...`, then add the window capture package first:

```text
https://github.com/<owner>/<repo>.git?path=Packages/com.willkyu.window-capture
```

Then add the ONNX inference package:

```text
https://github.com/<owner>/<repo>.git?path=Packages/com.willkyu.onnxruntime-inference
```

Replace `<owner>/<repo>` with the repository that contains these packages. You can also edit `Packages/manifest.json` directly:

```json
{
  "dependencies": {
    "com.willkyu.window-capture": "https://github.com/<owner>/<repo>.git?path=Packages/com.willkyu.window-capture",
    "com.willkyu.onnxruntime-inference": "https://github.com/<owner>/<repo>.git?path=Packages/com.willkyu.onnxruntime-inference"
  }
}
```

## Basic Usage

Run inference directly from a captured frame:

```csharp
using OnnxRuntimeInference;
using WindowCapture;

DetectorModelProfile profile = DetectorJsonConfigLoader.LoadFirstFromFile("Assets/detectors.json");
using var session = new OnnxRuntimeDetectorSession(
    "Assets/yolo26n.onnx",
    DetectorRuntimeKind.OnnxRuntimeDirectML);

var options = new FrameOnnxRunnerOptions
{
    ResizeAlgorithm = FrameResizeAlgorithm.Bilinear
};

using var runner = new FrameOnnxRunner(session, profile, options);

using CapturedFrame frame = frameSource.Capture();
if (runner.TryBeginRun(frame))
{
    // Poll the result later from Update.
}

if (runner.TryGetResult(out FrameOnnxInferenceResult result) && result.Succeeded)
{
    DetectionBatch batch = result.Batch;
}
```

Capture-thread prepared fast path:

```csharp
using var preparedBuffer = new PreparedFrameOnnxInputBuffer(
    profile.InputSpec,
    FrameResizeAlgorithm.Nearest);

if (preparedBuffer.TryAcquireWrite(out PreparedFrameOnnxInputBuffer.WriteLease write))
{
    using (write)
    using (CapturedFrame frame = frameSource.CaptureOriginal())
        write.TryPrepare(frame);
}

if (preparedBuffer.TryAcquireLatest(out PreparedFrameOnnxInputBuffer.ReadLease preparedInput))
{
    if (!runner.TryBeginRun(preparedInput))
        preparedInput.Dispose();
}
```

`FrameOnnxRunner.ActiveFps` only includes actual active work: resize, tensor conversion, ORT inference, and decode. It does not include caller-side waiting from `inferenceInterval`. In the prepared fast path, resize and tensor conversion are already completed on the capture thread, so inference results report `ResizeDuration` and `TensorDuration` as zero and `PreprocessBackend` as `PreparedCpuTensor`.

## Config Format

Example `detectors.json`:

```json
{
  "models": [
    {
      "detectorId": "APdetector",
      "displayName": "AutoPoke Gen3 Detector",
      "model": {
        "onnxModelName": "yolo26n.onnx",
        "inputWidth": 480,
        "inputHeight": 320,
        "tensorColorOrder": "rgb",
        "normalizeToUnitRange": true
      },
      "classes": [
        { "label": "Dialogue", "threshold": 0.5 },
        { "label": "Next", "threshold": 0.35 }
      ]
    }
  ]
}
```

## Runtime Notes

When creating a DirectML session, the package first loads native DLLs from `Runtime/Plugins/Windows/x86_64`. If `ActualRuntimeKind` remains `OnnxRuntimeDirectML`, the DirectML session was created successfully. `InitializationWarning` only needs attention when the runtime fell back to CPU or native library loading has a risk.

If DirectML initialization fails, check:

- Whether `onnxruntime.dll` and `DirectML.dll` are x64.
- Whether the GPU driver and Windows version support the bundled DirectML runtime.
- Whether the model operators are supported by the current DirectML provider.

## Documentation

Main classes, parameters, and return values:

```text
Documentation~/API.md
```

Chinese version:

```text
Documentation~/API.zh-CN.md
```
