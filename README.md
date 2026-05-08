# willkyu ONNX Runtime Inference

This is a lightweight Unity UPM package for sending CPU image frames or prepared NCHW tensors into ONNX Runtime detector models, with the help of Codex.

The core package can run independently of `com.willkyu.window-capture`. When Window Capture For Unity is also installed, the included `WindowCaptureBridge` assembly automatically provides `CapturedFrame` extension methods so the current capture-to-inference usage stays the same.

## Scope

- Runs ONNX models through `Microsoft.ML.OnnxRuntime.dll`.
- Supports `DetectorRuntimeKind.OnnxRuntimeDirectML` on Windows and falls back to CPU when DirectML initialization fails.
- Bundles x64 `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `DirectML.dll`, and managed `Microsoft.ML.OnnxRuntime.dll`.
- Uses `FrameOnnxRunner` for the image-frame to detection-result path: CPU resize, NCHW tensor conversion, ONNX Runtime, and YOLO end2end decode.
- Uses `PreparedFrameOnnxInputBuffer` for the original project style fast path: a capture thread preprocesses the original frame and writes model-sized preview bytes plus NCHW tensor in one pass; the inference thread only takes the latest tensor lease and calls `TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease)`.
- Uses `DetectorJsonConfigLoader` to read `detectors.json` model input size, tensor color order, normalization, and class thresholds.

This package does not include model files, business UI, window capture, or device capture implementations. Window/device capture is provided by `com.willkyu.window-capture`; this package only adapts it in the bridge layer.

## Install

For ONNX inference only, open `Window > Package Manager` in Unity, click `+`, choose `Add package from git URL...`, then enter:

```text
https://github.com/willkyu/onnx4Unity.git
```

To use it with Window Capture For Unity, also add:

```text
https://github.com/willkyu/WindowCapture4Unity.git
```

You can also edit `Packages/manifest.json` directly:

```json
{
  "dependencies": {
    "com.willkyu.onnxruntime-inference": "https://github.com/willkyu/onnx4Unity.git",
    "com.willkyu.window-capture": "https://github.com/willkyu/WindowCapture4Unity.git"
  }
}
```

`com.willkyu.window-capture` is not a hard dependency of this package. The core API still compiles and runs when it is not installed.

## Standalone Usage

Run inference directly from caller-provided RGBA32 pixels:

```csharp
using System;
using OnnxRuntimeInference;

DetectorModelProfile profile = DetectorJsonConfigLoader.LoadFirstFromFile("Assets/detectors.json");
using var session = new OnnxRuntimeDetectorSession(
    "Assets/yolo26n.onnx",
    DetectorRuntimeKind.OnnxRuntimeDirectML);

var options = new FrameOnnxRunnerOptions
{
    ResizeAlgorithm = OnnxResizeAlgorithm.Bilinear
};

using var runner = new FrameOnnxRunner(session, profile, options);

using var frame = new OnnxInputFrame(
    rgbaPixels,
    width,
    height,
    OnnxFramePixelFormat.Rgba32,
    rowsBottomUp: false,
    frameId: 1,
    timestampUtc: DateTime.UtcNow);

if (runner.TryBeginRun(frame))
{
    // Poll the result later from Update or the caller loop.
}

if (runner.TryGetResult(out FrameOnnxInferenceResult result) && result.Succeeded)
{
    DetectionBatch batch = result.Batch;
}
```

## Using WindowCapture4Unity

After `com.willkyu.window-capture` is installed, the bridge assembly is enabled automatically. Existing `CapturedFrame` calls can stay in place:

```csharp
using OnnxRuntimeInference;
using WindowCapture;

using CapturedFrame frame = frameSource.CaptureOriginal();
if (runner.TryBeginRun(frame))
{
    // The bridge temporarily wraps CapturedFrame as OnnxInputFrame.
}
```

Capture-thread prepared fast path:

```csharp
using var preparedBuffer = new PreparedFrameOnnxInputBuffer(
    profile.InputSpec,
    OnnxResizeAlgorithm.Nearest);

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

`FrameOnnxRunner.ActiveFps` only includes active work: resize, tensor conversion, ORT inference, and decode. It does not include caller-side waiting from `inferenceInterval`. In the prepared fast path, resize and tensor conversion are already completed on the capture thread, so inference results report `ResizeDuration` and `TensorDuration` as zero and `PreprocessBackend` as `PreparedCpuTensor`.

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
