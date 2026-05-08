# ONNX Runtime Inference for Unity

Standalone Unity UPM package for ONNX Runtime detector inference. It does not depend on the window capture package; callers provide RGBA32 frame bytes or an already prepared NCHW tensor. Codex was used to help develop this package.

## Scope

- Loads ONNX Runtime managed/native libraries bundled in `Runtime/`.
- Supports CPU and DirectML runtimes on Windows, with CPU fallback when DirectML initialization fails.
- Reads detector profiles from `detectors.json`.
- Converts RGBA32 frames to NCHW tensors with optional resize.
- Provides a prepared tensor fast path so capture/preprocess work can happen off the inference caller.
- Decodes YOLO end2end `[1,300,6]` outputs into boxes and class scores.

The package intentionally does not include model files, capture code, or UI.

## Install

In Unity, open `Window > Package Manager`, click `+`, choose `Add package from git URL...`, then add:

```text
https://github.com/willkyu/Onnx4Unity.git
```

## Basic Usage

```csharp
using OnnxRuntimeInference;

DetectorModelProfile profile = DetectorJsonConfigLoader.LoadFirstFromFile("Assets/detectors.json");
using var session = new OnnxRuntimeDetectorSession("Assets/yolo26n.onnx", DetectorRuntimeKind.OnnxRuntimeDirectML);

using var runner = new FrameOnnxRunner(session, profile, new FrameOnnxRunnerOptions
{
    ResizeAlgorithm = OnnxResizeAlgorithm.Bilinear
});

var frame = new RgbaFrameInput(rgbaPixels, width, height, rowsBottomUp: false, frameId, DateTime.UtcNow);
if (runner.TryBeginRun(frame))
{
    // Poll later from Update.
}

if (runner.TryGetResult(out FrameOnnxInferenceResult result) && result.Succeeded)
{
    DetectionBatch batch = result.Batch;
}
```

## Prepared Fast Path

Use this when another thread already captures frames. The write lease resizes once, writes model-sized preview bytes, and writes NCHW tensor in one pass.

```csharp
using var preparedBuffer = new PreparedFrameOnnxInputBuffer(profile.InputSpec, OnnxResizeAlgorithm.Nearest);

if (preparedBuffer.TryAcquireWrite(out PreparedFrameOnnxInputBuffer.WriteLease write))
{
    using (write)
        write.TryPrepare(new RgbaFrameInput(rgbaPixels, width, height, rowsBottomUp: false, frameId, timestampUtc));
}

if (preparedBuffer.TryAcquireLatest(out PreparedFrameOnnxInputBuffer.ReadLease input))
{
    if (!runner.TryBeginRun(input))
        input.Dispose();
}
```

`FrameOnnxRunner.ActiveFps` measures active work only: resize, tensor conversion, ORT inference, and decode. It does not include caller-side rate limiting. Prepared tensor results report zero resize/tensor duration and `PreprocessBackend.PreparedCpuTensor`.

## Window Capture Integration

When using `com.willkyu.window-capture`, keep the dependency in your project or sample, not in this package:

```csharp
static RgbaFrameInput ToOnnxInput(WindowCapture.CapturedFrame frame)
{
    if (frame.Format != WindowCapture.FramePixelFormat.Rgba32)
        throw new InvalidOperationException("ONNX input requires RGBA32.");

    return new RgbaFrameInput(frame.Pixels, frame.Width, frame.Height, frame.RowsBottomUp, frame.FrameId, frame.TimestampUtc);
}
```

## Documentation

See `Documentation~/API.md` and `Documentation~/API.zh-CN.md`.
