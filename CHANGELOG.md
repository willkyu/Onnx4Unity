# Changelog

## 0.1.10 - 2026-05-08

- Removed per-frame `OnnxInputFrame` wrapper allocation from the WindowCapture bridge hot path.
- Kept the public standalone and `CapturedFrame` bridge APIs unchanged.

## 0.1.9 - 2026-05-08

- Split the core ONNX package away from the WindowCapture package dependency.
- Added `OnnxInputFrame`, `OnnxFramePixelFormat`, and `OnnxResizeAlgorithm` for standalone CPU-frame inference.
- Added an optional `WindowCaptureBridge` assembly so `CapturedFrame` usage still works when Window Capture For Unity is installed.
- Updated Chinese and English README/API documentation for standalone and bridge usage.

## 0.1.8 - 2026-05-07

- Added `PreparedFrameOnnxInputBuffer` for the original project style worker-prepared model input path.
- Added `FrameOnnxRunner.TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease)` to run ONNX inference directly from a prepared NCHW tensor.
- Updated the example to use a capture worker by default, with model-sized preview and prepared tensor paths running in parallel with display.

## 0.1.7 - 2026-05-07

- Renamed the frame inference API to `FrameOnnxRunner`, `FrameOnnxRunnerOptions`, and `FrameOnnxInferenceResult`.
- Kept preprocessing CPU-only: resize, NCHW tensor conversion, ONNX Runtime inference, and YOLO decode.
- Removed nonessential experimental and test-only API surface.
