# Changelog

## 0.2.0 - 2026-05-08

- Made the runtime package independent from `com.willkyu.window-capture`.
- Replaced capture-package API types with package-local `RgbaFrameInput` and `OnnxResizeAlgorithm`.
- Kept the prepared tensor fast path direct: `FrameOnnxRunner.TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease)` still runs from the prepared `float[]` without frame wrapping.
- Removed optional bridge/package coupling and simplified README/API documentation for standalone publishing.

## 0.1.8 - 2026-05-07

- Added `PreparedFrameOnnxInputBuffer` for the original project style worker-prepared model input path.
- Added `FrameOnnxRunner.TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease)` to run ONNX inference directly from a prepared NCHW tensor.
- Updated the example to use a capture worker by default, with model-sized preview and prepared tensor paths running in parallel with display.

## 0.1.7 - 2026-05-07

- Renamed the frame inference API to `FrameOnnxRunner`, `FrameOnnxRunnerOptions`, and `FrameOnnxInferenceResult`.
- Kept preprocessing CPU-only: resize, NCHW tensor conversion, ONNX Runtime inference, and YOLO decode.
- Removed nonessential experimental and test-only API surface.
