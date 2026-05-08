# ONNX Runtime Inference for Unity

这是一个独立的 Unity UPM 包，用于 ONNX Runtime 检测模型推理。它不依赖窗口捕获包；调用方只需要提供 RGBA32 帧数据，或提供已经准备好的 NCHW tensor。借助 Codex 开发。

## 功能范围

- 加载随包放置的 ONNX Runtime 托管库和原生库。
- Windows 下支持 CPU 与 DirectML，DirectML 初始化失败时回退 CPU。
- 从 `detectors.json` 读取检测模型配置。
- 将 RGBA32 帧 resize 并写入 NCHW tensor。
- 提供 prepared tensor 快路径，方便捕获/预处理线程提前准备模型输入。
- 解码 YOLO end2end `[1,300,6]` 输出为检测框和类别分数。

本包不包含模型文件、捕获代码、业务 UI。

## 安装

在 Unity 中打开 `Window > Package Manager`，点击 `+`，选择 `Add package from git URL...`，添加：

```text
https://github.com/willkyu/Onnx4Unity.git
```

## 基本用法

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
    // 后续在 Update 中轮询结果。
}

if (runner.TryGetResult(out FrameOnnxInferenceResult result) && result.Succeeded)
{
    DetectionBatch batch = result.Batch;
}
```

## Prepared 快路径

适用于其它线程已经在捕获帧的场景。写入 lease 会一次完成 resize、模型尺寸 preview 写入和 NCHW tensor 写入。

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

`FrameOnnxRunner.ActiveFps` 只统计实际工作耗时：resize、tensor、ORT 推理和 decode，不包含调用方限频等待。Prepared tensor 结果中 resize/tensor 耗时为 0，`PreprocessBackend` 为 `PreparedCpuTensor`。

## 与窗口捕获包配合

如果使用 `com.willkyu.window-capture`，依赖应放在项目或示例中，而不是放在本包中：

```csharp
static RgbaFrameInput ToRgbaFrameInput(WindowCapture.CapturedFrame frame)
{
    if (frame.Format != WindowCapture.FramePixelFormat.Rgba32)
        throw new InvalidOperationException("ONNX input requires RGBA32.");

    return new RgbaFrameInput(frame.Pixels, frame.Width, frame.Height, frame.RowsBottomUp, frame.FrameId, frame.TimestampUtc);
}
```

## 文档

见 `Documentation~/API.zh-CN.md` 和 `Documentation~/API.md`。
