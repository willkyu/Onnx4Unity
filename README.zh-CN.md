# willkyu ONNX Runtime Inference

这是一个轻量 Unity UPM 包，用于把 CPU 图像帧或已准备好的 NCHW tensor 送入 ONNX Runtime 检测模型。借助 Codex 开发。

核心包可以独立于 `com.willkyu.window-capture` 使用；当项目同时安装 Window Capture For Unity 时，包内的 `WindowCaptureBridge` 会自动提供 `CapturedFrame` 扩展方法，让现有调用方式保持不变。

## 功能范围

- 使用 `Microsoft.ML.OnnxRuntime.dll` 执行 ONNX 模型推理。
- Windows 下支持 `DetectorRuntimeKind.OnnxRuntimeDirectML`，DirectML 初始化失败时自动回退 CPU。
- 随包放置 x64 `onnxruntime.dll`、`onnxruntime_providers_shared.dll`、`DirectML.dll` 和托管 `Microsoft.ML.OnnxRuntime.dll`。
- 使用 `FrameOnnxRunner` 完成图像帧到检测结果的主链路：CPU resize、NCHW tensor、ONNX Runtime、YOLO end2end decode。
- 使用 `PreparedFrameOnnxInputBuffer` 支持原项目式快路径：采集线程预处理原始帧，一次写入模型尺寸 preview 与 NCHW tensor，推理线程只取 latest tensor lease 调用 `TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease)`。
- 使用 `DetectorJsonConfigLoader` 读取 `detectors.json`，配置模型输入尺寸、颜色顺序、归一化和类别阈值。

本包不包含模型文件、业务 UI、窗口捕获或设备捕获实现。窗口/设备采集由 `com.willkyu.window-capture` 提供，ONNX 包只在桥接层适配它。

## 安装

只使用 ONNX 推理能力时，在 Unity 中打开 `Window > Package Manager`，点击 `+`，选择 `Add package from git URL...`，输入：

```text
https://github.com/willkyu/onnx4Unity.git
```

如果需要和 Window Capture For Unity 一起使用，再添加窗口捕获包：

```text
https://github.com/willkyu/WindowCapture4Unity.git
```

也可以直接编辑 `Packages/manifest.json`：

```json
{
  "dependencies": {
    "com.willkyu.onnxruntime-inference": "https://github.com/willkyu/onnx4Unity.git",
    "com.willkyu.window-capture": "https://github.com/willkyu/WindowCapture4Unity.git"
  }
}
```

`com.willkyu.window-capture` 不是 ONNX 包的强依赖；未安装时核心 API 仍可正常编译和运行。

## 独立用法

直接从调用方提供的 RGBA32 帧推理：

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
    // 后续在 Update 或调用方循环中轮询结果。
}

if (runner.TryGetResult(out FrameOnnxInferenceResult result) && result.Succeeded)
{
    DetectionBatch batch = result.Batch;
}
```

## 与 WindowCapture4Unity 一起使用

安装 `com.willkyu.window-capture` 后，桥接程序集会自动启用。已有的 `CapturedFrame` 调用可以继续使用：

```csharp
using OnnxRuntimeInference;
using WindowCapture;

using CapturedFrame frame = frameSource.CaptureOriginal();
if (runner.TryBeginRun(frame))
{
    // bridge 会把 CapturedFrame 临时包装为 OnnxInputFrame。
}
```

采集线程预处理快路径：

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

`FrameOnnxRunner.ActiveFps` 只统计实际工作耗时：resize、tensor、ORT 推理和 decode，不包含调用方用 `inferenceInterval` 做限频时的等待时间。prepared 快路径中 resize 和 tensor 已在采集线程完成，因此推理结果里的 `ResizeDuration` 和 `TensorDuration` 为 0，`PreprocessBackend` 为 `PreparedCpuTensor`。

## 配置格式

`detectors.json` 示例：

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

## 运行时说明

DirectML 会话创建时会先加载本包 `Runtime/Plugins/Windows/x86_64` 下的原生 DLL。若 `ActualRuntimeKind` 仍为 `OnnxRuntimeDirectML`，说明 DirectML session 已成功创建；`InitializationWarning` 只有在回退 CPU 或原生库加载存在风险时才需要处理。

如果 DirectML 初始化失败，请优先检查：

- `onnxruntime.dll` / `DirectML.dll` 是否为 x64。
- 显卡驱动和 Windows 版本是否支持当前 DirectML。
- 模型算子是否被当前 DirectML provider 支持。

## 文档

主要类、参数和返回值见：

```text
Documentation~/API.zh-CN.md
```
