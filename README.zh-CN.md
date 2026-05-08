# willkyu ONNX Runtime Inference

这是一个轻量 Unity UPM 包，用于把 `com.willkyu.window-capture` 捕获到的窗口帧送入 ONNX Runtime 检测模型。

## 功能范围

- 使用 `Microsoft.ML.OnnxRuntime.dll` 执行 ONNX 模型推理。
- Windows 下支持 `DetectorRuntimeKind.OnnxRuntimeDirectML`，DirectML 初始化失败时自动回退 CPU。
- 随包放置 x64 `onnxruntime.dll`、`onnxruntime_providers_shared.dll`、`DirectML.dll` 和托管 `Microsoft.ML.OnnxRuntime.dll`。
- 使用 `FrameOnnxRunner` 完成 `CapturedFrame` 到检测结果的主链路：CPU resize、NCHW tensor、ONNX Runtime、YOLO end2end decode。
- 使用 `PreparedFrameOnnxInputBuffer` 支持原项目式快路径：捕获线程预处理原始帧，一次写入模型尺寸 preview 与 NCHW tensor，主线程只取 latest tensor lease 调用 `TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease)`。
- 使用 `DetectorJsonConfigLoader` 读取 `detectors.json`，配置模型输入尺寸、颜色顺序、归一化和类别阈值。

本包不包含模型文件、业务 UI、窗口/设备捕获实现。捕获与原始/resize 帧能力由 `com.willkyu.window-capture` 提供。

## 安装

在 Unity 中打开 `Window > Package Manager`，点击 `+`，选择 `Add package from git URL...`，先添加窗口捕获包：

```text
https://github.com/<owner>/<repo>.git?path=Packages/com.willkyu.window-capture
```

再添加 ONNX 推理包：

```text
https://github.com/<owner>/<repo>.git?path=Packages/com.willkyu.onnxruntime-inference
```

将 `<owner>/<repo>` 替换为实际存放这些包的仓库。也可以直接编辑 `Packages/manifest.json`：

```json
{
  "dependencies": {
    "com.willkyu.window-capture": "https://github.com/<owner>/<repo>.git?path=Packages/com.willkyu.window-capture",
    "com.willkyu.onnxruntime-inference": "https://github.com/<owner>/<repo>.git?path=Packages/com.willkyu.onnxruntime-inference"
  }
}
```

## 基本用法

直接从捕获帧推理：

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
    // 后续在 Update 中轮询结果。
}

if (runner.TryGetResult(out FrameOnnxInferenceResult result) && result.Succeeded)
{
    DetectionBatch batch = result.Batch;
}
```

捕获线程预处理快路径：

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

`FrameOnnxRunner` 的 `ActiveFps` 只统计实际工作耗时：resize、tensor、ORT 推理和 decode，不包含调用方用 `inferenceInterval` 做限频时的等待时间。prepared 快路径中 resize 和 tensor 已在捕获线程完成，因此推理结果里的 `ResizeDuration` 和 `TensorDuration` 为 0，`PreprocessBackend` 为 `PreparedCpuTensor`。

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
