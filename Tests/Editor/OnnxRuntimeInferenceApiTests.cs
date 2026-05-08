using System;
using System.IO;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using WindowCapture;

namespace OnnxRuntimeInference.Tests
{
    public sealed class OnnxRuntimeInferenceApiTests
    {
        [Test]
        public void TensorPreprocessorWritesNchwFromCapturedFrame()
        {
            var frame = new CapturedFrame(
                new byte[]
                {
                    10, 20, 30, 255,
                    40, 50, 60, 255
                },
                2,
                1,
                FramePixelFormat.Rgba32,
                rowsBottomUp: false,
                frameId: 1,
                timestampUtc: DateTime.UtcNow);
            var spec = new DetectorInputSpec(2, 1, ColorOrder.Rgb, normalizeToUnitRange: false);

            float[] tensor = TensorPreprocessor.ToNchw(frame, spec);

            CollectionAssert.AreEqual(new[] { 10f, 40f, 20f, 50f, 30f, 60f }, tensor);
        }

        [Test]
        public void TestSessionInvokesInjectedInference()
        {
            var session = new TestOnnxDetectorSession((input, width, height) =>
                new[] { input[0], width, height });

            float[] output = session.Run(new[] { 7f }, 8, 9);

            CollectionAssert.AreEqual(new[] { 7f, 8f, 9f }, output);
        }

        [Test]
        public void DetectorJsonConfigLoaderParsesYoloProfile()
        {
            string json = @"{
  ""models"": [
    {
      ""detectorId"": ""APdetector"",
      ""displayName"": ""AutoPoke Gen3 Detector"",
      ""model"": {
        ""onnxModelName"": ""yolo26n.onnx"",
        ""inputWidth"": 480,
        ""inputHeight"": 320,
        ""tensorColorOrder"": ""rgb"",
        ""normalizeToUnitRange"": true
      },
      ""classes"": [
        { ""label"": ""Dialogue"", ""threshold"": 0.5 },
        { ""label"": ""Next"", ""threshold"": 0.35 }
      ]
    }
  ]
}";

            DetectorModelProfile profile = DetectorJsonConfigLoader.LoadFirst(json);

            Assert.AreEqual("APdetector", profile.DetectorId);
            Assert.AreEqual("AutoPoke Gen3 Detector", profile.DisplayName);
            Assert.AreEqual("yolo26n.onnx", profile.OnnxModelName);
            Assert.AreEqual(480, profile.InputSpec.Width);
            Assert.AreEqual(320, profile.InputSpec.Height);
            Assert.AreEqual(ColorOrder.Rgb, profile.InputSpec.TensorColorOrder);
            Assert.IsTrue(profile.InputSpec.NormalizeToUnitRange);
            Assert.AreEqual(2, profile.Classes.Count);
            Assert.AreEqual("Dialogue", profile.Classes[0].Label);
            Assert.AreEqual(0.35f, profile.Classes[1].Threshold);
        }

        [Test]
        public void YoloEnd2EndDecoderScalesFiltersAndReportsClassScores()
        {
            var inputSpec = new DetectorInputSpec(100, 50, ColorOrder.Rgb, normalizeToUnitRange: true);
            var classes = new[]
            {
                new DetectorClass(0, "A", 0.5f),
                new DetectorClass(1, "B", 0.5f)
            };
            var output = new float[YoloEnd2EndDecoder.ExpectedRowCount * YoloEnd2EndDecoder.ValuesPerRow];

            WriteYoloRow(output, 0, 10f, 5f, 30f, 25f, 0.9f, 1f);
            WriteYoloRow(output, 1, 20f, 10f, 40f, 20f, 0.4f, 0f);

            DetectionBatch batch = YoloEnd2EndDecoder.Decode(
                output,
                inputSpec,
                classes,
                originalWidth: 200,
                originalHeight: 100);

            Assert.AreEqual(1, batch.Detections.Count);
            DetectionResult detection = batch.Detections[0];
            Assert.AreEqual(1, detection.ClassId);
            Assert.AreEqual("B", detection.Label);
            Assert.AreEqual(0.9f, detection.Confidence);
            Assert.AreEqual(20f, detection.X1);
            Assert.AreEqual(10f, detection.Y1);
            Assert.AreEqual(60f, detection.X2);
            Assert.AreEqual(50f, detection.Y2);
            Assert.AreEqual(0.4f, batch.ClassScores["A"]);
            Assert.AreEqual(0.9f, batch.ClassScores["B"]);
        }

        [Test]
        public void FrameRunnerExposesCapturedFramePipeline()
        {
            Type runnerType = typeof(FrameOnnxRunner);
            MethodInfo begin = runnerType.GetMethod(
                "TryBeginRun",
                new[] { typeof(CapturedFrame) });
            MethodInfo result = runnerType.GetMethod("TryGetResult");

            Assert.IsNotNull(begin, "Runner must accept a captured CPU frame directly.");
            Assert.IsNotNull(result, "Runner must expose non-blocking result polling.");
            Assert.IsTrue(typeof(IDisposable).IsAssignableFrom(runnerType));

            Assert.IsNotNull(typeof(FrameOnnxInferenceResult).GetProperty("ResizeDuration"));
            Assert.IsNotNull(typeof(FrameOnnxInferenceResult).GetProperty("TensorDuration"));
            Assert.IsNotNull(typeof(FrameOnnxInferenceResult).GetProperty("InferenceDuration"));
            Assert.IsNotNull(typeof(FrameOnnxInferenceResult).GetProperty("DecodeDuration"));
            Assert.IsNotNull(typeof(FrameOnnxInferenceResult).GetProperty("ActiveFps"));
            Assert.IsNotNull(typeof(FrameOnnxInferenceResult).GetProperty("Batch"));
        }

        [Test]
        public void FrameRunnerUsesCpuResizeOnly()
        {
            Type runnerType = typeof(FrameOnnxRunner);
            MethodInfo beginFromFrame = runnerType.GetMethod(
                "TryBeginRun",
                new[] { typeof(CapturedFrame) });

            Assert.IsNotNull(beginFromFrame, "Runner must accept a captured CPU frame.");
            Assert.AreEqual(FrameResizeAlgorithm.Bilinear, new FrameOnnxRunnerOptions().ResizeAlgorithm);
            Assert.IsNotNull(typeof(FrameOnnxInferenceResult).GetProperty("PreprocessBackend"));
        }

        [Test]
        public void PreparedInputBufferFusesNearestResizePreviewAndTensor()
        {
            var spec = new DetectorInputSpec(1, 1, ColorOrder.Rgb, normalizeToUnitRange: false);
            using var buffer = new PreparedFrameOnnxInputBuffer(spec, FrameResizeAlgorithm.Nearest);
            using var frame = new CapturedFrame(
                new byte[]
                {
                    10, 20, 30, 255,
                    40, 50, 60, 255,
                    70, 80, 90, 255,
                    100, 110, 120, 255
                },
                2,
                2,
                FramePixelFormat.Rgba32,
                rowsBottomUp: false,
                frameId: 7,
                timestampUtc: new DateTime(2026, 5, 7, 0, 0, 0, DateTimeKind.Utc));

            Assert.IsTrue(buffer.TryAcquireWrite(out PreparedFrameOnnxInputBuffer.WriteLease write));
            using (write)
                Assert.IsTrue(write.TryPrepare(frame));

            Assert.IsTrue(buffer.TryAcquireLatest(out PreparedFrameOnnxInputBuffer.ReadLease read));
            using (read)
            {
                Assert.AreEqual(1, read.Width);
                Assert.AreEqual(1, read.Height);
                Assert.AreEqual(2, read.OriginalWidth);
                Assert.AreEqual(2, read.OriginalHeight);
                Assert.AreEqual(7, read.FrameId);
                CollectionAssert.AreEqual(new byte[] { 10, 20, 30, 255 }, read.PreviewPixels);
                CollectionAssert.AreEqual(new[] { 10f, 20f, 30f }, read.Tensor);

                using CapturedFrame preview = read.CreatePreviewFrame();
                Assert.AreEqual(1, preview.Width);
                Assert.AreEqual(1, preview.Height);
                CollectionAssert.AreEqual(new byte[] { 10, 20, 30, 255 }, preview.Pixels);
            }
        }

        [Test]
        public void RunnerCanUseWorkerPreparedInputWithoutResizeOrTensorWork()
        {
            float[] observedInput = null;
            var session = new TestOnnxDetectorSession((input, width, height) =>
            {
                observedInput = input;
                return new float[YoloEnd2EndDecoder.ExpectedRowCount * YoloEnd2EndDecoder.ValuesPerRow];
            });
            var profile = new DetectorModelProfile(
                "test",
                "Test",
                "test.onnx",
                new DetectorInputSpec(1, 1, ColorOrder.Rgb, normalizeToUnitRange: false),
                new[] { new DetectorClass(0, "A", 1f) });
            using var buffer = new PreparedFrameOnnxInputBuffer(profile.InputSpec, FrameResizeAlgorithm.Nearest);
            using var frame = new CapturedFrame(
                new byte[]
                {
                    10, 20, 30, 255,
                    20, 30, 40, 255,
                    30, 40, 50, 255,
                    40, 50, 60, 255
                },
                2,
                2,
                FramePixelFormat.Rgba32,
                rowsBottomUp: false,
                frameId: 8,
                timestampUtc: DateTime.UtcNow);

            Assert.IsTrue(buffer.TryAcquireWrite(out PreparedFrameOnnxInputBuffer.WriteLease write));
            using (write)
                Assert.IsTrue(write.TryPrepare(frame));
            Assert.IsTrue(buffer.TryAcquireLatest(out PreparedFrameOnnxInputBuffer.ReadLease preparedInput));
            float[] expectedTensor = preparedInput.Tensor;

            using var runner = new FrameOnnxRunner(session, profile, new FrameOnnxRunnerOptions());

            Assert.IsTrue(runner.TryBeginRun(preparedInput));
            Assert.IsTrue(WaitForResult(runner, out FrameOnnxInferenceResult result));

            Assert.IsTrue(result.Succeeded, result.ErrorMessage);
            Assert.AreSame(expectedTensor, observedInput);
            Assert.AreEqual(TimeSpan.Zero, result.ResizeDuration);
            Assert.AreEqual(TimeSpan.Zero, result.TensorDuration);
            Assert.AreEqual(InferencePreprocessBackend.PreparedCpuTensor, result.PreprocessBackend);
            Assert.AreEqual(2, result.OriginalWidth);
            Assert.AreEqual(2, result.OriginalHeight);
        }

        [Test]
        public void PackageDocsAndExampleDescribeCurrentFramePipeline()
        {
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string packageRoot = Path.Combine(projectRoot, "Packages", "com.willkyu.onnxruntime-inference");
            string examplePath = Path.Combine(projectRoot, "Assets", "Scripts", "WindowCaptureExample.cs");
            string combined =
                File.ReadAllText(Path.Combine(packageRoot, "README.zh-CN.md")) + "\n" +
                File.ReadAllText(Path.Combine(packageRoot, "Documentation~", "API.zh-CN.md")) + "\n" +
                File.ReadAllText(examplePath);

            StringAssert.Contains("FrameOnnxRunner", combined);
            StringAssert.Contains("TryBeginRun(CapturedFrame", combined);
            StringAssert.Contains("PreparedFrameOnnxInputBuffer", combined);
            StringAssert.Contains("TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease", combined);
            StringAssert.Contains("捕获线程预处理", combined);
            StringAssert.Contains("CPU resize", combined);
        }

        [Test]
        public void RuntimePackageContainsOnlyMainRuntimeScripts()
        {
            string packageRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "com.willkyu.onnxruntime-inference"));
            string runtimeRoot = Path.Combine(packageRoot, "Runtime");
            string[] expected =
            {
                "DetectionContracts.cs",
                "DetectorJsonConfigLoader.cs",
                "FrameOnnxInferenceResult.cs",
                "FrameOnnxRunner.cs",
                "FrameOnnxRunnerOptions.cs",
                "IOnnxDetectorSession.cs",
                "OnnxRuntimeDetectorSession.cs",
                "OrtNativeLibraryPreloader.cs",
                "PreparedFrameOnnxInputBuffer.cs",
                "TensorPreprocessor.cs",
                "YoloEnd2EndDecoder.cs"
            };

            string[] actual = Directory.GetFiles(runtimeRoot, "*.cs", SearchOption.TopDirectoryOnly)
                .Select(Path.GetFileName)
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();
            Array.Sort(expected, StringComparer.Ordinal);

            CollectionAssert.AreEqual(expected, actual);
        }

        [Test]
        public void DirectMlSessionCreationDisablesUnsupportedOrtFeatures()
        {
            string packageRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "com.willkyu.onnxruntime-inference"));
            string source = File.ReadAllText(Path.Combine(packageRoot, "Runtime", "OnnxRuntimeDetectorSession.cs"));

            StringAssert.Contains("EnableMemoryPattern = false", source);
            StringAssert.Contains("ExecutionMode.ORT_SEQUENTIAL", source);
            StringAssert.Contains("AppendExecutionProvider_DML", source);
        }

        [Test]
        public void DirectMlRedistributableIsBundledForDeterministicProviderInitialization()
        {
            string packageRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "com.willkyu.onnxruntime-inference"));
            string directMlPath = Path.Combine(packageRoot, "Runtime", "Plugins", "Windows", "x86_64", "DirectML.dll");

            Assert.IsTrue(File.Exists(directMlPath), "DirectML.dll must be bundled beside onnxruntime.dll.");
            Assert.AreEqual(0x8664, ReadPeMachine(directMlPath), "Bundled DirectML.dll must be x64.");
        }

        [Test]
        public void DirectMlPreloaderDoesNotWarnWhenEditorAlreadyLoadedDirectMl()
        {
            string packageRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Packages", "com.willkyu.onnxruntime-inference"));
            string source = File.ReadAllText(Path.Combine(packageRoot, "Runtime", "OrtNativeLibraryPreloader.cs"));

            StringAssert.DoesNotContain("Restart the Unity Editor to use the bundled DirectML redistributable.", source);
        }

        [Test]
        public void RunnerCanUseCpuResizeFallbackFromCapturedFrame()
        {
            float[] observedInput = null;
            var session = new TestOnnxDetectorSession((input, width, height) =>
            {
                observedInput = (float[])input.Clone();
                return new float[YoloEnd2EndDecoder.ExpectedRowCount * YoloEnd2EndDecoder.ValuesPerRow];
            });
            var profile = new DetectorModelProfile(
                "test",
                "Test",
                "test.onnx",
                new DetectorInputSpec(1, 1, ColorOrder.Rgb, normalizeToUnitRange: false),
                new[] { new DetectorClass(0, "A", 1f) });
            var options = new FrameOnnxRunnerOptions
            {
                ResizeAlgorithm = FrameResizeAlgorithm.Bilinear
            };
            using var runner = new FrameOnnxRunner(session, profile, options);
            using var frame = new CapturedFrame(
                new byte[]
                {
                    10, 20, 30, 255,
                    20, 30, 40, 255,
                    30, 40, 50, 255,
                    40, 50, 60, 255
                },
                2,
                2,
                FramePixelFormat.Rgba32,
                rowsBottomUp: false,
                frameId: 1,
                timestampUtc: DateTime.UtcNow);

            Assert.IsTrue(runner.TryBeginRun(frame));
            Assert.IsTrue(WaitForResult(runner, out FrameOnnxInferenceResult result));

            Assert.IsTrue(result.Succeeded, result.ErrorMessage);
            Assert.Greater(result.ResizeDuration, TimeSpan.Zero);
            Assert.AreEqual(InferencePreprocessBackend.CpuResizeCpuTensor, result.PreprocessBackend);
            CollectionAssert.AreEqual(new[] { 25f, 35f, 45f }, observedInput);
        }

        private static void WriteYoloRow(
            float[] output,
            int row,
            float x1,
            float y1,
            float x2,
            float y2,
            float confidence,
            float classId)
        {
            int baseIndex = row * YoloEnd2EndDecoder.ValuesPerRow;
            output[baseIndex + 0] = x1;
            output[baseIndex + 1] = y1;
            output[baseIndex + 2] = x2;
            output[baseIndex + 3] = y2;
            output[baseIndex + 4] = confidence;
            output[baseIndex + 5] = classId;
        }

        private static ushort ReadPeMachine(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            int peOffset = BitConverter.ToInt32(bytes, 0x3C);
            return BitConverter.ToUInt16(bytes, peOffset + 4);
        }

        private static bool WaitForResult(
            FrameOnnxRunner runner,
            out FrameOnnxInferenceResult result,
            int maxAttempts = 100)
        {
            for (int i = 0; i < maxAttempts; i++)
            {
                if (runner.TryGetResult(out result))
                    return true;

                System.Threading.Thread.Sleep(10);
            }

            result = null;
            return false;
        }

        private sealed class TestOnnxDetectorSession : IOnnxDetectorSession
        {
            private readonly Func<float[], int, int, float[]> run;

            public TestOnnxDetectorSession(Func<float[], int, int, float[]> run)
            {
                this.run = run ?? throw new ArgumentNullException(nameof(run));
            }

            public float[] Run(float[] nchwInput, int width, int height)
            {
                return run(nchwInput, width, height);
            }

            public void Dispose()
            {
            }
        }
    }
}
