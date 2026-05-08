using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxRuntimeInference
{
    public sealed class OnnxRuntimeDetectorSession : IOnnxDetectorSession
    {
        private readonly InferenceSession session;
        private readonly string inputName;
        private readonly string outputName;

        public OnnxRuntimeDetectorSession(
            string modelPath,
            DetectorRuntimeKind runtimeKind,
            string preferredInputName = "images",
            string preferredOutputName = "output0")
        {
            if (string.IsNullOrWhiteSpace(modelPath))
                throw new ArgumentException("Model path cannot be empty.", nameof(modelPath));

            OrtNativeLibraryPreloader.EnsureLoaded();
            RequestedRuntimeKind = runtimeKind;

            DetectorRuntimeKind actualKind = runtimeKind;
            string warning = string.Empty;

            if (runtimeKind == DetectorRuntimeKind.OnnxRuntimeDirectML)
            {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
                if (!TryCreateDirectMlSession(modelPath, out session, out string dmlWarning))
                {
                    using var cpuOptions = new SessionOptions();
                    session = new InferenceSession(modelPath, cpuOptions);
                    actualKind = DetectorRuntimeKind.OnnxRuntimeCpu;
                    warning = dmlWarning;
                }
                else
                {
                    warning = OrtNativeLibraryPreloader.LoadWarning;
                }
#else
                throw new PlatformNotSupportedException("ONNX Runtime DirectML is only supported on Windows.");
#endif
            }
            else
            {
                using var cpuOptions = new SessionOptions();
                session = new InferenceSession(modelPath, cpuOptions);
                warning = OrtNativeLibraryPreloader.LoadWarning;
            }

            ActualRuntimeKind = actualKind;
            InitializationWarning = warning ?? string.Empty;
            inputName = ResolveName(session.InputMetadata, preferredInputName);
            outputName = ResolveName(session.OutputMetadata, preferredOutputName);
        }

        public DetectorRuntimeKind RequestedRuntimeKind { get; }
        public DetectorRuntimeKind ActualRuntimeKind { get; }
        public string InitializationWarning { get; }
        public string InputName => inputName;
        public string OutputName => outputName;

        public float[] Run(float[] nchwInput, int width, int height)
        {
            if (nchwInput == null)
                throw new ArgumentNullException(nameof(nchwInput));
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");
            if (height <= 0)
                throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");

            int requiredLength = checked(width * height * 3);
            if (nchwInput.Length < requiredLength)
                throw new ArgumentException("Input tensor is smaller than 1x3xHxW.", nameof(nchwInput));

            var shape = new[] { 1, 3, height, width };
            var tensor = new DenseTensor<float>(nchwInput, shape);
            var inputs = new List<NamedOnnxValue>(1)
            {
                NamedOnnxValue.CreateFromTensor(inputName, tensor)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);
            DisposableNamedOnnxValue output = null;
            foreach (DisposableNamedOnnxValue value in results)
            {
                if (string.Equals(value.Name, outputName, StringComparison.Ordinal))
                {
                    output = value;
                    break;
                }
            }

            output ??= results.FirstOrDefault();
            if (output == null)
                throw new InvalidOperationException("ONNX Runtime returned no outputs.");

            Tensor<float> outTensor = output.AsTensor<float>();
            return outTensor.ToArray();
        }

        public void Dispose()
        {
            session?.Dispose();
        }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        private static bool TryCreateDirectMlSession(string modelPath, out InferenceSession dmlSession, out string warning)
        {
            dmlSession = null;
            warning = string.Empty;

            var errors = new List<string>(4);
            int[] deviceCandidates = { 0, 1, 2, 3 };

            for (int i = 0; i < deviceCandidates.Length; i++)
            {
                int deviceId = deviceCandidates[i];
                try
                {
                    dmlSession = CreateDirectMlSessionOnMtaThread(modelPath, deviceId);
                    return true;
                }
                catch (Exception failure)
                {
                    errors.Add("deviceId=" + deviceId + " -> " + failure.Message);
                }
            }

            var sb = new StringBuilder();
            sb.Append("DirectML init failed; fell back to CPU.");
            if (!string.IsNullOrWhiteSpace(OrtNativeLibraryPreloader.LoadedPath))
                sb.Append(" Loaded ORT native path: ").Append(OrtNativeLibraryPreloader.LoadedPath).Append('.');
            if (!string.IsNullOrWhiteSpace(OrtNativeLibraryPreloader.DirectMlLoadedPath))
                sb.Append(" Loaded DirectML path: ").Append(OrtNativeLibraryPreloader.DirectMlLoadedPath).Append('.');
            if (!string.IsNullOrWhiteSpace(OrtNativeLibraryPreloader.LoadWarning))
                sb.Append(' ').Append(OrtNativeLibraryPreloader.LoadWarning);
            if (errors.Count > 0)
                sb.Append(" DML errors: ").Append(string.Join(" | ", errors));

            warning = sb.ToString();
            return false;
        }

        private static InferenceSession CreateDirectMlSessionOnMtaThread(string modelPath, int deviceId)
        {
            Exception error = null;
            InferenceSession created = null;

            var thread = new Thread(() =>
            {
                try
                {
                    using var dmlOptions = new SessionOptions();
                    ConfigureDirectMlSessionOptions(dmlOptions);
                    dmlOptions.AppendExecutionProvider_DML(deviceId);
                    created = new InferenceSession(modelPath, dmlOptions);
                }
                catch (Exception ex)
                {
                    error = ex;
                }
            });

            thread.IsBackground = true;
            thread.SetApartmentState(ApartmentState.MTA);
            thread.Start();
            thread.Join();

            if (error != null)
                throw error;
            if (created == null)
                throw new InvalidOperationException("DirectML session creation returned null without exception.");

            return created;
        }

        private static void ConfigureDirectMlSessionOptions(SessionOptions options)
        {
            if (options == null)
                throw new ArgumentNullException(nameof(options));

            options.EnableMemoryPattern = false;
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
        }

#endif

        private static string ResolveName(IReadOnlyDictionary<string, NodeMetadata> metadata, string preferred)
        {
            if (metadata == null || metadata.Count == 0)
                return preferred ?? string.Empty;

            if (!string.IsNullOrWhiteSpace(preferred) && metadata.ContainsKey(preferred))
                return preferred;

            foreach (string key in metadata.Keys)
                return key;

            return preferred ?? string.Empty;
        }
    }
}
