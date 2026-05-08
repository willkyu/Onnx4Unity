using System;

namespace OnnxRuntimeInference
{
    public sealed class FrameOnnxInferenceResult
    {
        public FrameOnnxInferenceResult(
            DetectionBatch batch,
            float[] rawOutput,
            int originalWidth,
            int originalHeight,
            int inputWidth,
            int inputHeight,
            long sourceFrameId,
            DateTime sourceTimestampUtc,
            TimeSpan resizeDuration,
            TimeSpan tensorDuration,
            TimeSpan inferenceDuration,
            TimeSpan decodeDuration,
            string errorMessage = "",
            InferencePreprocessBackend preprocessBackend = InferencePreprocessBackend.CpuResizeCpuTensor)
        {
            Batch = batch ?? new DetectionBatch(null, null);
            RawOutput = rawOutput ?? Array.Empty<float>();
            OriginalWidth = originalWidth;
            OriginalHeight = originalHeight;
            InputWidth = inputWidth;
            InputHeight = inputHeight;
            SourceFrameId = sourceFrameId;
            SourceTimestampUtc = sourceTimestampUtc;
            ResizeDuration = resizeDuration;
            TensorDuration = tensorDuration;
            InferenceDuration = inferenceDuration;
            DecodeDuration = decodeDuration;
            ErrorMessage = errorMessage ?? string.Empty;
            PreprocessBackend = preprocessBackend;
        }

        public DetectionBatch Batch { get; }
        public float[] RawOutput { get; }
        public int OriginalWidth { get; }
        public int OriginalHeight { get; }
        public int InputWidth { get; }
        public int InputHeight { get; }
        public long SourceFrameId { get; }
        public DateTime SourceTimestampUtc { get; }
        public TimeSpan ResizeDuration { get; }
        public TimeSpan TensorDuration { get; }
        public TimeSpan InferenceDuration { get; }
        public TimeSpan DecodeDuration { get; }
        public string ErrorMessage { get; }
        public InferencePreprocessBackend PreprocessBackend { get; }
        public bool Succeeded => string.IsNullOrEmpty(ErrorMessage);

        public TimeSpan PreprocessDuration =>
            ResizeDuration + TensorDuration;

        public TimeSpan TotalActiveDuration =>
            ResizeDuration + TensorDuration + InferenceDuration + DecodeDuration;

        public double ActiveFps => TotalActiveDuration.TotalSeconds > 0d
            ? 1d / TotalActiveDuration.TotalSeconds
            : 0d;

        public static FrameOnnxInferenceResult FromError(
            string errorMessage,
            int originalWidth,
            int originalHeight,
            int inputWidth,
            int inputHeight,
            long sourceFrameId,
            DateTime sourceTimestampUtc,
            TimeSpan resizeDuration,
            InferencePreprocessBackend preprocessBackend = InferencePreprocessBackend.CpuResizeCpuTensor)
        {
            return new FrameOnnxInferenceResult(
                new DetectionBatch(null, null),
                null,
                originalWidth,
                originalHeight,
                inputWidth,
                inputHeight,
                sourceFrameId,
                sourceTimestampUtc,
                resizeDuration,
                TimeSpan.Zero,
                TimeSpan.Zero,
                TimeSpan.Zero,
                errorMessage,
                preprocessBackend);
        }
    }
}
