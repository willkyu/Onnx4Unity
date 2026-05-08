using WindowCapture;

namespace OnnxRuntimeInference
{
    public enum InferencePreprocessBackend
    {
        CpuResizeCpuTensor = 0,
        PreparedCpuTensor = 1
    }

    public sealed class FrameOnnxRunnerOptions
    {
        public FrameResizeAlgorithm ResizeAlgorithm { get; set; } = FrameResizeAlgorithm.Bilinear;
        public bool ApplyClassNms { get; set; }
        public float NmsIouThreshold { get; set; } = 0.5f;
        public bool DisposeSession { get; set; }

        public FrameOnnxRunnerOptions Clone()
        {
            return new FrameOnnxRunnerOptions
            {
                ResizeAlgorithm = ResizeAlgorithm,
                ApplyClassNms = ApplyClassNms,
                NmsIouThreshold = NmsIouThreshold,
                DisposeSession = DisposeSession
            };
        }
    }
}
