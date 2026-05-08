using System;

namespace OnnxRuntimeInference
{
    public interface IOnnxDetectorSession : IDisposable
    {
        float[] Run(float[] nchwInput, int width, int height);
    }
}
