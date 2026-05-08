using System;

namespace OnnxRuntimeInference
{
    public sealed class OnnxInputFrame : IDisposable
    {
        private readonly Action<byte[]> releasePixels;
        private bool disposed;

        public OnnxInputFrame(
            byte[] pixels,
            int width,
            int height,
            OnnxFramePixelFormat format,
            bool rowsBottomUp,
            long frameId,
            DateTime timestampUtc,
            Action<byte[]> releasePixels = null)
        {
            if (width < 0)
                throw new ArgumentOutOfRangeException(nameof(width), "Width cannot be negative.");
            if (height < 0)
                throw new ArgumentOutOfRangeException(nameof(height), "Height cannot be negative.");

            Pixels = pixels ?? Array.Empty<byte>();
            Width = width;
            Height = height;
            Format = format;
            RowsBottomUp = rowsBottomUp;
            FrameId = frameId;
            TimestampUtc = timestampUtc.Kind == DateTimeKind.Utc
                ? timestampUtc
                : timestampUtc.ToUniversalTime();
            this.releasePixels = releasePixels;
        }

        public byte[] Pixels { get; }
        public int Width { get; }
        public int Height { get; }
        public OnnxFramePixelFormat Format { get; }
        public bool RowsBottomUp { get; }
        public long FrameId { get; }
        public DateTime TimestampUtc { get; }

        public void Dispose()
        {
            if (disposed)
                return;

            disposed = true;
            releasePixels?.Invoke(Pixels);
        }
    }
}
