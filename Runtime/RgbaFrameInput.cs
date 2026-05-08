using System;

namespace OnnxRuntimeInference
{
    public readonly struct RgbaFrameInput
    {
        public RgbaFrameInput(
            byte[] pixels,
            int width,
            int height,
            bool rowsBottomUp = false,
            long frameId = 0,
            DateTime timestampUtc = default)
        {
            if (pixels == null)
                throw new ArgumentNullException(nameof(pixels));
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");
            if (height <= 0)
                throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");

            int required = checked(width * height * 4);
            if (pixels.Length < required)
                throw new ArgumentException("Pixel buffer is smaller than width * height * 4.", nameof(pixels));

            Pixels = pixels;
            Width = width;
            Height = height;
            RowsBottomUp = rowsBottomUp;
            FrameId = frameId;
            TimestampUtc = timestampUtc == default
                ? default
                : timestampUtc.Kind == DateTimeKind.Utc
                    ? timestampUtc
                    : timestampUtc.ToUniversalTime();
        }

        public byte[] Pixels { get; }
        public int Width { get; }
        public int Height { get; }
        public bool RowsBottomUp { get; }
        public long FrameId { get; }
        public DateTime TimestampUtc { get; }
    }
}
