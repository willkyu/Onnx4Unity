using System;

namespace OnnxRuntimeInference
{
    public enum OnnxFramePixelFormat
    {
        Rgba32 = 0,
        Bgra32 = 1,
        Rgb24 = 2,
        Bgr24 = 3
    }

    public static class OnnxFramePixelFormatUtility
    {
        public static int GetBytesPerPixel(OnnxFramePixelFormat format)
        {
            switch (format)
            {
                case OnnxFramePixelFormat.Rgba32:
                case OnnxFramePixelFormat.Bgra32:
                    return 4;
                case OnnxFramePixelFormat.Rgb24:
                case OnnxFramePixelFormat.Bgr24:
                    return 3;
                default:
                    throw new ArgumentOutOfRangeException(nameof(format), format, "Unsupported frame format.");
            }
        }

        public static int GetByteCount(int width, int height, OnnxFramePixelFormat format)
        {
            if (width < 0)
                throw new ArgumentOutOfRangeException(nameof(width), "Width cannot be negative.");
            if (height < 0)
                throw new ArgumentOutOfRangeException(nameof(height), "Height cannot be negative.");

            return checked(width * height * GetBytesPerPixel(format));
        }
    }
}
