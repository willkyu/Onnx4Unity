using System;

namespace OnnxRuntimeInference
{
    public static class TensorPreprocessor
    {
        public static float[] ToNchw(
            OnnxInputFrame frame,
            DetectorInputSpec inputSpec,
            ColorOrder sourceColorOrderOverride = ColorOrder.Rgb)
        {
            if (frame == null)
                throw new ArgumentNullException(nameof(frame));

            return ToNchw(
                frame.Pixels,
                frame.Width,
                frame.Height,
                frame.Format,
                frame.RowsBottomUp,
                inputSpec,
                sourceColorOrderOverride);
        }

        public static float[] ToNchw(
            byte[] pixels,
            int width,
            int height,
            OnnxFramePixelFormat format,
            bool rowsBottomUp,
            DetectorInputSpec inputSpec,
            ColorOrder sourceColorOrderOverride = ColorOrder.Rgb)
        {
            if (inputSpec == null)
                throw new ArgumentNullException(nameof(inputSpec));

            int pixelCount = checked(width * height);
            var tensor = new float[pixelCount * 3];
            WriteNchw(
                pixels,
                width,
                height,
                format,
                rowsBottomUp,
                inputSpec,
                sourceColorOrderOverride,
                tensor);
            return tensor;
        }

        public static void WriteNchw(
            byte[] pixels,
            int width,
            int height,
            OnnxFramePixelFormat format,
            bool rowsBottomUp,
            DetectorInputSpec inputSpec,
            ColorOrder sourceColorOrderOverride,
            float[] tensor)
        {
            if (pixels == null)
                throw new ArgumentNullException(nameof(pixels));
            if (inputSpec == null)
                throw new ArgumentNullException(nameof(inputSpec));
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (width != inputSpec.Width || height != inputSpec.Height)
                throw new InvalidOperationException("Frame size does not match detector input size.");

            int pixelCount = checked(width * height);
            if (tensor.Length < pixelCount * 3)
                throw new ArgumentException("Tensor buffer is smaller than expected NCHW float count.", nameof(tensor));

            int channelsPerPixel = GetChannelCount(format);
            int expectedBytes = checked(width * height * channelsPerPixel);
            if (pixels.Length < expectedBytes)
                throw new ArgumentException("Pixel buffer is smaller than expected frame byte count.", nameof(pixels));

            for (int y = 0; y < height; y++)
            {
                int sourceY = rowsBottomUp ? (height - 1 - y) : y;
                int rowOffset = sourceY * width * channelsPerPixel;
                for (int x = 0; x < width; x++)
                {
                    int sourceIndex = rowOffset + (x * channelsPerPixel);
                    ReadRgb(pixels, sourceIndex, format, sourceColorOrderOverride, out float r, out float g, out float b);

                    if (inputSpec.TensorColorOrder == ColorOrder.Bgr)
                    {
                        float temp = r;
                        r = b;
                        b = temp;
                    }

                    if (inputSpec.NormalizeToUnitRange)
                    {
                        r /= 255f;
                        g /= 255f;
                        b /= 255f;
                    }

                    int flatIndex = y * width + x;
                    tensor[flatIndex] = r;
                    tensor[pixelCount + flatIndex] = g;
                    tensor[(pixelCount * 2) + flatIndex] = b;
                }
            }
        }

        private static int GetChannelCount(OnnxFramePixelFormat format)
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

        private static void ReadRgb(
            byte[] pixels,
            int sourceIndex,
            OnnxFramePixelFormat format,
            ColorOrder sourceColorOrderOverride,
            out float r,
            out float g,
            out float b)
        {
            byte c0 = pixels[sourceIndex];
            byte c1 = pixels[sourceIndex + 1];
            byte c2 = pixels[sourceIndex + 2];

            bool sourceIsBgr = format == OnnxFramePixelFormat.Bgr24 ||
                format == OnnxFramePixelFormat.Bgra32 ||
                sourceColorOrderOverride == ColorOrder.Bgr;

            if (sourceIsBgr)
            {
                b = c0;
                g = c1;
                r = c2;
            }
            else
            {
                r = c0;
                g = c1;
                b = c2;
            }
        }
    }
}
