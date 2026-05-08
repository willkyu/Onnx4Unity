using System;

namespace OnnxRuntimeInference
{
    internal static class OnnxRgba32Resizer
    {
        public static void ResizeNearest(
            byte[] srcRgba,
            int srcWidth,
            int srcHeight,
            byte[] dstRgba,
            int dstWidth,
            int dstHeight)
        {
            ValidateResizeArgs(srcRgba, srcWidth, srcHeight, dstRgba, dstWidth, dstHeight);

            int dstRequired = checked(dstWidth * dstHeight * 4);
            if (srcWidth == dstWidth && srcHeight == dstHeight)
            {
                Buffer.BlockCopy(srcRgba, 0, dstRgba, 0, dstRequired);
                return;
            }

            for (int y = 0; y < dstHeight; y++)
            {
                int srcY = y * srcHeight / dstHeight;
                int srcRow = srcY * srcWidth * 4;
                int dstRow = y * dstWidth * 4;

                for (int x = 0; x < dstWidth; x++)
                {
                    int srcX = x * srcWidth / dstWidth;
                    int srcIndex = srcRow + (srcX * 4);
                    int dstIndex = dstRow + (x * 4);

                    dstRgba[dstIndex + 0] = srcRgba[srcIndex + 0];
                    dstRgba[dstIndex + 1] = srcRgba[srcIndex + 1];
                    dstRgba[dstIndex + 2] = srcRgba[srcIndex + 2];
                    dstRgba[dstIndex + 3] = srcRgba[srcIndex + 3];
                }
            }
        }

        public static void ResizeBilinear(
            byte[] srcRgba,
            int srcWidth,
            int srcHeight,
            byte[] dstRgba,
            int dstWidth,
            int dstHeight)
        {
            ValidateResizeArgs(srcRgba, srcWidth, srcHeight, dstRgba, dstWidth, dstHeight);

            int dstRequired = checked(dstWidth * dstHeight * 4);
            if (srcWidth == dstWidth && srcHeight == dstHeight)
            {
                Buffer.BlockCopy(srcRgba, 0, dstRgba, 0, dstRequired);
                return;
            }

            float scaleX = srcWidth / (float)dstWidth;
            float scaleY = srcHeight / (float)dstHeight;

            for (int y = 0; y < dstHeight; y++)
            {
                float sy = ((y + 0.5f) * scaleY) - 0.5f;
                int y0 = (int)Math.Floor(sy);
                int y1 = y0 + 1;
                float fy = sy - y0;

                if (y0 < 0)
                {
                    y0 = 0;
                    fy = 0f;
                }

                if (y1 >= srcHeight)
                    y1 = srcHeight - 1;

                for (int x = 0; x < dstWidth; x++)
                {
                    float sx = ((x + 0.5f) * scaleX) - 0.5f;
                    int x0 = (int)Math.Floor(sx);
                    int x1 = x0 + 1;
                    float fx = sx - x0;

                    if (x0 < 0)
                    {
                        x0 = 0;
                        fx = 0f;
                    }

                    if (x1 >= srcWidth)
                        x1 = srcWidth - 1;

                    int dstIndex = (y * dstWidth + x) * 4;
                    int i00 = (y0 * srcWidth + x0) * 4;
                    int i10 = (y0 * srcWidth + x1) * 4;
                    int i01 = (y1 * srcWidth + x0) * 4;
                    int i11 = (y1 * srcWidth + x1) * 4;

                    for (int c = 0; c < 4; c++)
                    {
                        float v00 = srcRgba[i00 + c];
                        float v10 = srcRgba[i10 + c];
                        float v01 = srcRgba[i01 + c];
                        float v11 = srcRgba[i11 + c];
                        float vx0 = v00 + ((v10 - v00) * fx);
                        float vx1 = v01 + ((v11 - v01) * fx);
                        int value = (int)Math.Round(vx0 + ((vx1 - vx0) * fy));
                        dstRgba[dstIndex + c] = ClampToByte(value);
                    }
                }
            }
        }

        private static void ValidateResizeArgs(
            byte[] srcRgba,
            int srcWidth,
            int srcHeight,
            byte[] dstRgba,
            int dstWidth,
            int dstHeight)
        {
            if (srcRgba == null)
                throw new ArgumentNullException(nameof(srcRgba));
            if (dstRgba == null)
                throw new ArgumentNullException(nameof(dstRgba));
            if (srcWidth <= 0 || srcHeight <= 0)
                throw new ArgumentOutOfRangeException(nameof(srcWidth), "Source size must be positive.");
            if (dstWidth <= 0 || dstHeight <= 0)
                throw new ArgumentOutOfRangeException(nameof(dstWidth), "Destination size must be positive.");

            int srcRequired = checked(srcWidth * srcHeight * 4);
            int dstRequired = checked(dstWidth * dstHeight * 4);
            if (srcRgba.Length < srcRequired)
                throw new ArgumentException("Source buffer is too small for the given RGBA32 size.", nameof(srcRgba));
            if (dstRgba.Length < dstRequired)
                throw new ArgumentException("Destination buffer is too small for the given RGBA32 size.", nameof(dstRgba));
        }

        private static byte ClampToByte(int value)
        {
            if (value < 0)
                return 0;
            if (value > 255)
                return 255;
            return (byte)value;
        }
    }
}
