using System;

namespace OnnxRuntimeInference
{
    public static class Rgba32Resizer
    {
        public static void ResizeNearest(
            byte[] source,
            int sourceWidth,
            int sourceHeight,
            byte[] destination,
            int destinationWidth,
            int destinationHeight)
        {
            Validate(source, sourceWidth, sourceHeight, destination, destinationWidth, destinationHeight);

            for (int y = 0; y < destinationHeight; y++)
            {
                int sourceY = y * sourceHeight / destinationHeight;
                int sourceRow = sourceY * sourceWidth * 4;
                int destinationRow = y * destinationWidth * 4;

                for (int x = 0; x < destinationWidth; x++)
                {
                    int sourceX = x * sourceWidth / destinationWidth;
                    Buffer.BlockCopy(source, sourceRow + sourceX * 4, destination, destinationRow + x * 4, 4);
                }
            }
        }

        public static void ResizeBilinear(
            byte[] source,
            int sourceWidth,
            int sourceHeight,
            byte[] destination,
            int destinationWidth,
            int destinationHeight)
        {
            Validate(source, sourceWidth, sourceHeight, destination, destinationWidth, destinationHeight);

            if (sourceWidth == destinationWidth && sourceHeight == destinationHeight)
            {
                Buffer.BlockCopy(source, 0, destination, 0, checked(destinationWidth * destinationHeight * 4));
                return;
            }

            float xScale = sourceWidth / (float)destinationWidth;
            float yScale = sourceHeight / (float)destinationHeight;

            for (int y = 0; y < destinationHeight; y++)
            {
                float sourceY = (y + 0.5f) * yScale - 0.5f;
                if (sourceY < 0f)
                    sourceY = 0f;

                int y0 = (int)sourceY;
                int y1 = Math.Min(y0 + 1, sourceHeight - 1);
                float yWeight = sourceY - y0;

                for (int x = 0; x < destinationWidth; x++)
                {
                    float sourceX = (x + 0.5f) * xScale - 0.5f;
                    if (sourceX < 0f)
                        sourceX = 0f;

                    int x0 = (int)sourceX;
                    int x1 = Math.Min(x0 + 1, sourceWidth - 1);
                    float xWeight = sourceX - x0;

                    int topLeft = (y0 * sourceWidth + x0) * 4;
                    int topRight = (y0 * sourceWidth + x1) * 4;
                    int bottomLeft = (y1 * sourceWidth + x0) * 4;
                    int bottomRight = (y1 * sourceWidth + x1) * 4;
                    int destinationIndex = (y * destinationWidth + x) * 4;

                    for (int channel = 0; channel < 4; channel++)
                    {
                        float top = source[topLeft + channel] + (source[topRight + channel] - source[topLeft + channel]) * xWeight;
                        float bottom = source[bottomLeft + channel] + (source[bottomRight + channel] - source[bottomLeft + channel]) * xWeight;
                        destination[destinationIndex + channel] = (byte)Math.Round(top + (bottom - top) * yWeight);
                    }
                }
            }
        }

        private static void Validate(
            byte[] source,
            int sourceWidth,
            int sourceHeight,
            byte[] destination,
            int destinationWidth,
            int destinationHeight)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));
            if (sourceWidth <= 0 || sourceHeight <= 0)
                throw new ArgumentOutOfRangeException(nameof(sourceWidth), "Source size must be positive.");
            if (destinationWidth <= 0 || destinationHeight <= 0)
                throw new ArgumentOutOfRangeException(nameof(destinationWidth), "Destination size must be positive.");
            if (source.Length < checked(sourceWidth * sourceHeight * 4))
                throw new ArgumentException("Source buffer is smaller than sourceWidth * sourceHeight * 4.", nameof(source));
            if (destination.Length < checked(destinationWidth * destinationHeight * 4))
                throw new ArgumentException("Destination buffer is smaller than destinationWidth * destinationHeight * 4.", nameof(destination));
        }
    }
}
