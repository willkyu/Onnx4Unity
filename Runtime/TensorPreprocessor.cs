using System;

namespace OnnxRuntimeInference
{
    public static class TensorPreprocessor
    {
        public static float[] ToNchw(RgbaFrameInput frame, DetectorInputSpec inputSpec)
        {
            return ToNchw(
                frame.Pixels,
                frame.Width,
                frame.Height,
                frame.RowsBottomUp,
                inputSpec);
        }

        public static float[] ToNchw(
            byte[] rgba32,
            int width,
            int height,
            bool rowsBottomUp,
            DetectorInputSpec inputSpec)
        {
            if (inputSpec == null)
                throw new ArgumentNullException(nameof(inputSpec));

            int pixelCount = checked(width * height);
            var tensor = new float[pixelCount * 3];
            WriteNchw(rgba32, width, height, rowsBottomUp, inputSpec, tensor);
            return tensor;
        }

        public static void WriteNchw(
            byte[] rgba32,
            int width,
            int height,
            bool rowsBottomUp,
            DetectorInputSpec inputSpec,
            float[] tensor)
        {
            if (rgba32 == null)
                throw new ArgumentNullException(nameof(rgba32));
            if (inputSpec == null)
                throw new ArgumentNullException(nameof(inputSpec));
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (width != inputSpec.Width || height != inputSpec.Height)
                throw new InvalidOperationException("Frame size does not match detector input size.");

            int pixelCount = checked(width * height);
            if (rgba32.Length < pixelCount * 4)
                throw new ArgumentException("RGBA32 buffer is smaller than width * height * 4.", nameof(rgba32));
            if (tensor.Length < pixelCount * 3)
                throw new ArgumentException("Tensor buffer is smaller than expected NCHW float count.", nameof(tensor));

            float scale = inputSpec.NormalizeToUnitRange ? 1f / 255f : 1f;
            bool bgrTensor = inputSpec.TensorColorOrder == ColorOrder.Bgr;

            for (int y = 0; y < height; y++)
            {
                int sourceY = rowsBottomUp ? (height - 1 - y) : y;
                int rowOffset = sourceY * width * 4;
                for (int x = 0; x < width; x++)
                {
                    int sourceIndex = rowOffset + x * 4;
                    int flatIndex = y * width + x;

                    float r = rgba32[sourceIndex] * scale;
                    float g = rgba32[sourceIndex + 1] * scale;
                    float b = rgba32[sourceIndex + 2] * scale;

                    if (bgrTensor)
                    {
                        tensor[flatIndex] = b;
                        tensor[pixelCount + flatIndex] = g;
                        tensor[(pixelCount * 2) + flatIndex] = r;
                    }
                    else
                    {
                        tensor[flatIndex] = r;
                        tensor[pixelCount + flatIndex] = g;
                        tensor[(pixelCount * 2) + flatIndex] = b;
                    }
                }
            }
        }
    }
}
