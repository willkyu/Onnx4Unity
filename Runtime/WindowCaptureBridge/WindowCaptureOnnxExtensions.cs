using System;
using WindowCapture;

namespace OnnxRuntimeInference
{
    public static class WindowCaptureOnnxExtensions
    {
        public static OnnxInputFrame ToOnnxInputFrame(this CapturedFrame frame)
        {
            if (frame == null)
                throw new ArgumentNullException(nameof(frame));

            return new OnnxInputFrame(
                frame.Pixels,
                frame.Width,
                frame.Height,
                ToOnnxFramePixelFormat(frame.Format),
                frame.RowsBottomUp,
                frame.FrameId,
                frame.TimestampUtc);
        }

        public static bool TryBeginRun(this FrameOnnxRunner runner, CapturedFrame sourceFrame)
        {
            if (runner == null)
                throw new ArgumentNullException(nameof(runner));

            using OnnxInputFrame inputFrame = sourceFrame.ToOnnxInputFrame();
            return runner.TryBeginRun(inputFrame);
        }

        public static bool TryPrepare(this PreparedFrameOnnxInputBuffer.WriteLease lease, CapturedFrame sourceFrame)
        {
            if (lease == null)
                throw new ArgumentNullException(nameof(lease));

            using OnnxInputFrame inputFrame = sourceFrame.ToOnnxInputFrame();
            return lease.TryPrepare(inputFrame);
        }

        public static CapturedFrame CreatePreviewFrame(this PreparedFrameOnnxInputBuffer.ReadLease lease)
        {
            if (lease == null)
                throw new ArgumentNullException(nameof(lease));

            return new CapturedFrame(
                lease.PreviewPixels,
                lease.Width,
                lease.Height,
                FramePixelFormat.Rgba32,
                rowsBottomUp: false,
                lease.FrameId,
                lease.TimestampUtc);
        }

        private static OnnxFramePixelFormat ToOnnxFramePixelFormat(FramePixelFormat format)
        {
            switch (format)
            {
                case FramePixelFormat.Rgba32:
                    return OnnxFramePixelFormat.Rgba32;
                case FramePixelFormat.Bgra32:
                    return OnnxFramePixelFormat.Bgra32;
                case FramePixelFormat.Rgb24:
                    return OnnxFramePixelFormat.Rgb24;
                case FramePixelFormat.Bgr24:
                    return OnnxFramePixelFormat.Bgr24;
                default:
                    throw new ArgumentOutOfRangeException(nameof(format), format, "Unsupported frame format.");
            }
        }
    }
}
