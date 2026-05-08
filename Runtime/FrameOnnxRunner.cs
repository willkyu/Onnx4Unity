using System;
using System.Diagnostics;
using System.Threading.Tasks;
using WindowCapture;

namespace OnnxRuntimeInference
{
    public sealed class FrameOnnxRunner : IDisposable
    {
        private readonly object syncRoot = new object();
        private readonly IOnnxDetectorSession session;
        private readonly DetectorModelProfile profile;
        private readonly FrameOnnxRunnerOptions options;

        private byte[] cpuResizeBuffer;
        private float[] tensorBuffer;
        private FrameOnnxInferenceResult completedResult;
        private Task runningTask;
        private bool inFlight;
        private bool disposed;

        public FrameOnnxRunner(
            IOnnxDetectorSession session,
            DetectorModelProfile profile,
            FrameResizeAlgorithm resizeAlgorithm = FrameResizeAlgorithm.Bilinear,
            bool applyClassNms = false,
            float nmsIouThreshold = 0.5f,
            bool disposeSession = false)
            : this(
                session,
                profile,
                new FrameOnnxRunnerOptions
                {
                    ResizeAlgorithm = resizeAlgorithm,
                    ApplyClassNms = applyClassNms,
                    NmsIouThreshold = nmsIouThreshold,
                    DisposeSession = disposeSession
                })
        {
        }

        public FrameOnnxRunner(
            IOnnxDetectorSession session,
            DetectorModelProfile profile,
            FrameOnnxRunnerOptions options)
        {
            this.session = session ?? throw new ArgumentNullException(nameof(session));
            this.profile = profile ?? throw new ArgumentNullException(nameof(profile));
            this.options = (options ?? new FrameOnnxRunnerOptions()).Clone();
        }

        public bool IsRunning
        {
            get
            {
                lock (syncRoot)
                    return inFlight;
            }
        }

        public DetectorModelProfile Profile => profile;

        public bool TryBeginRun(CapturedFrame sourceFrame)
        {
            if (sourceFrame == null)
                throw new ArgumentNullException(nameof(sourceFrame));
            if (sourceFrame.Width <= 0 || sourceFrame.Height <= 0)
                throw new ArgumentException("Captured frame size must be positive.", nameof(sourceFrame));
            ThrowIfDisposed();

            return TryBeginCpuRun(sourceFrame);
        }

        public bool TryBeginRun(PreparedFrameOnnxInputBuffer.ReadLease preparedInput)
        {
            if (preparedInput == null)
                throw new ArgumentNullException(nameof(preparedInput));
            if (preparedInput.Width <= 0 || preparedInput.Height <= 0)
                throw new ArgumentException("Prepared input size must be positive.", nameof(preparedInput));
            ThrowIfDisposed();

            return TryBeginPreparedRun(preparedInput);
        }

        public bool TryGetResult(out FrameOnnxInferenceResult result)
        {
            lock (syncRoot)
            {
                result = completedResult;
                if (completedResult == null)
                    return false;

                completedResult = null;
                return true;
            }
        }

        public void Dispose()
        {
            disposed = true;
            Task taskToWait;
            lock (syncRoot)
                taskToWait = runningTask;

            taskToWait?.Wait();

            if (options.DisposeSession)
                session.Dispose();
        }

        private bool TryBeginPreparedRun(PreparedFrameOnnxInputBuffer.ReadLease preparedInput)
        {
            if (!TryEnterRun())
                return false;

            try
            {
                BeginPreparedInference(preparedInput);
                return true;
            }
            catch
            {
                ExitRun();
                preparedInput.Dispose();
                throw;
            }
        }

        private bool TryBeginCpuRun(CapturedFrame sourceFrame)
        {
            if (sourceFrame.Format != FramePixelFormat.Rgba32)
                throw new InvalidOperationException("CPU resize currently requires RGBA32 captured frames.");

            if (!TryEnterRun())
                return false;

            try
            {
                int sourceByteCount = FramePixelFormatUtility.GetByteCount(sourceFrame.Width, sourceFrame.Height, sourceFrame.Format);
                var sourcePixels = new byte[sourceByteCount];
                Buffer.BlockCopy(sourceFrame.Pixels, 0, sourcePixels, 0, sourceByteCount);

                if (sourceFrame.RowsBottomUp)
                    FlipRgbaVerticalInPlace(sourcePixels, sourceFrame.Width, sourceFrame.Height);

                BeginCpuResizeInference(sourcePixels, sourceFrame.Width, sourceFrame.Height);
                return true;
            }
            catch
            {
                ExitRun();
                throw;
            }
        }

        private void BeginCpuResizeInference(byte[] sourcePixels, int sourceWidth, int sourceHeight)
        {
            Task task = Task.Run(() =>
            {
                try
                {
                    int inputWidth = profile.InputSpec.Width;
                    int inputHeight = profile.InputSpec.Height;
                    EnsureCpuResizeBuffer();
                    EnsureTensorBuffer();

                    var resizeWatch = Stopwatch.StartNew();
                    if (options.ResizeAlgorithm == FrameResizeAlgorithm.Nearest)
                    {
                        Rgba32Resizer.ResizeNearest(
                            sourcePixels,
                            sourceWidth,
                            sourceHeight,
                            cpuResizeBuffer,
                            inputWidth,
                            inputHeight);
                    }
                    else
                    {
                        Rgba32Resizer.ResizeBilinear(
                            sourcePixels,
                            sourceWidth,
                            sourceHeight,
                            cpuResizeBuffer,
                            inputWidth,
                            inputHeight);
                    }

                    resizeWatch.Stop();

                    var tensorWatch = Stopwatch.StartNew();
                    TensorPreprocessor.WriteNchw(
                        cpuResizeBuffer,
                        inputWidth,
                        inputHeight,
                        FramePixelFormat.Rgba32,
                        rowsBottomUp: false,
                        profile.InputSpec,
                        ColorOrder.Rgb,
                        tensorBuffer);
                    tensorWatch.Stop();

                    RunInferenceAndDecode(
                        tensorBuffer,
                        sourceWidth,
                        sourceHeight,
                        resizeWatch.Elapsed,
                        tensorWatch.Elapsed);
                }
                catch (Exception ex)
                {
                    CompleteResult(FrameOnnxInferenceResult.FromError(
                        ex.GetType().Name + ": " + ex.Message,
                        sourceWidth,
                        sourceHeight,
                        profile.InputSpec.Width,
                        profile.InputSpec.Height,
                        TimeSpan.Zero));
                }
            });

            SetRunningTask(task);
        }

        private void BeginPreparedInference(PreparedFrameOnnxInputBuffer.ReadLease preparedInput)
        {
            int originalWidth = preparedInput.OriginalWidth;
            int originalHeight = preparedInput.OriginalHeight;
            float[] tensor = preparedInput.Tensor;

            Task task = Task.Run(() =>
            {
                try
                {
                    RunInferenceAndDecode(
                        tensor,
                        originalWidth,
                        originalHeight,
                        TimeSpan.Zero,
                        TimeSpan.Zero,
                        InferencePreprocessBackend.PreparedCpuTensor);
                }
                catch (Exception ex)
                {
                    CompleteResult(FrameOnnxInferenceResult.FromError(
                        ex.GetType().Name + ": " + ex.Message,
                        originalWidth,
                        originalHeight,
                        profile.InputSpec.Width,
                        profile.InputSpec.Height,
                        TimeSpan.Zero,
                        InferencePreprocessBackend.PreparedCpuTensor));
                }
                finally
                {
                    preparedInput.Dispose();
                }
            });

            SetRunningTask(task);
        }

        private void RunInferenceAndDecode(
            float[] tensor,
            int originalWidth,
            int originalHeight,
            TimeSpan resizeDuration,
            TimeSpan tensorDuration,
            InferencePreprocessBackend preprocessBackend = InferencePreprocessBackend.CpuResizeCpuTensor)
        {
            int inputWidth = profile.InputSpec.Width;
            int inputHeight = profile.InputSpec.Height;

            var inferenceWatch = Stopwatch.StartNew();
            float[] output = session.Run(tensor, inputWidth, inputHeight);
            inferenceWatch.Stop();

            var decodeWatch = Stopwatch.StartNew();
            DetectionBatch batch = YoloEnd2EndDecoder.Decode(
                output,
                profile,
                originalWidth,
                originalHeight,
                options.ApplyClassNms,
                options.NmsIouThreshold);
            decodeWatch.Stop();

            CompleteResult(new FrameOnnxInferenceResult(
                batch,
                output,
                originalWidth,
                originalHeight,
                inputWidth,
                inputHeight,
                resizeDuration,
                tensorDuration,
                inferenceWatch.Elapsed,
                decodeWatch.Elapsed,
                string.Empty,
                preprocessBackend));
        }

        private void CompleteResult(FrameOnnxInferenceResult result)
        {
            lock (syncRoot)
            {
                completedResult = result;
                inFlight = false;
            }
        }

        private bool TryEnterRun()
        {
            lock (syncRoot)
            {
                if (inFlight)
                    return false;

                inFlight = true;
                return true;
            }
        }

        private void ExitRun()
        {
            lock (syncRoot)
                inFlight = false;
        }

        private void SetRunningTask(Task task)
        {
            lock (syncRoot)
                runningTask = task;
        }

        private void EnsureCpuResizeBuffer()
        {
            int required = checked(profile.InputSpec.Width * profile.InputSpec.Height * 4);
            if (cpuResizeBuffer == null || cpuResizeBuffer.Length != required)
                cpuResizeBuffer = new byte[required];
        }

        private void EnsureTensorBuffer()
        {
            int required = checked(profile.InputSpec.Width * profile.InputSpec.Height * 3);
            if (tensorBuffer == null || tensorBuffer.Length != required)
                tensorBuffer = new float[required];
        }

        private static void FlipRgbaVerticalInPlace(byte[] rgba, int width, int height)
        {
            int rowBytes = checked(width * 4);
            var temp = new byte[rowBytes];
            int half = height / 2;
            for (int y = 0; y < half; y++)
            {
                int top = y * rowBytes;
                int bottom = (height - 1 - y) * rowBytes;
                Buffer.BlockCopy(rgba, top, temp, 0, rowBytes);
                Buffer.BlockCopy(rgba, bottom, rgba, top, rowBytes);
                Buffer.BlockCopy(temp, 0, rgba, bottom, rowBytes);
            }
        }

        private void ThrowIfDisposed()
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(FrameOnnxRunner));
        }
    }
}
