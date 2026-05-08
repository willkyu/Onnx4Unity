using System;

namespace OnnxRuntimeInference
{
    public sealed class PreparedFrameOnnxInputBuffer : IDisposable
    {
        private readonly object syncRoot = new object();
        private readonly DetectorInputSpec inputSpec;
        private readonly OnnxResizeAlgorithm resizeAlgorithm;
        private readonly Slot[] slots;
        private int latestSlotIndex = -1;
        private bool disposed;

        public PreparedFrameOnnxInputBuffer(
            DetectorInputSpec inputSpec,
            OnnxResizeAlgorithm resizeAlgorithm = OnnxResizeAlgorithm.Nearest)
        {
            this.inputSpec = inputSpec ?? throw new ArgumentNullException(nameof(inputSpec));
            this.resizeAlgorithm = resizeAlgorithm;

            int previewBytes = checked(inputSpec.Width * inputSpec.Height * 4);
            int tensorFloats = checked(inputSpec.Width * inputSpec.Height * 3);
            slots = new[]
            {
                new Slot(previewBytes, tensorFloats),
                new Slot(previewBytes, tensorFloats)
            };
        }

        public PreparedFrameOnnxInputBuffer(
            DetectorInputSpec inputSpec,
            Enum resizeAlgorithm)
            : this(inputSpec, OnnxResizeAlgorithmUtility.FromEnumName(resizeAlgorithm))
        {
        }

        public int Width => inputSpec.Width;
        public int Height => inputSpec.Height;
        public OnnxResizeAlgorithm ResizeAlgorithm => resizeAlgorithm;

        public bool TryAcquireWrite(out WriteLease lease)
        {
            ThrowIfDisposed();

            lock (syncRoot)
            {
                for (int i = 0; i < slots.Length; i++)
                {
                    if (slots[i].State != SlotState.Free)
                        continue;

                    slots[i].State = SlotState.Writing;
                    lease = new WriteLease(this, i);
                    return true;
                }
            }

            lease = null;
            return false;
        }

        public bool TryAcquireLatest(out ReadLease lease)
        {
            ThrowIfDisposed();

            lock (syncRoot)
            {
                if (latestSlotIndex < 0)
                {
                    lease = null;
                    return false;
                }

                Slot slot = slots[latestSlotIndex];
                if (slot.State != SlotState.Ready)
                {
                    lease = null;
                    return false;
                }

                slot.State = SlotState.Reading;
                lease = new ReadLease(this, latestSlotIndex);
                return true;
            }
        }

        public bool TryCopyLatestPreviewPixels(
            byte[] destination,
            out int width,
            out int height,
            out int originalWidth,
            out int originalHeight,
            out long frameId,
            out DateTime timestampUtc)
        {
            if (destination == null)
                throw new ArgumentNullException(nameof(destination));
            ThrowIfDisposed();

            lock (syncRoot)
            {
                if (latestSlotIndex < 0)
                    return EmptyCopyResult(out width, out height, out originalWidth, out originalHeight, out frameId, out timestampUtc);

                Slot slot = slots[latestSlotIndex];
                if (slot.State != SlotState.Ready && slot.State != SlotState.Reading)
                    return EmptyCopyResult(out width, out height, out originalWidth, out originalHeight, out frameId, out timestampUtc);

                int byteCount = checked(inputSpec.Width * inputSpec.Height * 4);
                if (destination.Length < byteCount)
                    throw new ArgumentException("Destination buffer is smaller than the prepared preview image.", nameof(destination));

                Buffer.BlockCopy(slot.PreviewPixels, 0, destination, 0, byteCount);
                width = inputSpec.Width;
                height = inputSpec.Height;
                originalWidth = slot.OriginalWidth;
                originalHeight = slot.OriginalHeight;
                frameId = slot.FrameId;
                timestampUtc = slot.TimestampUtc;
                return true;
            }
        }

        public void Clear()
        {
            lock (syncRoot)
            {
                latestSlotIndex = -1;
                for (int i = 0; i < slots.Length; i++)
                    slots[i].State = SlotState.Free;
            }
        }

        public void ClearReadyFrames()
        {
            ThrowIfDisposed();

            lock (syncRoot)
            {
                latestSlotIndex = -1;
                for (int i = 0; i < slots.Length; i++)
                {
                    if (slots[i].State == SlotState.Ready)
                        slots[i].State = SlotState.Free;
                }
            }
        }

        public void Dispose()
        {
            disposed = true;
            Clear();
        }

        private static bool EmptyCopyResult(
            out int width,
            out int height,
            out int originalWidth,
            out int originalHeight,
            out long frameId,
            out DateTime timestampUtc)
        {
            width = 0;
            height = 0;
            originalWidth = 0;
            originalHeight = 0;
            frameId = 0;
            timestampUtc = default;
            return false;
        }

        private bool TryPrepareSlot(int slotIndex, OnnxInputFrame sourceFrame)
        {
            if (sourceFrame == null)
                throw new ArgumentNullException(nameof(sourceFrame));

            return TryPrepareSlot(
                slotIndex,
                sourceFrame.Pixels,
                sourceFrame.Width,
                sourceFrame.Height,
                sourceFrame.Format,
                sourceFrame.RowsBottomUp,
                sourceFrame.FrameId,
                sourceFrame.TimestampUtc);
        }

        private bool TryPrepareSlot(
            int slotIndex,
            byte[] pixels,
            int width,
            int height,
            OnnxFramePixelFormat format,
            bool rowsBottomUp,
            long frameId,
            DateTime timestampUtc)
        {
            if (pixels == null)
                throw new ArgumentNullException(nameof(pixels));
            if (format != OnnxFramePixelFormat.Rgba32 || rowsBottomUp)
                return false;
            if (width <= 0 || height <= 0)
                return false;

            Slot slot = slots[slotIndex];
            if (resizeAlgorithm == OnnxResizeAlgorithm.Nearest)
                PrepareNearest(pixels, width, height, slot);
            else
                PrepareBilinear(pixels, width, height, slot);

            lock (syncRoot)
            {
                slot.OriginalWidth = width;
                slot.OriginalHeight = height;
                slot.FrameId = frameId;
                slot.TimestampUtc = timestampUtc.Kind == DateTimeKind.Utc
                    ? timestampUtc
                    : timestampUtc.ToUniversalTime();

                if (latestSlotIndex >= 0 && latestSlotIndex != slotIndex && slots[latestSlotIndex].State == SlotState.Ready)
                    slots[latestSlotIndex].State = SlotState.Free;

                latestSlotIndex = slotIndex;
                slot.State = SlotState.Ready;
            }

            return true;
        }

        private void ReleaseWrite(int slotIndex)
        {
            lock (syncRoot)
            {
                if (slots[slotIndex].State == SlotState.Writing)
                    slots[slotIndex].State = SlotState.Free;
            }
        }

        private void ReleaseRead(int slotIndex)
        {
            lock (syncRoot)
            {
                if (slots[slotIndex].State != SlotState.Reading)
                    return;

                slots[slotIndex].State = SlotState.Free;
                if (latestSlotIndex == slotIndex)
                    latestSlotIndex = -1;
            }
        }

        private void PrepareNearest(byte[] source, int srcWidth, int srcHeight, Slot slot)
        {
            byte[] preview = slot.PreviewPixels;
            float[] tensor = slot.Tensor;
            int dstWidth = inputSpec.Width;
            int dstHeight = inputSpec.Height;
            int pixelCount = checked(dstWidth * dstHeight);
            float scale = inputSpec.NormalizeToUnitRange ? 1f / 255f : 1f;

            for (int y = 0; y < dstHeight; y++)
            {
                int srcY = y * srcHeight / dstHeight;
                int srcRow = srcY * srcWidth * 4;
                int dstRow = y * dstWidth * 4;

                for (int x = 0; x < dstWidth; x++)
                {
                    int srcX = x * srcWidth / dstWidth;
                    int srcIndex = srcRow + (srcX * 4);
                    int dstPixel = (y * dstWidth) + x;
                    int dstIndex = dstRow + (x * 4);

                    byte r = source[srcIndex + 0];
                    byte g = source[srcIndex + 1];
                    byte b = source[srcIndex + 2];
                    byte a = source[srcIndex + 3];

                    preview[dstIndex + 0] = r;
                    preview[dstIndex + 1] = g;
                    preview[dstIndex + 2] = b;
                    preview[dstIndex + 3] = a;

                    if (inputSpec.TensorColorOrder == ColorOrder.Bgr)
                    {
                        tensor[dstPixel] = b * scale;
                        tensor[pixelCount + dstPixel] = g * scale;
                        tensor[(pixelCount * 2) + dstPixel] = r * scale;
                    }
                    else
                    {
                        tensor[dstPixel] = r * scale;
                        tensor[pixelCount + dstPixel] = g * scale;
                        tensor[(pixelCount * 2) + dstPixel] = b * scale;
                    }
                }
            }
        }

        private void PrepareBilinear(byte[] pixels, int width, int height, Slot slot)
        {
            OnnxRgba32Resizer.ResizeBilinear(
                pixels,
                width,
                height,
                slot.PreviewPixels,
                inputSpec.Width,
                inputSpec.Height);

            TensorPreprocessor.WriteNchw(
                slot.PreviewPixels,
                inputSpec.Width,
                inputSpec.Height,
                OnnxFramePixelFormat.Rgba32,
                rowsBottomUp: false,
                inputSpec,
                ColorOrder.Rgb,
                slot.Tensor);
        }

        private void ThrowIfDisposed()
        {
            if (disposed)
                throw new ObjectDisposedException(nameof(PreparedFrameOnnxInputBuffer));
        }

        private sealed class Slot
        {
            public Slot(int previewBytes, int tensorFloats)
            {
                PreviewPixels = new byte[previewBytes];
                Tensor = new float[tensorFloats];
            }

            public byte[] PreviewPixels { get; }
            public float[] Tensor { get; }
            public int OriginalWidth { get; set; }
            public int OriginalHeight { get; set; }
            public long FrameId { get; set; }
            public DateTime TimestampUtc { get; set; }
            public SlotState State { get; set; }
        }

        private enum SlotState
        {
            Free,
            Writing,
            Ready,
            Reading
        }

        public sealed class WriteLease : IDisposable
        {
            private PreparedFrameOnnxInputBuffer owner;
            private readonly int slotIndex;

            internal WriteLease(PreparedFrameOnnxInputBuffer owner, int slotIndex)
            {
                this.owner = owner;
                this.slotIndex = slotIndex;
            }

            public bool TryPrepare(OnnxInputFrame sourceFrame)
            {
                if (owner == null)
                    throw new ObjectDisposedException(nameof(WriteLease));

                bool prepared = owner.TryPrepareSlot(slotIndex, sourceFrame);
                MarkPublishedIfPrepared(prepared);

                return prepared;
            }

            internal bool TryPrepare(
                byte[] pixels,
                int width,
                int height,
                OnnxFramePixelFormat format,
                bool rowsBottomUp,
                long frameId,
                DateTime timestampUtc)
            {
                if (owner == null)
                    throw new ObjectDisposedException(nameof(WriteLease));

                bool prepared = owner.TryPrepareSlot(
                    slotIndex,
                    pixels,
                    width,
                    height,
                    format,
                    rowsBottomUp,
                    frameId,
                    timestampUtc);
                MarkPublishedIfPrepared(prepared);

                return prepared;
            }

            private void MarkPublishedIfPrepared(bool prepared)
            {
                if (prepared)
                    owner = null;
            }

            public void Dispose()
            {
                PreparedFrameOnnxInputBuffer buffer = owner;
                owner = null;
                buffer?.ReleaseWrite(slotIndex);
            }
        }

        public sealed class ReadLease : IDisposable
        {
            private PreparedFrameOnnxInputBuffer owner;
            private readonly int slotIndex;

            internal ReadLease(PreparedFrameOnnxInputBuffer owner, int slotIndex)
            {
                this.owner = owner;
                this.slotIndex = slotIndex;
            }

            public int Width => Owner.inputSpec.Width;
            public int Height => Owner.inputSpec.Height;
            public int OriginalWidth => Slot.OriginalWidth;
            public int OriginalHeight => Slot.OriginalHeight;
            public long FrameId => Slot.FrameId;
            public DateTime TimestampUtc => Slot.TimestampUtc;
            public byte[] PreviewPixels => Slot.PreviewPixels;
            public float[] Tensor => Slot.Tensor;

            public OnnxInputFrame CreatePreviewInputFrame()
            {
                return new OnnxInputFrame(
                    PreviewPixels,
                    Width,
                    Height,
                    OnnxFramePixelFormat.Rgba32,
                    rowsBottomUp: false,
                    FrameId,
                    TimestampUtc);
            }

            public void Dispose()
            {
                PreparedFrameOnnxInputBuffer buffer = owner;
                owner = null;
                buffer?.ReleaseRead(slotIndex);
            }

            private PreparedFrameOnnxInputBuffer Owner =>
                owner ?? throw new ObjectDisposedException(nameof(ReadLease));

            private Slot Slot => Owner.slots[slotIndex];
        }
    }
}
