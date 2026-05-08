using System;
using System.Collections.Generic;

namespace OnnxRuntimeInference
{
    public enum ColorOrder
    {
        Rgb = 0,
        Bgr = 1
    }

    public enum DetectorRuntimeKind
    {
        OnnxRuntimeCpu = 0,
        OnnxRuntimeDirectML = 1
    }

    public sealed class DetectorInputSpec
    {
        public DetectorInputSpec(int width, int height, ColorOrder tensorColorOrder, bool normalizeToUnitRange)
        {
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), "Width must be positive.");
            if (height <= 0)
                throw new ArgumentOutOfRangeException(nameof(height), "Height must be positive.");

            Width = width;
            Height = height;
            TensorColorOrder = tensorColorOrder;
            NormalizeToUnitRange = normalizeToUnitRange;
        }

        public int Width { get; }
        public int Height { get; }
        public ColorOrder TensorColorOrder { get; }
        public bool NormalizeToUnitRange { get; }
    }

    public sealed class DetectorClass
    {
        public DetectorClass(int id, string label, float threshold)
        {
            Id = id;
            Label = label ?? string.Empty;
            Threshold = threshold;
        }

        public int Id { get; }
        public string Label { get; }
        public float Threshold { get; }
    }

    public sealed class DetectorModelProfile
    {
        private readonly DetectorClass[] classes;

        public DetectorModelProfile(
            string detectorId,
            string displayName,
            string onnxModelName,
            DetectorInputSpec inputSpec,
            IReadOnlyList<DetectorClass> classes)
        {
            DetectorId = detectorId ?? string.Empty;
            DisplayName = displayName ?? string.Empty;
            OnnxModelName = onnxModelName ?? string.Empty;
            InputSpec = inputSpec ?? throw new ArgumentNullException(nameof(inputSpec));

            if (classes == null || classes.Count == 0)
            {
                this.classes = Array.Empty<DetectorClass>();
            }
            else
            {
                this.classes = new DetectorClass[classes.Count];
                for (int i = 0; i < classes.Count; i++)
                    this.classes[i] = classes[i] ?? new DetectorClass(i, string.Empty, 0f);
            }
        }

        public string DetectorId { get; }
        public string DisplayName { get; }
        public string OnnxModelName { get; }
        public DetectorInputSpec InputSpec { get; }
        public IReadOnlyList<DetectorClass> Classes => classes;
    }

    public sealed class DetectionResult
    {
        public DetectionResult(int classId, string label, float confidence, float x1, float y1, float x2, float y2)
        {
            ClassId = classId;
            Label = label ?? string.Empty;
            Confidence = confidence;
            X1 = x1;
            Y1 = y1;
            X2 = x2;
            Y2 = y2;
        }

        public int ClassId { get; }
        public string Label { get; }
        public float Confidence { get; }
        public float X1 { get; }
        public float Y1 { get; }
        public float X2 { get; }
        public float Y2 { get; }
    }

    public sealed class DetectionBatch
    {
        public DetectionBatch(IReadOnlyList<DetectionResult> detections, IReadOnlyDictionary<string, float> classScores)
        {
            Detections = detections ?? Array.Empty<DetectionResult>();
            ClassScores = classScores ?? new Dictionary<string, float>();
        }

        public IReadOnlyList<DetectionResult> Detections { get; }
        public IReadOnlyDictionary<string, float> ClassScores { get; }
    }
}
