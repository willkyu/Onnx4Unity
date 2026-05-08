using System;
using System.Collections.Generic;

namespace OnnxRuntimeInference
{
    public static class YoloEnd2EndDecoder
    {
        public const int ExpectedRowCount = 300;
        public const int ValuesPerRow = 6;

        public static DetectionBatch Decode(
            float[] output,
            DetectorModelProfile profile,
            int originalWidth,
            int originalHeight,
            bool applyClassNms = false,
            float nmsIouThreshold = 0.5f)
        {
            if (profile == null)
                throw new ArgumentNullException(nameof(profile));

            return Decode(
                output,
                profile.InputSpec,
                profile.Classes,
                originalWidth,
                originalHeight,
                applyClassNms,
                nmsIouThreshold);
        }

        public static DetectionBatch Decode(
            float[] output,
            DetectorInputSpec inputSpec,
            IReadOnlyList<DetectorClass> classes,
            int originalWidth,
            int originalHeight,
            bool applyClassNms = false,
            float nmsIouThreshold = 0.5f)
        {
            if (output == null)
                throw new ArgumentNullException(nameof(output));
            if (inputSpec == null)
                throw new ArgumentNullException(nameof(inputSpec));

            int expectedLength = ExpectedRowCount * ValuesPerRow;
            if (output.Length < expectedLength)
                throw new InvalidOperationException("Detector output length is smaller than expected [1,300,6].");

            var detections = new List<DetectionResult>();
            var classScores = CreateClassScoreMap(classes);
            if (classes == null || classes.Count == 0)
                return new DetectionBatch(detections, classScores);

            int classCount = classes.Count;
            float clipMaxX = inputSpec.Width - 1f;
            float clipMaxY = inputSpec.Height - 1f;
            float sx = inputSpec.Width > 0 ? originalWidth / (float)inputSpec.Width : 1f;
            float sy = inputSpec.Height > 0 ? originalHeight / (float)inputSpec.Height : 1f;

            for (int i = 0; i < ExpectedRowCount; i++)
            {
                int baseIndex = i * ValuesPerRow;
                float x1 = output[baseIndex + 0];
                float y1 = output[baseIndex + 1];
                float x2 = output[baseIndex + 2];
                float y2 = output[baseIndex + 3];
                float confidence = output[baseIndex + 4];

                int classId = Clamp((int)Math.Round(output[baseIndex + 5]), 0, classCount - 1);
                DetectorClass targetClass = classes[classId];
                if (targetClass == null)
                    continue;

                if (classScores.TryGetValue(targetClass.Label, out float currentScore) && confidence > currentScore)
                    classScores[targetClass.Label] = confidence;

                if (confidence < targetClass.Threshold)
                    continue;

                x1 = Clamp(x1, 0f, clipMaxX) * sx;
                x2 = Clamp(x2, 0f, clipMaxX) * sx;
                y1 = Clamp(y1, 0f, clipMaxY) * sy;
                y2 = Clamp(y2, 0f, clipMaxY) * sy;

                if (x2 <= x1 || y2 <= y1)
                    continue;

                detections.Add(new DetectionResult(classId, targetClass.Label, confidence, x1, y1, x2, y2));
            }

            if (applyClassNms)
                detections = ApplyPerClassNms(detections, nmsIouThreshold);

            return new DetectionBatch(detections, classScores);
        }

        private static Dictionary<string, float> CreateClassScoreMap(IReadOnlyList<DetectorClass> classes)
        {
            var map = new Dictionary<string, float>(StringComparer.OrdinalIgnoreCase);
            if (classes == null)
                return map;

            for (int i = 0; i < classes.Count; i++)
            {
                string label = classes[i]?.Label ?? string.Empty;
                if (!map.ContainsKey(label))
                    map.Add(label, 0f);
            }

            return map;
        }

        private static List<DetectionResult> ApplyPerClassNms(IReadOnlyList<DetectionResult> detections, float iouThreshold)
        {
            var kept = new List<DetectionResult>();
            var grouped = new Dictionary<int, List<DetectionResult>>();

            for (int i = 0; i < detections.Count; i++)
            {
                DetectionResult detection = detections[i];
                if (!grouped.TryGetValue(detection.ClassId, out List<DetectionResult> classList))
                {
                    classList = new List<DetectionResult>();
                    grouped.Add(detection.ClassId, classList);
                }

                classList.Add(detection);
            }

            foreach (KeyValuePair<int, List<DetectionResult>> pair in grouped)
            {
                pair.Value.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
                var suppressed = new bool[pair.Value.Count];

                for (int i = 0; i < pair.Value.Count; i++)
                {
                    if (suppressed[i])
                        continue;

                    DetectionResult picked = pair.Value[i];
                    kept.Add(picked);

                    for (int j = i + 1; j < pair.Value.Count; j++)
                    {
                        if (suppressed[j])
                            continue;

                        if (ComputeIoU(picked, pair.Value[j]) >= iouThreshold)
                            suppressed[j] = true;
                    }
                }
            }

            return kept;
        }

        private static float ComputeIoU(DetectionResult a, DetectionResult b)
        {
            float ix1 = Math.Max(a.X1, b.X1);
            float iy1 = Math.Max(a.Y1, b.Y1);
            float ix2 = Math.Min(a.X2, b.X2);
            float iy2 = Math.Min(a.Y2, b.Y2);
            float iw = Math.Max(0f, ix2 - ix1);
            float ih = Math.Max(0f, iy2 - iy1);
            float intersection = iw * ih;
            if (intersection <= 0f)
                return 0f;

            float areaA = (a.X2 - a.X1) * (a.Y2 - a.Y1);
            float areaB = (b.X2 - b.X1) * (b.Y2 - b.Y1);
            float union = areaA + areaB - intersection;
            if (union <= 0f)
                return 0f;

            return intersection / union;
        }

        private static int Clamp(int value, int min, int max)
        {
            if (value < min)
                return min;
            if (value > max)
                return max;
            return value;
        }

        private static float Clamp(float value, float min, float max)
        {
            if (value < min)
                return min;
            if (value > max)
                return max;
            return value;
        }
    }
}
