using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace OnnxRuntimeInference
{
    public static class DetectorJsonConfigLoader
    {
        public static DetectorModelProfile LoadFirst(string json)
        {
            IReadOnlyList<DetectorModelProfile> profiles = LoadAll(json);
            if (profiles.Count == 0)
                throw new InvalidOperationException("Detector json contains no models.");

            return profiles[0];
        }

        public static DetectorModelProfile LoadFirstFromFile(string path)
        {
            return LoadFirst(ReadJsonFile(path));
        }

        public static IReadOnlyList<DetectorModelProfile> LoadAllFromFile(string path)
        {
            return LoadAll(ReadJsonFile(path));
        }

        public static IReadOnlyList<DetectorModelProfile> LoadAll(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
                throw new ArgumentException("Detector json cannot be empty.", nameof(json));

            DetectorConfigRoot root = JsonUtility.FromJson<DetectorConfigRoot>(json);
            if (root == null || root.models == null || root.models.Length == 0)
                return Array.Empty<DetectorModelProfile>();

            var profiles = new List<DetectorModelProfile>(root.models.Length);
            for (int i = 0; i < root.models.Length; i++)
            {
                DetectorConfigModel item = root.models[i];
                if (item == null || item.model == null)
                    continue;

                DetectorInputSpec inputSpec = new DetectorInputSpec(
                    item.model.inputWidth,
                    item.model.inputHeight,
                    ParseColorOrder(item.model.tensorColorOrder),
                    item.model.normalizeToUnitRange);

                profiles.Add(new DetectorModelProfile(
                    item.detectorId,
                    item.displayName,
                    item.model.onnxModelName,
                    inputSpec,
                    ParseClasses(item.classes)));
            }

            return profiles;
        }

        private static string ReadJsonFile(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("Detector json path cannot be empty.", nameof(path));

            return File.ReadAllText(path);
        }

        private static IReadOnlyList<DetectorClass> ParseClasses(DetectorConfigClass[] source)
        {
            if (source == null || source.Length == 0)
                return Array.Empty<DetectorClass>();

            var classes = new DetectorClass[source.Length];
            for (int i = 0; i < source.Length; i++)
            {
                DetectorConfigClass item = source[i];
                classes[i] = new DetectorClass(i, item?.label, item?.threshold ?? 0f);
            }

            return classes;
        }

        private static ColorOrder ParseColorOrder(string value)
        {
            if (string.Equals(value, "bgr", StringComparison.OrdinalIgnoreCase))
                return ColorOrder.Bgr;

            return ColorOrder.Rgb;
        }

#pragma warning disable 0649
        [Serializable]
        private sealed class DetectorConfigRoot
        {
            public DetectorConfigModel[] models;
        }

        [Serializable]
        private sealed class DetectorConfigModel
        {
            public string detectorId;
            public string displayName;
            public DetectorConfigModelFile model;
            public DetectorConfigClass[] classes;
        }

        [Serializable]
        private sealed class DetectorConfigModelFile
        {
            public string onnxModelName;
            public int inputWidth;
            public int inputHeight;
            public string tensorColorOrder;
            public bool normalizeToUnitRange;
        }

        [Serializable]
        private sealed class DetectorConfigClass
        {
            public string label;
            public float threshold;
        }
#pragma warning restore 0649
    }
}
