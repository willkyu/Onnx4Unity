namespace OnnxRuntimeInference
{
    public enum OnnxResizeAlgorithm
    {
        Nearest = 0,
        Bilinear = 1
    }

    internal static class OnnxResizeAlgorithmUtility
    {
        public static OnnxResizeAlgorithm FromEnumName(System.Enum value)
        {
            if (value == null)
                throw new System.ArgumentNullException(nameof(value));

            string name = value.ToString();
            if (System.Enum.TryParse(name, ignoreCase: false, out OnnxResizeAlgorithm algorithm))
                return algorithm;

            throw new System.ArgumentOutOfRangeException(nameof(value), value, "Unsupported resize algorithm.");
        }
    }
}
