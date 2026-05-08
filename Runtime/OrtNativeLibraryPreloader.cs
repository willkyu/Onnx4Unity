using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;

namespace OnnxRuntimeInference
{
    public static class OrtNativeLibraryPreloader
    {
        private const uint LoadWithAlteredSearchPath = 0x00000008;
        private const ushort ImageFileMachineAmd64 = 0x8664;
        private const ushort ImageFileMachineArm64 = 0xAA64;
        private const ushort ImageFileMachineI386 = 0x014C;

        private static readonly object LoadLock = new object();
        private static IntPtr loadedHandle;
        private static string loadedPath;
        private static string loadWarning;
        private static string directMlLoadedPath;
        private static IntPtr providersSharedHandle;
        private static IntPtr providersDmlHandle;
        private static IntPtr directMlHandle;

        public static string LoadedPath => loadedPath ?? string.Empty;
        public static string LoadWarning => loadWarning ?? string.Empty;
        internal static string DirectMlLoadedPath => directMlLoadedPath ?? string.Empty;

        public static void EnsureLoaded()
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            lock (LoadLock)
            {
                if (loadedHandle != IntPtr.Zero)
                    return;

                string dllPath = FindOnnxRuntimeDll();
                if (string.IsNullOrWhiteSpace(dllPath))
                {
                    throw new DllNotFoundException(
                        "Could not find onnxruntime.dll. Expected it in "
                        + "Packages/com.willkyu.onnxruntime-inference/Runtime/Plugins/Windows/x86_64.");
                }

                string expectedPath = NormalizePath(dllPath);
                string dllDir = Path.GetDirectoryName(expectedPath);

                IntPtr existing = GetModuleHandle("onnxruntime.dll");
                if (existing != IntPtr.Zero)
                {
                    string existingPath = NormalizePath(GetModuleFilePath(existing));
                    loadedHandle = existing;
                    loadedPath = existingPath;
                    EnsureDirectMlLoaded(dllDir);

                    if (!PathEquals(existingPath, expectedPath))
                    {
                        AppendWarning(
                            "onnxruntime.dll was already loaded from a different path before preloader. "
                            + "expected='" + expectedPath + "' actual='" + existingPath + "'.");
                    }

                    return;
                }

                EnsureDirectMlLoaded(dllDir);

                if (!string.IsNullOrWhiteSpace(dllDir))
                {
                    providersSharedHandle = TryLoadOptional(Path.Combine(dllDir, "onnxruntime_providers_shared.dll"));
                    providersDmlHandle = TryLoadOptional(Path.Combine(dllDir, "onnxruntime_providers_dml.dll"));
                }

                loadedHandle = LoadLibraryEx(expectedPath, IntPtr.Zero, LoadWithAlteredSearchPath);
                if (loadedHandle == IntPtr.Zero)
                    loadedHandle = LoadLibrary(expectedPath);

                if (loadedHandle == IntPtr.Zero)
                {
                    int error = Marshal.GetLastWin32Error();
                    throw new DllNotFoundException("Failed to load '" + expectedPath + "'. Win32 error=" + error + ".");
                }

                loadedPath = expectedPath;
                Debug.Log("ONNX Runtime native library preloaded: " + expectedPath);
            }
#endif
        }

        private static void EnsureDirectMlLoaded(string dllDir)
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            if (directMlHandle != IntPtr.Zero)
                return;

            string bundledDirectMl = FindCompatibleBundledDirectMl(dllDir);
            IntPtr existingDirectMl = GetModuleHandle("DirectML.dll");
            if (existingDirectMl != IntPtr.Zero)
            {
                directMlHandle = existingDirectMl;
                directMlLoadedPath = NormalizePath(GetModuleFilePath(existingDirectMl));
                if (!string.IsNullOrWhiteSpace(bundledDirectMl) && !PathEquals(directMlLoadedPath, bundledDirectMl))
                {
                    Debug.Log(
                        "DirectML.dll was already loaded before preloader; using existing module. "
                        + "bundled='" + bundledDirectMl + "' actual='" + directMlLoadedPath + "'.");
                }

                return;
            }

            if (!string.IsNullOrWhiteSpace(bundledDirectMl))
            {
                directMlHandle = TryLoadOptional(bundledDirectMl);
                if (directMlHandle != IntPtr.Zero)
                    directMlLoadedPath = NormalizePath(GetModuleFilePath(directMlHandle));
            }

            if (directMlHandle == IntPtr.Zero)
            {
                string systemDirectMl = NormalizePath(Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.Windows),
                    "System32",
                    "DirectML.dll"));
                directMlHandle = TryLoadOptional(systemDirectMl);
                if (directMlHandle != IntPtr.Zero)
                    directMlLoadedPath = NormalizePath(GetModuleFilePath(directMlHandle));
                if (directMlHandle == IntPtr.Zero)
                    AppendWarning("Failed to preload DirectML.dll from bundled copy or system path: " + systemDirectMl);
            }
#else
            _ = dllDir;
#endif
        }

        private static IntPtr TryLoadOptional(string path)
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
                return IntPtr.Zero;

            string fullPath = NormalizePath(path);
            IntPtr handle = LoadLibraryEx(fullPath, IntPtr.Zero, LoadWithAlteredSearchPath);
            if (handle == IntPtr.Zero)
                handle = LoadLibrary(fullPath);

            if (handle == IntPtr.Zero)
            {
                int error = Marshal.GetLastWin32Error();
                Debug.LogWarning("Optional native dependency failed to load: " + fullPath + " (Win32 error=" + error + ")");
            }

            return handle;
#else
            _ = path;
            return IntPtr.Zero;
#endif
        }

        private static string FindCompatibleBundledDirectMl(string dllDir)
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            if (string.IsNullOrWhiteSpace(dllDir))
                return string.Empty;

            string bundledDirectMl = NormalizePath(Path.Combine(dllDir, "DirectML.dll"));
            if (!File.Exists(bundledDirectMl))
                return string.Empty;

            ushort processMachine = GetCurrentProcessPeMachine();
            if (IsPeMachineCompatible(bundledDirectMl, processMachine, out ushort dllMachine))
                return bundledDirectMl;

            AppendWarning(
                "Bundled DirectML.dll machine=0x" + dllMachine.ToString("X4")
                + " is not compatible with current process architecture ("
                + RuntimeInformation.ProcessArchitecture + ") and will be ignored: " + bundledDirectMl + ".");
            return string.Empty;
#else
            _ = dllDir;
            return string.Empty;
#endif
        }

        private static string FindOnnxRuntimeDll()
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            string dataPath = Application.dataPath;
            string projectRoot = Path.GetFullPath(Path.Combine(dataPath, ".."));
            string[] archFolders = GetNativeArchFolderCandidates();
            var candidates = new System.Collections.Generic.List<string>(32);

            for (int i = 0; i < archFolders.Length; i++)
            {
                string archFolder = archFolders[i];
                if (string.IsNullOrWhiteSpace(archFolder))
                    continue;

                candidates.Add(Path.Combine(projectRoot, "Packages", "com.willkyu.onnxruntime-inference", "Runtime", "Plugins", "Windows", archFolder, "onnxruntime.dll"));
                candidates.Add(Path.Combine(projectRoot, "Packages", "com.willkyu.onnxruntime", "Runtime", "Plugins", "Windows", archFolder, "onnxruntime.dll"));
                candidates.Add(Path.Combine(dataPath, "Plugins", "OnnxRuntime", "Windows", archFolder, "onnxruntime.dll"));
                candidates.Add(Path.Combine(dataPath, "Plugins", "Windows", archFolder, "onnxruntime.dll"));
                candidates.Add(Path.Combine(dataPath, "Plugins", archFolder, "onnxruntime.dll"));
            }

            candidates.Add(Path.Combine(dataPath, "Plugins", "onnxruntime.dll"));
            candidates.Add(Path.Combine(dataPath, "..", "onnxruntime.dll"));

            for (int i = 0; i < candidates.Count; i++)
            {
                string candidate = Path.GetFullPath(candidates[i]);
                if (File.Exists(candidate))
                    return candidate;
            }

            return string.Empty;
#else
            return string.Empty;
#endif
        }

        private static string GetModuleFilePath(IntPtr module)
        {
#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            if (module == IntPtr.Zero)
                return string.Empty;

            var sb = new StringBuilder(1024);
            int len = GetModuleFileName(module, sb, sb.Capacity);
            return len > 0 ? sb.ToString(0, len) : string.Empty;
#else
            _ = module;
            return string.Empty;
#endif
        }

        private static string NormalizePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return string.Empty;

            return Path.GetFullPath(path).TrimEnd('\\', '/');
        }

        private static bool PathEquals(string a, string b)
        {
            return string.Equals(NormalizePath(a), NormalizePath(b), StringComparison.OrdinalIgnoreCase);
        }

        private static void AppendWarning(string message)
        {
            if (string.IsNullOrWhiteSpace(message))
                return;

            loadWarning = string.IsNullOrWhiteSpace(loadWarning)
                ? message
                : loadWarning + " " + message;

            Debug.LogWarning(message);
        }

        private static string[] GetNativeArchFolderCandidates()
        {
            switch (RuntimeInformation.ProcessArchitecture)
            {
                case Architecture.X64:
                    return new[] { "x86_64", "x64", "amd64" };
                case Architecture.Arm64:
                    return new[] { "ARM64", "arm64" };
                case Architecture.X86:
                    return new[] { "x86", "Win32" };
                default:
                    return new[] { "x86_64", "ARM64", "x86" };
            }
        }

        private static ushort GetCurrentProcessPeMachine()
        {
            switch (RuntimeInformation.ProcessArchitecture)
            {
                case Architecture.X64:
                    return ImageFileMachineAmd64;
                case Architecture.Arm64:
                    return ImageFileMachineArm64;
                case Architecture.X86:
                    return ImageFileMachineI386;
                default:
                    return 0;
            }
        }

        private static bool IsPeMachineCompatible(string path, ushort processMachine, out ushort machine)
        {
            if (!TryReadPeMachine(path, out machine))
                return false;
            if (processMachine == 0)
                return true;

            return machine == processMachine;
        }

        private static bool TryReadPeMachine(string path, out ushort machine)
        {
            machine = 0;

            try
            {
                using FileStream stream = File.OpenRead(path);
                if (stream.Length < 0x40)
                    return false;

                Span<byte> mzHeader = stackalloc byte[64];
                if (stream.Read(mzHeader) < mzHeader.Length)
                    return false;

                int peOffset = BitConverter.ToInt32(mzHeader.Slice(0x3C, 4));
                if (peOffset < 0 || peOffset > stream.Length - 6)
                    return false;

                stream.Position = peOffset + 4;
                Span<byte> machineBytes = stackalloc byte[2];
                if (stream.Read(machineBytes) < machineBytes.Length)
                    return false;

                machine = BitConverter.ToUInt16(machineBytes);
                return true;
            }
            catch
            {
                return false;
            }
        }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr LoadLibrary(string lpFileName);

        [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hFile, uint dwFlags);

        [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr GetModuleHandle(string lpModuleName);

        [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern int GetModuleFileName(IntPtr hModule, StringBuilder lpFilename, int nSize);
#endif
    }
}
