using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace RussianDocs.Conform;

/// <summary>
/// Version and provider reporting for <c>info</c>.
///
/// <para>
/// Each probe touches the NATIVE half of its dependency rather than reading a managed assembly
/// version, because that is the thing that actually fails: a managed/native version mismatch in
/// OpenCvSharp shows up as a missing entry point, and a missing CUDA library shows up as a provider
/// that is listed but unusable. Making <c>info</c> exercise both means the cheapest command in the
/// CLI is also the one that proves the deployment.
/// </para>
///
/// <para>
/// Every probe returns a string instead of throwing. A port whose <c>info</c> crashes cannot tell
/// the checker anything at all, including which of its two native dependencies is broken.
/// </para>
/// </summary>
internal static class Probes
{
    /// <summary>OpenCV's own version string, from the native library.</summary>
    internal static string OpenCvVersion()
    {
        try
        {
            // Cv2.GetVersionString() forces the native load, which is the point: a broken
            // OpenCvSharp native package is discovered here and not at the first Mat.
            // It is declared as returning a nullable string, so the fallback is explicit rather
            // than suppressed — an empty version is itself a symptom worth seeing.
            return Cv2.GetVersionString() ?? "unknown";
        }
        catch (Exception ex)
        {
            return $"unavailable: {ex.GetType().Name}: {ex.Message}";
        }
    }

    /// <summary>ONNX Runtime's version, from the native library.</summary>
    internal static string OnnxRuntimeVersion()
    {
        try
        {
            return OrtEnv.Instance().GetVersionString();
        }
        catch (Exception ex)
        {
            return $"unavailable: {ex.GetType().Name}: {ex.Message}";
        }
    }

    /// <summary>
    /// Execution providers ONNX Runtime reports as AVAILABLE.
    ///
    /// <para>
    /// **Available is not the same as working, and the difference has already cost this project
    /// real time.** `CUDAExecutionProvider` appears in this list whenever the GPU build is
    /// installed, with no GPU, no driver and no cuDNN present; in a container started without
    /// `--gpus` the provider then SEGFAULTS instead of raising. So this list is diagnostic only —
    /// device selection uses a device probe plus a `[gpu, cpu]` attempt loop, and reports the
    /// provider it actually bound (D-13).
    /// </para>
    /// </summary>
    internal static string[] AvailableProviders()
    {
        try
        {
            return OrtEnv.Instance().GetAvailableProviders();
        }
        catch (Exception ex)
        {
            return [$"unavailable: {ex.GetType().Name}"];
        }
    }
}
