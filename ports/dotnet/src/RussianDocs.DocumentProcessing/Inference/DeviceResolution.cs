using System.Runtime.InteropServices;

namespace RussianDocs.DocumentProcessing.Inference;

/// <summary>
/// Deciding which device to actually use.
///
/// <para>
/// This is the "GPU if available, else CPU" logic, and the reason it is more than a try/catch is
/// worth stating plainly: **in a container started without <c>--gpus</c>, the CUDA provider
/// SEGFAULTS rather than throwing.** No exception handler can rescue that, so the device must be
/// probed before the provider is ever asked for.
/// </para>
/// </summary>
public static class DeviceResolution
{
    /// <summary>
    /// Whether a GPU is visible to THIS process.
    ///
    /// <para>
    /// On Windows the answer is always yes: the driver is a system component and there is no device
    /// node to look for, so the attempt loop is the only check available. On Linux the device nodes
    /// are the honest signal — and they are what distinguishes "the host has a GPU" from "this
    /// container was given one", which is the case that segfaults.
    /// </para>
    ///
    /// <para>
    /// <c>/dev/dxg</c> is in the list for WSL2, where CUDA works through a paravirtualised device
    /// rather than <c>/dev/nvidia0</c>.
    /// </para>
    /// </summary>
    public static bool GpuVisible()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return true;
        }
        foreach (string node in new[] { "/dev/nvidiactl", "/dev/dxg", "/dev/nvidia0" })
        {
            // Path.Exists rather than File.Exists: these are CHARACTER DEVICES, and File.Exists
            // returns false for them on some runtimes.
            if (Path.Exists(node))
            {
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Resolves the requested device to one that works, and says WHY.
    ///
    /// <para>
    /// The reason string is returned rather than logged here, so the caller can put it in its own log
    /// with its own prefix — and so this function stays testable without a logger.
    /// </para>
    /// </summary>
    /// <param name="requested">What the operator asked for.</param>
    /// <param name="probe">
    /// Builds something on a device and throws if it cannot. Passed in rather than called directly
    /// because "usable" means "the twelve real sessions construct", which only the caller knows how
    /// to attempt.
    /// </param>
    public static (Device Device, string Reason) Resolve(Device requested, Action<Device> probe)
    {
        // **A CPU request is honoured exactly, with no probing.** Sometimes "give me CPU" is a
        // correctness requirement rather than a preference — the conformance goldens were generated
        // on CPU — so silently upgrading it would be wrong.
        if (requested != Device.Gpu)
        {
            probe(Device.Cpu);
            return (Device.Cpu, "cpu requested");
        }

        if (!GpuVisible())
        {
            probe(Device.Cpu);
            return (Device.Cpu,
                "gpu requested but no device is visible to this process; CUDA was NOT attempted, " +
                "because without a device the provider can terminate the process instead of " +
                "returning an error");
        }

        try
        {
            probe(Device.Gpu);
            return (Device.Gpu, "gpu requested and available");
        }
        catch (Exception gpuError)
        {
            // A listed provider is not a working GPU: CUDAExecutionProvider appears whenever the GPU
            // package is installed, with no driver and no cuDNN present. Falling back is the point of
            // the loop.
            try
            {
                probe(Device.Cpu);
            }
            catch (Exception cpuError)
            {
                throw new InvalidOperationException(
                    $"inference: neither device usable; cpu: {cpuError.Message} " +
                    $"(gpu: {gpuError.Message})", cpuError);
            }
            return (Device.Cpu, $"gpu requested but unusable, fell back to cpu: {gpuError.Message}");
        }
    }
}
