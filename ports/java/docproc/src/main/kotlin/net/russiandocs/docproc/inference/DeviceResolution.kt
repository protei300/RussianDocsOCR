package net.russiandocs.docproc.inference

import net.russiandocs.docproc.pipeline.Device
import java.io.File

/**
 * Deciding which device to actually use, and refusing to find out the hard way.
 *
 * Two rules, both carried from `service/ml/runtime.py` and both re-proved by the Go port:
 *
 * 1. **A listed CUDA provider does not mean a working GPU.** `getAvailableProviders()` reports what the
 *    artifact was BUILT with, so the GPU jar lists CUDA on a machine with no driver at all.
 * 2. **In a container started without `--gpus`, creating a CUDA session SEGFAULTS rather than throwing.**
 *    No try/catch can save the process from that, which is why the device-node probe runs FIRST.
 */
public object DeviceResolution {

    /**
     * Whether a GPU is visible to THIS process.
     *
     * Checks for the device nodes rather than asking the driver, because the question is precisely whether
     * the container was given access — and a library call that answers it is the call that crashes.
     *
     * `/dev/nvidiactl` and `/dev/nvidia0` are the Linux passthrough; `/dev/dxg` is WSL2's. On Windows the
     * probe returns whatever the provider list says, because there is no container boundary to be on the
     * wrong side of and no device node to inspect.
     */
    @JvmStatic
    public fun gpuVisible(): Boolean {
        if (isWindows()) {
            return providerList().any { it.contains("CUDA") }
        }
        // File.exists() is enough here: these are character devices, and unlike some APIs the JVM's
        // File.exists does report them. (In .NET the equivalent needed Path.Exists rather than
        // File.Exists, which answers false for a device node — worth noting because the two look alike.)
        return listOf("/dev/nvidiactl", "/dev/nvidia0", "/dev/dxg").any { File(it).exists() }
    }

    /** The providers ONNX Runtime ADVERTISES. Not what it can actually bind — see the type note. */
    @JvmStatic
    public fun providerList(): List<String> =
        try {
            ai.onnxruntime.OrtEnvironment.getAvailableProviders().map { it.getName() }
        } catch (e: Throwable) {
            // A runtime that cannot even enumerate its providers is a CPU-only situation as far as the
            // caller is concerned; the real failure surfaces when a session is built.
            listOf("CPUExecutionProvider")
        }

    /**
     * Resolves a requested device into an ordered list of attempts.
     *
     * `auto` becomes GPU-then-CPU when a device is visible and CPU alone when it is not. An explicit `gpu`
     * that cannot be honoured falls back to CPU with a loud message rather than crashing — the reference logs
     * and continues, and a service that refuses to start cannot explain why it refused.
     *
     * An explicit `cpu` is honoured EXACTLY and probes nothing. "Give me CPU" is sometimes a correctness
     * requirement rather than a preference: the conformance goldens were generated on CPU.
     */
    @JvmStatic
    public fun resolve(requested: String, log: (String) -> Unit = {}): List<Device> = when (requested) {
        "cpu" -> listOf(Device.CPU)
        "gpu" -> if (gpuVisible()) {
            listOf(Device.GPU, Device.CPU)
        } else {
            log(
                "compute_device=gpu but no GPU is visible to this process — refusing to attempt CUDA, " +
                    "which would TERMINATE the process rather than fail cleanly. Using CPU. " +
                    "In Docker, pass --gpus all.",
            )
            listOf(Device.CPU)
        }
        "auto" -> if (gpuVisible()) listOf(Device.GPU, Device.CPU) else listOf(Device.CPU)
        else -> throw IllegalArgumentException(
            "device must be auto, cpu or gpu, got \"$requested\"")
    }

    private fun isWindows(): Boolean =
        System.getProperty("os.name").orEmpty().startsWith("Windows", ignoreCase = true)
}
