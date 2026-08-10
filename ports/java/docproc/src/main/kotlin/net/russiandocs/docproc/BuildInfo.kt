package net.russiandocs.docproc

/**
 * What this build actually is, for the conformance sidecar and the status page.
 *
 * **Reported rather than configured.** Every value is read from the thing itself at runtime, and the
 * reason is the whole point of the sidecar: more than half of the plausible ways a port can diverge
 * are a version mismatch or a thread-count difference wearing the costume of a numeric difference.
 * A hardcoded version string would describe the intent instead of the run.
 */
public object BuildInfo {

    public val javaVersion: String
        get() = "${System.getProperty("java.vendor")} ${System.getProperty("java.version")}"

    public val kotlinVersion: String
        get() = KotlinVersion.CURRENT.toString()

    /**
     * ONNX Runtime's version.
     *
     * **An INSTANCE method on the JVM binding**, unlike the static accessors the Python and .NET
     * packages expose — so a port copying the idiom from either of those does not compile. Noted
     * because it is exactly the kind of small asymmetry that gets "fixed" by hardcoding a literal.
     */
    public val onnxRuntimeVersion: String
        get() = ai.onnxruntime.OrtEnvironment.getEnvironment().version

    /**
     * The execution providers ONNX Runtime *offers*, under their CANONICAL names.
     *
     * **`getName()`, not `name`.** The JVM binding's enum constants are `CPU`, `CUDA`, `TENSOR_RT`,
     * while ONNX Runtime's own names — the ones Python and .NET return, and the ones
     * `conformance/spec/cli.md` shows — are `CPUExecutionProvider`, `CUDAExecutionProvider`. Using the
     * enum's Kotlin name would have this port report a different vocabulary than its three siblings
     * for the same machine, which is the kind of divergence that gets noticed only when somebody
     * diffs two sidecars and cannot tell whether the hosts differ.
     *
     * **A listed provider is not a working GPU** — this is what the runtime advertises, and on a host
     * with the GPU artefact installed but no usable device it still contains CUDA. What actually bound
     * is reported separately, after a session has really been built on it.
     */
    public val availableProviders: List<String>
        get() = ai.onnxruntime.OrtEnvironment.getAvailableProviders().map { it.getName() }

    public val openCvVersion: String
        get() = org.opencv.core.Core.VERSION

    /**
     * The commit, from the environment.
     *
     * Empty rather than "unknown" when unset, so the sidecar can tell "not stamped" from a commit
     * literally named unknown. The Go port shipped a run with an empty commit field for a while, and
     * that is precisely the provenance you need when a divergence appears months later.
     */
    public val commit: String
        get() = System.getenv("RDOCS_COMMIT").orEmpty()
}
