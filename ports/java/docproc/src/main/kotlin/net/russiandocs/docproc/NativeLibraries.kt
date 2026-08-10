package net.russiandocs.docproc

import java.io.File

/**
 * Loads the native libraries the pipeline needs, and fails with an explanation rather than a
 * `UnsatisfiedLinkError` when it cannot.
 *
 * **This file exists because of two findings from the Phase-0 spike, and both cost hours to
 * diagnose from an error message that named nothing useful.** They are the JVM's counterpart to
 * `D-09` in the Go port — the same class of defect (a DLL resolved by base name to the wrong copy),
 * in two directions at once.
 *
 * ### 1. A MinGW-built OpenCV binds to `System32`'s runtime, not to the toolchain's
 *
 * `libopencv_java4130.dll` imports `libstdc++-6.dll` and `libgcc_s_seh-1.dll`. Windows resolves
 * those **by base name**: the executable's directory first, then `System32`, then `PATH`. On a
 * machine where `System32` carries its own copies — as the one this was written on does, for all
 * four — the foreign copies win and the load fails with:
 *
 * ```
 * UnsatisfiedLinkError: libopencv_java4130.dll: The specified procedure could not be found
 * ```
 *
 * The Go port's fix was to ship the DLLs beside the binary. That is not available here, because the
 * "binary" is `java.exe` and copying files into a JDK installation is not something a deployment can
 * do. **Loading them explicitly first works**: each [System.load] registers the module under its base
 * name, so the later import binds to the copy we chose. Verified on a clean JDK.
 *
 * ### 2. The JDK's own C runtime can be too old for ONNX Runtime — and this one cannot be fixed here
 *
 * `C:\Program Files\Java\jdk-21\bin` ships `msvcp140.dll` at 14.31 (VS 2022 17.1, early 2022), and
 * the directory of the executable is searched first. `jvm.dll` therefore loads that CRT before
 * `main()` runs, and `onnxruntime.dll` — built against a newer one — fails `DllMain` with Windows
 * error 1114:
 *
 * ```
 * UnsatisfiedLinkError: onnxruntime.dll: A dynamic link library (DLL) initialization routine failed
 * ```
 *
 * It survives every plausible remedy — the CPU artefact fails identically to the GPU one, loading
 * from disk rather than `%TEMP%` changes nothing, a scrubbed `PATH` changes nothing, every import is
 * present and current, and the very same file loads fine in a .NET process. Only replacing the JDK's
 * three CRT files makes it work.
 *
 * **So this class cannot repair it — by the time any Kotlin runs, the old CRT is already in the
 * process.** What it can do is [checkWindowsRuntime], which turns error 1114 into a sentence naming
 * the JDK and the fix. Neither problem exists on Linux or in Docker.
 */
public object NativeLibraries {

    /** Set once, so a second [load] in the same process is a no-op rather than a second attempt. */
    @Volatile
    private var loaded = false

    private val gate = Any()

    /**
     * The MinGW runtime, in dependency order.
     *
     * Order matters: `libstdc++` needs `libgcc` and `libwinpthread`, and loading a dependent first
     * would let the loader satisfy it from `System32` before we get to choose.
     */
    private val toolchainRuntime = listOf(
        "libwinpthread-1.dll",
        "libgcc_s_seh-1.dll",
        "libstdc++-6.dll",
        "zlib1.dll",
    )

    /**
     * Loads OpenCV's JNI library, preloading the toolchain runtime first when one is configured.
     *
     * @param openCvHome the directory holding the built OpenCV, as passed to the build
     * @param toolchainBin where the toolchain's runtime DLLs live, e.g. `C:\msys64\mingw64\bin`.
     *   Ignored on non-Windows, and ignored when the OpenCV build is not MinGW-based.
     */
    @JvmStatic
    public fun load(openCvHome: String? = null, toolchainBin: String? = null) {
        if (loaded) {
            return
        }
        synchronized(gate) {
            if (loaded) {
                return
            }
            val bin = toolchainBin ?: System.getenv("RDOCS_TOOLCHAIN_BIN")
            if (bin != null && isWindows()) {
                for (dependency in toolchainRuntime) {
                    val file = File(bin, dependency)
                    if (file.isFile) {
                        System.load(file.absolutePath)
                    }
                }
            }

            System.load(findOpenCvJni(openCvHome).absolutePath)
            loaded = true
        }
    }

    /**
     * The JNI library, under whichever of the three names the building toolchain produced.
     *
     * **The name is not stable across toolchains**, and hardcoding one spelling is a packaging
     * failure that only appears on the machine that did not do the build: MinGW gives
     * `libopencv_java4130.dll`, MSVC `opencv_java4130.dll`, Linux `libopencv_java4130.so`.
     * `Core.NATIVE_LIBRARY_NAME` supplies the base name without prefix or extension.
     */
    private fun findOpenCvJni(openCvHome: String?): File {
        val base = org.opencv.core.Core.NATIVE_LIBRARY_NAME
        val home = openCvHome ?: System.getenv("RDOCS_OPENCV_HOME")
        val roots = buildList {
            if (home != null) {
                add(File(home, "lib"))
                add(File(home, "bin"))
                add(File(home))
                add(File(home, "share/java/opencv4"))
            }
        }
        val names = listOf("lib$base.dll", "$base.dll", "lib$base.so", "lib$base.dylib")

        for (root in roots) {
            for (name in names) {
                val candidate = File(root, name)
                if (candidate.isFile) {
                    return candidate
                }
            }
        }
        throw IllegalStateException(
            """
            OpenCV's JNI library ($base) was not found.

            This port builds OpenCV itself — there is no official org.opencv artefact, and 4.13.0 is
            required because contour approximation changed in 4.8 and the goldens encode 4.13
            behaviour. Point RDOCS_OPENCV_HOME at the build output.

            Looked under:
            ${roots.joinToString("\n            ") { it.path }}
            """.trimIndent(),
        )
    }

    /**
     * Initialises ONNX Runtime, and turns its one cryptic Windows failure into a sentence.
     *
     * **Diagnosis on failure, not a warning up front**, and the difference matters. The first version
     * of this warned whenever the JDK bundled a CRT at all — which is nearly every JDK, including the
     * ones that work — so it fired on a healthy setup and would have been muted within a week. A check
     * that cries wolf is worse than no check: it trains the reader to ignore the one time it is right.
     *
     * There is no reliable way to read a Windows PE version resource from the JVM, so the version
     * cannot be compared up front. But the failure is unmistakable when it happens, and that is a
     * better trigger: [UnsatisfiedLinkError] naming `onnxruntime.dll`.
     */
    @JvmStatic
    public fun loadOnnxRuntime() {
        try {
            ai.onnxruntime.OrtEnvironment.getEnvironment()
        } catch (e: UnsatisfiedLinkError) {
            val bundled = File(System.getProperty("java.home").orEmpty(), "bin/msvcp140.dll")
            if (isWindows() && bundled.isFile) {
                throw UnsatisfiedLinkError(
                    """
                    ${e.message}

                    Windows error 1114 (ERROR_DLL_INIT_FAILED): onnxruntime.dll loaded but its DllMain
                    failed. On this platform that is almost always the JDK's own C runtime.

                    This JDK bundles one at ${bundled.path}, and the directory of java.exe is searched
                    BEFORE System32 — so jvm.dll loads that copy before any of this code runs, and ONNX
                    Runtime, built against a newer runtime, cannot initialise against it.

                    Measured during the port's spike: a JDK shipping 14.31.31103.0 (VS 2022 17.1) fails
                    while the system had 14.50.35719.0, and the SAME onnxruntime.dll works as soon as
                    msvcp140.dll, vcruntime140.dll and vcruntime140_1.dll are replaced with the
                    system's. The same file also loads fine in a .NET process, which is how the JDK was
                    identified as the variable.

                    Fix: use a JDK whose bundled runtime is current — Temurin, Zulu or Corretto builds
                    of 21 track a newer redistributable than Oracle 21.0.1 does.

                    Cannot occur on Linux or in Docker: there is no such shadowing.
                    """.trimIndent(),
                ).apply { initCause(e) }
            }
            throw e
        }
    }

    private fun isWindows(): Boolean =
        System.getProperty("os.name").orEmpty().startsWith("Windows", ignoreCase = true)
}
