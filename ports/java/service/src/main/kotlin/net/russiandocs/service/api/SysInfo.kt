package net.russiandocs.service.api

import java.io.File
import java.lang.management.ManagementFactory
import java.util.Locale
import java.util.concurrent.TimeUnit
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** The host block. JSON names match the frontend exactly. */
@Serializable
public data class ServerStats(
    @SerialName("cpu_pct") val cpuPct: Double = 0.0,
    @SerialName("cpu_name") val cpuName: String = "",
    @SerialName("cpu_cores") val cpuCores: Int = 0,
    @SerialName("cpu_threads") val cpuThreads: Int = 0,
    @SerialName("ram_used_gb") val ramUsedGb: Double = 0.0,
    @SerialName("ram_total_gb") val ramTotalGb: Double = 0.0,
    @SerialName("disk_used_gb") val diskUsedGb: Double = 0.0,
    @SerialName("disk_total_gb") val diskTotalGb: Double = 0.0,
)

/** The accelerator block, or `null` when there is no GPU to describe. */
@Serializable
public data class GpuStats(
    @SerialName("name") val name: String = "",
    @SerialName("utilization_pct") val utilizationPct: Int = 0,
    @SerialName("vram_used_gb") val vramUsedGb: Double = 0.0,
    @SerialName("vram_total_gb") val vramTotalGb: Double = 0.0,
    @SerialName("temperature_c") val temperatureC: Int = 0,
)

/**
 * Host CPU, memory, disk and GPU for the status page.
 *
 * **The field names here are fixed by the SHARED FRONTEND, not chosen.** `web/` is reused unchanged by
 * every port, so `web/src/views/pages/status/Index.vue` is the contract: it reads `server.cpu_pct`,
 * `server.cpu_name`, `server.ram_used_gb` and the rest by name. An earlier version of the Go port returned
 * a thinner, more idiomatic block on the reasoning that pulling in a dependency to render a CPU gauge was a
 * poor trade — and the status page rendered completely empty. The lesson is worth writing down: when a UI is
 * shared, the UI owns the wire format.
 *
 * **Every probe is INDIVIDUALLY GUARDED and degrades to a zero value.** A service that cannot describe its
 * own host must still recognise documents, so nothing in here may throw at a caller.
 *
 * Port of the `_server_stats` / `_gpu_stats` helpers in `service/api/status.py`.
 */
public object SysInfo {

    private val os = ManagementFactory.getOperatingSystemMXBean()
    private val cpuGate = Any()
    private var lastCpuNanos = 0L
    private var lastStampNanos = 0L

    public fun readServer(): ServerStats {
        val (ramUsed, ramTotal) = memory()
        val (diskUsed, diskTotal) = disk()
        return ServerStats(
            cpuName = cpuName(),
            // The JVM exposes only logical processors portably. Reporting the same number for both is
            // honest — the alternative is a per-platform probe (WMI on Windows, /proc/cpuinfo on Linux) for
            // a figure the status page shows and nothing acts on.
            cpuCores = Runtime.getRuntime().availableProcessors(),
            cpuThreads = Runtime.getRuntime().availableProcessors(),
            cpuPct = cpuPercent(),
            ramUsedGb = ramUsed,
            ramTotalGb = ramTotal,
            diskUsedGb = diskUsed,
            diskTotalGb = diskTotal,
        )
    }

    /**
     * Process CPU usage since the previous call, as a percentage of one core-second per wall second,
     * scaled by processor count.
     *
     * The FIRST call returns 0 by construction: there is no earlier sample to difference against, and
     * inventing one from process start time would report an average over the whole lifetime, which is not
     * what a live gauge means.
     *
     * The CPU time comes from `com.sun.management.OperatingSystemMXBean` by reflection rather than a cast.
     * That interface is present on HotSpot but is NOT part of the `java.lang.management` contract, so a
     * hard cast is a `ClassCastException` on a JVM that omits it — and a status gauge must never be able to
     * take down the page it appears on.
     */
    private fun cpuPercent(): Double = synchronized(cpuGate) {
        try {
            val method = os.javaClass.getMethod("getProcessCpuTime")
            method.isAccessible = true
            val cpuNanos = (method.invoke(os) as? Long) ?: return 0.0
            val stamp = System.nanoTime()
            if (lastStampNanos == 0L) {
                lastCpuNanos = cpuNanos
                lastStampNanos = stamp
                return 0.0
            }
            val elapsed = (stamp - lastStampNanos) / 1e9
            val used = (cpuNanos - lastCpuNanos) / 1e9
            lastCpuNanos = cpuNanos
            lastStampNanos = stamp
            if (elapsed <= 0) {
                return 0.0
            }
            val cores = Runtime.getRuntime().availableProcessors()
            round1((used / elapsed / cores * 100).coerceIn(0.0, 100.0))
        } catch (e: Exception) {
            0.0
        }
    }

    private fun cpuName(): String = try {
        val cpuinfo = File("/proc/cpuinfo")
        if (cpuinfo.isFile) {
            cpuinfo.useLines { lines ->
                lines.firstOrNull { it.startsWith("model name") }
                    ?.substringAfter(':')
                    ?.let { trimSpaces(it) }
            } ?: trimSpaces(System.getenv("PROCESSOR_IDENTIFIER") ?: os.arch)
        } else {
            // Both the registry and /proc/cpuinfo pad CPU names, sometimes in the middle, so the value
            // goes through trimSpaces either way.
            trimSpaces(System.getenv("PROCESSOR_IDENTIFIER") ?: os.arch)
        }
    } catch (e: Exception) {
        ""
    }

    private fun memory(): Pair<Double, Double> = try {
        val meminfo = File("/proc/meminfo")
        if (meminfo.isFile) {
            var totalKb = 0.0
            var availableKb = 0.0
            meminfo.forEachLine { line ->
                when {
                    line.startsWith("MemTotal:") -> totalKb = parseMeminfoKb(line)
                    line.startsWith("MemAvailable:") -> availableKb = parseMeminfoKb(line)
                }
            }
            if (totalKb > 0) {
                gb(((totalKb - availableKb) * 1024).toLong()) to gb((totalKb * 1024).toLong())
            } else {
                heapFallback()
            }
        } else {
            heapFallback()
        }
    } catch (e: Exception) {
        0.0 to 0.0
    }

    /**
     * Without a host probe the honest answer is this process's heap against its maximum — a smaller number
     * than the host's, and labelled as used rather than pretended to be host-wide.
     *
     * Note this UNDERSTATES the real figure on this service specifically: 215 MB of ONNX weights and every
     * OpenCV `Mat` live outside the Java heap, so the JVM's own accounting cannot see them. The soak
     * measurement that matters (RSS) is taken from the OS by the conformance CLI, not from here.
     */
    private fun heapFallback(): Pair<Double, Double> {
        val runtime = Runtime.getRuntime()
        return gb(runtime.totalMemory() - runtime.freeMemory()) to gb(runtime.maxMemory())
    }

    private fun parseMeminfoKb(line: String): Double {
        val parts = line.split(Regex("\\s+")).filter { it.isNotEmpty() }
        return if (parts.size >= 2) parts[1].toDoubleOrNull() ?: 0.0 else 0.0
    }

    private fun disk(): Pair<Double, Double> = try {
        val root = File(".").absoluteFile
        val total = root.totalSpace
        total.let { gb(it - root.usableSpace) to gb(it) }
    } catch (e: Exception) {
        0.0 to 0.0
    }

    /**
     * Queries the GPU through `nvidia-smi`, returning `null` when there is none.
     *
     * **Why a subprocess rather than NVML.** NVML is the proper API and the Python service uses it through
     * pynvml. Reaching it from the JVM means a JNI binding to a library that may not exist, on two
     * platforms, for information that is purely diagnostic and polled by one page. `nvidia-smi` ships with
     * the driver, is present in the CUDA runtime images, and its CSV output is stable across driver
     * generations.
     *
     * The cost is real and bounded: one process spawn per status request, with a hard timeout. If that ever
     * becomes a problem the fix is a cached value with a TTL, not NVML.
     *
     * **Absence is NOT an error.** No GPU, no driver, or a CPU-only container all mean `null`, and the
     * status page then shows the compute block alone — which is the part that answers whether the GPU is
     * actually being used.
     */
    public fun readGpu(): GpuStats? = try {
        val process = ProcessBuilder(
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ).redirectErrorStream(false).start()

        val output = process.inputStream.bufferedReader().readText()
        if (!process.waitFor(3, TimeUnit.SECONDS) || process.exitValue() != 0) {
            process.destroyForcibly()
            null
        } else {
            // The first line only: a multi-GPU host reports one row per device, and the pipeline pins
            // device 0.
            val line = output.lineSequence().firstOrNull()?.trim() ?: ""
            val parts = line.split(',').map { it.trim() }
            if (line.isEmpty() || parts.size < 5) {
                null
            } else {
                // memory.* is reported in MiB with `nounits`.
                GpuStats(
                    name = parts[0],
                    utilizationPct = parts[1].toIntOrNull() ?: 0,
                    vramUsedGb = round1((parts[2].toDoubleOrNull() ?: 0.0) * 1024 * 1024 / 1e9),
                    vramTotalGb = round1((parts[3].toDoubleOrNull() ?: 0.0) * 1024 * 1024 / 1e9),
                    temperatureC = parts[4].toIntOrNull() ?: 0,
                )
            }
        }
    } catch (e: Exception) {
        null
    }

    /**
     * Bytes to gigabytes at one decimal.
     *
     * DECIMAL gigabytes (1e9), matching the Python service, so every implementation reports the same number
     * for the same machine. Not GiB — a status page saying 32.0 GB for a 32 GB stick is what an operator
     * expects, whatever the pedantically correct unit is.
     */
    private fun gb(bytes: Long): Double = round1(bytes / 1e9)

    internal fun round1(v: Double): Double = (v * 10 + 0.5).toInt() / 10.0

    private fun trimSpaces(s: String): String =
        s.split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
}
