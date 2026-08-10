package net.russiandocs.conform

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import net.russiandocs.docproc.BuildInfo
import net.russiandocs.docproc.NativeLibraries
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.pipeline.DirectoryStageSink
import net.russiandocs.docproc.pipeline.OcrTier
import net.russiandocs.docproc.pipeline.Recognizer
import net.russiandocs.docproc.pipeline.RunOptions
import net.russiandocs.docproc.viewmodel.Payload
import java.io.File
import kotlin.system.exitProcess

/**
 * The conformance CLI. Port of this contract's Go and .NET equivalents; the contract itself is
 * `conformance/spec/cli.md`.
 *
 * ## Exit codes are the interface, not an afterthought
 *
 * | code | meaning | the checker's reaction |
 * |---|---|---|
 * | 0 | ran | compare the output |
 * | **2** | **not implemented** | **SKIP, do not fail** |
 * | 3 | bad input (missing or undecodable image) | report as an input error |
 * | 1 | crashed | fail |
 *
 * **2 is what makes a partial port gradeable**, and getting it wrong in either direction is worse
 * than not having it: returning 1 for an unimplemented stage fails a port for work not yet started,
 * and returning 0 with nothing written makes an absent stage look like a passing one.
 *
 * ## stdout carries only the payload
 *
 * Every diagnostic goes to stderr. Not a stylistic rule: `document_processing` prints to stdout, so
 * the *reference* CLI has to redirect the library's output — and a port that logs to stdout produces
 * unparseable output that looks like a serialisation bug.
 */
private val json = Json { prettyPrint = false; encodeDefaults = true }

/**
 * The stages this port can emit.
 *
 * **It must stay honest.** The checker skips what is not claimed, so an over-claiming list turns a
 * missing stage into a failure with a confusing message, while an under-claiming one silently stops
 * grading work that is finished. The Go port shipped a real defect of the second kind:
 * `borders.segments` was missing from its list, so the REFERENCE skipped that stage in its own
 * self-check while the port was being graded on it — a check that could not fail.
 *
 * Taken from the pipeline itself rather than repeated here, so the two cannot drift.
 */
private val stagesImplemented: List<String> = Recognizer.STAGES_IMPLEMENTED

public fun main(args: Array<String>) {
    if (args.isEmpty()) {
        usage()
        exitProcess(2)
    }
    try {
        val code = when (args[0]) {
            "info" -> cmdInfo()
            "recognize" -> cmdRecognize(args)
            "probe" -> cmdProbe(args)
            "soak" -> cmdSoak(args)
            "--help", "-h", "help" -> { usage(); 0 }
            else -> {
                System.err.println("unknown subcommand ${args[0]}")
                usage()
                2
            }
        }
        exitProcess(code)
    } catch (e: FlagException) {
        System.err.println("rdocs-conform: ${e.message}")
        exitProcess(2)
    } catch (e: Throwable) {
        // Printed to stderr with the stack trace, because a crash the checker reports as exit 1 is
        // useless without one.
        System.err.println("rdocs-conform: ${e.message}")
        e.printStackTrace()
        exitProcess(1)
    }
}

private class FlagException(message: String) : Exception(message)

/**
 * stdout, forced to UTF-8.
 *
 * **`System.out` is NOT UTF-8 on Windows and `println` silently mangles Cyrillic.** JDK 18 made
 * `file.encoding` default to UTF-8, but the CONSOLE stream keeps the platform code page — so every Russian
 * character in the view model came out as `?`. The failure is maximally misleading: the OCR was right, the
 * JSON was well-formed, and the checker reported `'ВАСИЛЬЕВА' vs '?????????'`, which reads as a decoder bug
 * in the very component that had just been proved correct.
 *
 * One PrintStream, used for every payload write. Not a JVM flag: `-Dstdout.encoding=UTF-8` would work on the
 * developer's machine and be absent wherever the checker launches the jar itself.
 */
private val stdout = java.io.PrintStream(
    java.io.FileOutputStream(java.io.FileDescriptor.out), true, Charsets.UTF_8)

private fun usage() {
    System.err.println(
        """
        rdocs-conform info
        rdocs-conform recognize --image <path> [--device cpu|gpu] [--ocr accurate|fast]
                                [--img-size N] [--docconf F] [--include-debug]
        rdocs-conform probe --image <path> --dump-dir <dir> [--upto <stage>] [same flags]
        """.trimIndent(),
    )
}

/**
 * Emits one JSON object describing this implementation.
 *
 * The version fields are the reason the sidecar exists: more than half of the plausible ways a port
 * diverges are a version or thread-count mismatch wearing the costume of a numeric difference.
 *
 * Loading the native libraries here is deliberate — `info` is the cheapest subcommand, so it is where
 * a broken native setup should surface, with the diagnostics `NativeLibraries` prints rather than as
 * a mysterious failure inside `probe`.
 */
private fun cmdInfo(): Int {
    NativeLibraries.load()
    // Touched here so a broken native setup surfaces in the CHEAPEST subcommand, with the diagnosis
    // this method attaches, rather than as a bare error 1114 from inside `probe`.
    NativeLibraries.loadOnnxRuntime()

    val info = InfoPayload(
        port = "java",
        language = "Kotlin ${BuildInfo.kotlinVersion} on ${BuildInfo.javaVersion}",
        versions = mapOf(
            "runtime" to BuildInfo.javaVersion,
            "kotlin" to BuildInfo.kotlinVersion,
            "onnxruntime" to BuildInfo.onnxRuntimeVersion,
            "opencv" to BuildInfo.openCvVersion,
        ),
        device = "cpu",
        ocrDevice = "cpu",
        providers = BuildInfo.availableProviders,
        modelFormat = "ONNX",
        ocrMode = "accurate",
        stagesImplemented = stagesImplemented,
        commit = BuildInfo.commit,
    )
    stdout.println(json.encodeToString(info))
    return 0
}

/**
 * Not implemented at M0 — exit 2, so the checker SKIPS rather than fails.
 *
 * Wired and returning 2 rather than omitted: an absent subcommand makes the checker report the port
 * as broken, which is a different and more alarming thing than "this milestone is not done".
 */
/**
 * Emits the VIEW MODEL on stdout and nothing else.
 *
 * **Shares the pipeline pass with `probe` rather than repeating it.** Two separate paths could diverge, and
 * then a golden would disagree with a live run for a reason that is not a behaviour change — the same
 * argument that makes the reference's `regen` reuse its own `probe`/`recognize`.
 *
 * stdout carries only the payload; every diagnostic goes to stderr. Not a style rule: the checker parses
 * this, and a stray log line makes it a JSON error that looks like a serialisation bug.
 */
private fun cmdRecognize(args: Array<String>): Int {
    val flags = parseFlags(args)
    val image = flags.image ?: throw FlagException("recognize requires --image")

    if (!File(image).isFile) {
        System.err.println("recognize: no such image: $image")
        return 3
    }

    NativeLibraries.load()
    NativeLibraries.loadOnnxRuntime()

    Recognizer(
        device = Device.parse(flags.device),
        intraOpThreads = IntraOpThreads,
        ocrTier = OcrTier.parse(flags.ocr),
    ).use { recognizer ->
        recognizer.run(image, RunOptions(
            docconf = flags.docconf,
            imgSize = flags.imgSize,
            includeDebug = flags.includeDebug,
        )).use { results ->
            val payload = recognizer.buildViewModel(results, flags.includeDebug)
            stdout.println(encodeViewModel(payload))
        }
    }
    return 0
}

/**
 * Serialises the view model: fourteen keys with nulls PRESENT, and `debug` present only when asked for.
 *
 * Two rules that pull in opposite directions, which is why this is a function rather than a serialiser
 * setting:
 *
 * - **`explicitNulls = true` (J-04).** kotlinx.serialization omits nulls by default, and this contract needs
 *   every absent field written as `null` — the SPA distinguishes "not read" from "read as empty", and a
 *   missing key makes a page render blank rather than showing a dash.
 * - **`debug` is the ONE key that must be ABSENT when null**, because the reference omits it rather than
 *   sending null. A serialiser flag cannot express "all nulls except this one", so the key is removed after
 *   encoding. Caught by the checker reporting `viewmodel.debug: extra: not present in the golden` — an extra
 *   key is as much a contract break as a missing one.
 */
private fun encodeViewModel(payload: Payload): String {
    val tree = viewModelJson.encodeToJsonElement(Payload.serializer(), payload).jsonObject
    val trimmed = if (payload.debug == null) {
        JsonObject(tree.filterKeys { it != "debug" })
    } else {
        tree
    }
    return viewModelJson.encodeToString(JsonElement.serializer(), trimmed)
}

private val viewModelJson = Json {
    explicitNulls = true
    encodeDefaults = true
    prettyPrint = false
}

/**
 * Runs the pipeline, writing one file per stage into the dump directory.
 *
 * **A missing or unreadable image is exit 3, not 1.** The checker distinguishes "your port crashed"
 * from "you were handed something you could not read", and conflating them turns a wrong path in the
 * case list into a bug report against the port.
 */
private fun cmdProbe(args: Array<String>): Int {
    val flags = parseFlags(args)
    val image = flags.image ?: throw FlagException("probe requires --image")
    val dumpDir = flags.dumpDir ?: throw FlagException("probe requires --dump-dir")

    if (!File(image).isFile) {
        System.err.println("probe: no such image: $image")
        return 3
    }

    NativeLibraries.load()
    NativeLibraries.loadOnnxRuntime()

    val sink = DirectoryStageSink(dumpDir, flags.upTo)
    try {
        Recognizer(
            device = Device.parse(flags.device),
            intraOpThreads = IntraOpThreads,
            ocrTier = OcrTier.parse(flags.ocr),
        ).use { recognizer ->
            // `use` on the results as well: the canvas and every intermediate are native memory, and a
            // probe run that leaks them is a probe run that cannot be looped over a corpus.
            recognizer.run(image, RunOptions(
                docconf = flags.docconf,
                imgSize = flags.imgSize,
                sink = sink,
                upTo = flags.upTo,
                includeDebug = flags.includeDebug,
            )).use { }
        }
    } catch (e: IllegalArgumentException) {
        // Thrown by the decoder for an unreadable file — input, not a crash.
        System.err.println("probe: ${e.message}")
        return 3
    } finally {
        sink.close()
    }
    return 0
}

/**
 * Pushes a whole directory of documents through ONE Recognizer, several times, reporting RSS between rounds.
 *
 * **This is the check the conformance harness structurally cannot perform.** It runs one document per
 * process, so a path that never releases its intermediates passes every stage and still dies in production —
 * measured in the Go port at 12.7 MB per document, unbounded, with the suite green throughout.
 *
 * A leak and an allocator plateau are indistinguishable in a single measurement. They differ only in the
 * SHAPE of the curve across rounds, which is why the corpus is repeated rather than measured once.
 *
 * **RSS comes from the OS, not from the JVM.** `Runtime.totalMemory` and every heap counter are blind to
 * OpenCV Mats and ONNX Runtime arenas, which is where all of the memory is — the Go port had the identical
 * trap with `runtime.MemStats`. `System.gc()` is called between rounds so the curve measures RETENTION
 * rather than collection lag: on the JVM a missed release is delayed rather than permanent, which looks like
 * a leak and cannot be reasoned about.
 */
private fun cmdSoak(args: Array<String>): Int {
    var dir = "samples"
    var rounds = 4
    var device = "cpu"
    var i = 1
    while (i < args.size) {
        when (args[i]) {
            "--dir" -> dir = next(args, ++i, "--dir")
            "--rounds" -> rounds = int(next(args, ++i, "--rounds"), "--rounds")
            "--device" -> device = oneOf(next(args, ++i, "--device"), "cpu", "gpu")
            else -> throw FlagException("unknown flag ${args[i]}")
        }
        i++
    }

    val files = File(dir).walkTopDown()
        .filter { it.isFile && it.extension.equals("jpg", ignoreCase = true) }
        // Only the per-type subdirectories, matching the corpus the other ports soak over.
        .filter { it.parentFile?.absolutePath != File(dir).absolutePath }
        .sortedBy { it.absolutePath }
        .toList()
    if (files.isEmpty()) {
        throw FlagException("no *.jpg found under $dir")
    }

    NativeLibraries.load()
    NativeLibraries.loadOnnxRuntime()

    Recognizer(device = Device.parse(device), intraOpThreads = IntraOpThreads).use { recognizer ->
        stdout.println("ready, rss=${rss()} MB   (${files.size} documents, $rounds rounds)")

        for (round in 1..rounds) {
            var failed = 0
            for (file in files) {
                try {
                    recognizer.run(file.path, RunOptions()).use { results ->
                        // Force the view model too: it is what the service builds, and building it is where
                        // the Go port's leak actually lived.
                        recognizer.buildViewModel(results, includeDebug = false)
                    }
                } catch (e: Throwable) {
                    failed++
                    System.err.println("[soak] ${file.name}: ${e.message}")
                }
            }

            System.gc()
            Thread.sleep(200)   // let the collector and any finalizers actually run before sampling
            System.gc()

            stdout.println(
                "round $round: ${round * files.size} docs cumulative, rss=${rss()} MB, failed=$failed")
        }
    }
    return 0
}

/** Resident set size in MB, from the OS. */
private fun rss(): Long {
    val pid = ProcessHandle.current().pid()
    return if (System.getProperty("os.name").orEmpty().startsWith("Windows", true)) {
        // tasklist is always present; a WMI query would need another dependency and a shell.
        val out = ProcessBuilder("tasklist", "/FI", "PID eq $pid", "/FO", "CSV", "/NH")
            .redirectErrorStream(true).start().inputStream.bufferedReader().readText()
        // The LAST CSV field is the working set, and it is locale-formatted: this machine reports
        // `"1 408 700 K"` with a NON-BREAKING space as the thousands separator.
        //
        // **Java's `\s` does not match U+00A0**, so a regex built around it silently found nothing and RSS
        // came out as 0 for every round — a soak report that looks like a leak-free run and measures nothing.
        // Stripping every non-digit from the field cannot care what the separator is.
        out.trim().split("\",\"").lastOrNull()
            ?.filter { it.isDigit() }
            ?.toLongOrNull()
            ?.div(1024) ?: 0
    } else {
        File("/proc/self/status").readLines()
            .firstOrNull { it.startsWith("VmRSS:") }
            ?.let { Regex("(\\d+)").find(it)?.groupValues?.get(1)?.toLongOrNull()?.div(1024) }
            ?: 0
    }
}

/**
 * Pinned to 1 for every conformance run.
 *
 * ONNX Runtime's CPU reductions split across threads, so a different thread count legitimately shifts
 * a result by ~1e-6 — inside the float tolerance, but enough to flip an argmax on near-equal values,
 * which is an exact-match failure with no float anywhere near it. The service passes 0 instead and
 * lets ORT choose, because it has no goldens to match and wants the throughput.
 */
private const val IntraOpThreads: Int = 1

private class Flags(
    val image: String? = null,
    val dumpDir: String? = null,
    val upTo: String? = null,
    val device: String = "cpu",
    val ocr: String = "accurate",
    val imgSize: Int = 1500,
    val docconf: Double = 0.5,
    val includeDebug: Boolean = false,
)

/**
 * Parses the flags the contract defines, and rejects anything else.
 *
 * Unknown flags are an ERROR rather than being ignored: the checker passes them, and a silently
 * ignored `--device gpu` would produce a CPU run reported as a GPU one — which poisons every timing
 * conclusion drawn from it afterwards.
 */
private fun parseFlags(args: Array<String>): Flags {
    var image: String? = null
    var dumpDir: String? = null
    var upTo: String? = null
    var device = "cpu"
    var ocr = "accurate"
    var imgSize = 1500
    var docconf = 0.5
    var includeDebug = false

    var i = 1
    while (i < args.size) {
        val flag = args[i]
        when (flag) {
            "--image" -> image = next(args, ++i, flag)
            "--dump-dir" -> dumpDir = next(args, ++i, flag)
            "--upto" -> upTo = next(args, ++i, flag)
            "--device" -> device = oneOf(next(args, ++i, flag), "cpu", "gpu")
            "--ocr" -> ocr = oneOf(next(args, ++i, flag), "accurate", "fast")
            "--img-size" -> imgSize = int(next(args, ++i, flag), flag)
            "--docconf" -> docconf = double(next(args, ++i, flag), flag)
            "--include-debug" -> includeDebug = true
            else -> throw FlagException("unknown flag $flag")
        }
        i++
    }
    return Flags(image, dumpDir, upTo, device, ocr, imgSize, docconf, includeDebug)
}

private fun next(args: Array<String>, index: Int, flag: String): String =
    if (index < args.size) args[index] else throw FlagException("$flag needs a value")

private fun oneOf(value: String, vararg allowed: String): String =
    if (value in allowed) value
    else throw FlagException("expected one of ${allowed.joinToString("|")}, got $value")

private fun int(value: String, flag: String): Int =
    value.toIntOrNull() ?: throw FlagException("$flag expects an integer, got $value")

private fun double(value: String, flag: String): Double =
    value.toDoubleOrNull() ?: throw FlagException("$flag expects a number, got $value")
