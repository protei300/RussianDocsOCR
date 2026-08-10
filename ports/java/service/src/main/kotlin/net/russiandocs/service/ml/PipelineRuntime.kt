package net.russiandocs.service.ml

import java.io.File
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.inference.DeviceResolution
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.pipeline.OcrTier
import net.russiandocs.docproc.pipeline.Recognizer
import net.russiandocs.docproc.pipeline.RunOptions
import net.russiandocs.docproc.viewmodel.Payload
import net.russiandocs.service.errors.ServiceException

/**
 * Owns and safely calls the recognition pipeline.
 *
 * **This is the reference part of the reference project.** Everything below encodes a rule that is easy to
 * get wrong and expensive to debug, and each was verified against the library rather than inferred from
 * documentation. The Python original states ten; the ones that survive into Kotlin, and the two that do
 * NOT, are laid out here.
 *
 * 1. **A pipeline instance is not re-entrant.** In Python `process_img` rebinds `self.results` and
 *    `self.ocr_options` on every call, so two concurrent calls on one instance silently return each
 *    other's fields — no crash, no reproduction in single-user testing, corrupted data under load. THIS
 *    PORT DOES NOT HAVE THAT BUG: [Recognizer.run] holds its state in locals and returns it. The lease is
 *    kept anyway, for rules 3 and 9, and because removing it would make this port differ structurally
 *    from the reference, from Go and from .NET.
 * 2. **The per-session CUDA lock does not help with (1).** It serialises individual ONNX `run()` calls on
 *    GPU; it fixes device wedging, not re-entrancy. Different problem, different scope.
 * 3. **Transform the result before releasing the lease.** In Python `results` IS `pipeline.results`, and
 *    the next call replaces it. [recognise] does the whole read-and-convert inside the lease for that
 *    reason — and the signature makes it structural rather than a rule to remember, because [use] is the
 *    only way in.
 * 4. **The library's own warmup cannot report failure** (it swallows exceptions into a print), which is
 *    why warmup here calls the ordinary path and lets the error surface.
 * 5. **Warmup needs a REAL document.** A synthetic grey frame classifies as 'NONE' and short-circuits
 *    before the border, field and OCR stages, warming perhaps a fifth of the graph. It must be an
 *    ANONYMISED repository sample — warmup re-reads the file at every start, so pointing it at a real
 *    document is a data-handling error, not just a taste one.
 * 6. **The library prints to stdout** in Python, which would corrupt a JSON log stream. Not applicable
 *    here: this port logs through the injected sink and prints nothing.
 * 7. **A listed CUDA provider does not mean a working GPU**, and in a container without `--gpus` the
 *    provider SEGFAULTS instead of erroring — a JVM `try`/`catch` cannot catch that, because the process
 *    is gone. Hence the device-node probe gating the attempt, [DeviceResolution.gpuVisible].
 * 8. **GPU does not mean GPU OCR.** The detectors run on CUDA while the OCR engines stay on CPU, because
 *    per-word dynamic widths are far slower on CUDA — measured at 13.7x end-to-end in the Go port.
 *    [RuntimeInfo] reports `device` and `ocr_device` SEPARATELY so the status page can say so instead of
 *    claiming "GPU active".
 * 9. **Models load eagerly and cost 215 MB.** Twelve sessions per instance; a second instance on one card
 *    is also a second CUDA context. Hence a pool of size 1.
 * 10. **Only this package touches the library from the service side.** That keeps the rest of the service
 *     testable without 215 MB of models and bounds the work of porting the service again.
 */
public class PipelineRuntime(private val log: (String) -> Unit) : AutoCloseable {

    public companion object {
        /**
         * How long a caller waits for a free pipeline before giving up.
         *
         * SHORT ON PURPOSE: a queued job that cannot get a pipeline should go back on the queue and
         * surface as "degraded", not block a worker indefinitely.
         */
        public const val LEASE_TIMEOUT_MS: Long = 5_000

        public const val STATE_INITIALIZING: String = "initializing"
        public const val STATE_READY: String = "ready"
        public const val STATE_ERROR: String = "error"

        private fun ocrTierOf(mode: String): OcrTier =
            if (mode == "fast") OcrTier.FAST else OcrTier.ACCURATE
    }

    private val gate = Any()
    private var info = RuntimeInfo(state = STATE_INITIALIZING, providers = emptyList())

    /**
     * The pool: a counting semaphore plus a queue of instances.
     *
     * A semaphore rather than a plain lock because the lease needs a TIMEOUT, which `synchronized` cannot
     * express, and because "wait for one of N" is exactly what a semaphore is. The Go port uses a buffered
     * channel for the same reason; .NET uses `SemaphoreSlim` and a bag.
     */
    private var pool: ArrayBlockingQueue<Instance>? = null
    private var available: Semaphore? = null
    private var poolSize = 0

    /**
     * One recognition pipeline.
     *
     * A type rather than a bare set of modules so the pool has something to hold, and so closing has one
     * place to release twelve sessions.
     */
    public class Instance internal constructor(
        public val device: Device,
        public val modelFormat: String,
        public val ocrMode: String,
        internal val pipeline: Recognizer,
    ) : AutoCloseable {
        /**
         * Separate from [device] — rule 8. A separate value rather than a derived one so the status page
         * can report the two independently, which is the whole point: an operator looking at nvidia-smi
         * and seeing idle OCR needs the service to have said so first.
         */
        public val ocrDevice: Device = Device.CPU

        override fun close(): Unit = pipeline.close()
    }

    /** Configuration for [init]. */
    public data class Options(
        /** auto | cpu | gpu */
        val computeDevice: String = "auto",
        val modelFormat: String = "ONNX",
        val ocrMode: String = "accurate",
        val warmupImage: String = "",
        val poolSize: Int = 1,
        /** Locates `samples/` for the warmup fallback. */
        val repoRoot: String? = null,
    )

    /** A snapshot, with the live pool counts filled in. */
    public fun info(): RuntimeInfo {
        val snapshot = synchronized(gate) { info }
        return snapshot.copy(
            poolSize = poolSize,
            poolAvailable = available?.availablePermits() ?: 0,
            // Copied so a caller cannot mutate the published list.
            providers = snapshot.providers.toList(),
        )
    }

    private fun set(mutate: (RuntimeInfo) -> RuntimeInfo) {
        synchronized(gate) { info = mutate(info) }
    }

    public val isReady: Boolean get() = info().state == STATE_READY

    /**
     * Builds the pipelines, warms them, and publishes what actually happened.
     *
     * Blocking and slow. **NEVER THROWS for a recognition failure**: a failure is recorded as
     * `state == "error"` so the service can still serve its status page and explain itself rather than
     * refusing to start. A service that will not boot cannot tell you why it will not boot.
     */
    public fun init(opts: Options): RuntimeInfo {
        // Providers are OBSERVED, not advertised. CPU is always real; CUDA is added below only if a
        // session actually builds on it. The reference reports what the library advertises, which is the
        // very list rule 7 says cannot be trusted.
        val providers = listOf("CPUExecutionProvider")
        set {
            it.copy(
                providers = providers,
                modelFormat = opts.modelFormat,
                ocrMode = opts.ocrMode,
                requestedDevice = opts.computeDevice,
            )
        }

        // TWO INDEPENDENT CONDITIONS, and both are required. The provider list says the GPU build is
        // present; gpuVisible says a device was actually passed through. With the first true and the
        // second false, building a CUDA session TERMINATES the process instead of returning an error.
        val hasDevice = DeviceResolution.gpuVisible()

        var wanted = opts.computeDevice
        if (opts.computeDevice == "auto") {
            wanted = if (hasDevice) "gpu" else "cpu"
        } else if (opts.computeDevice == "gpu" && !hasDevice) {
            log("[RUNTIME] compute_device=gpu but no GPU is visible to this process — refusing to " +
                "attempt CUDA, which would TERMINATE the process rather than fail cleanly. Using " +
                "CPU. In Docker, pass --gpus all.")
            wanted = "cpu"
        }

        val attempts = if (wanted == "gpu") listOf(Device.GPU, Device.CPU) else listOf(Device.CPU)

        val sample = findWarmupSample(opts.warmupImage, opts.repoRoot)
        if (sample == null) {
            log("[RUNTIME] no warmup sample found; the first real document will pay the cold-start " +
                "cost")
        }

        val size = maxOf(1, opts.poolSize)
        var lastError: Throwable? = null

        for ((idx, attempt) in attempts.withIndex()) {
            log("[RUNTIME] building pipeline on ${attempt.wire} " +
                "(${opts.modelFormat}, ocr=${opts.ocrMode})")
            val started = System.nanoTime()

            val built = ArrayList<Instance>(size)
            var buildError: Throwable? = null
            for (n in 0 until size) {
                try {
                    // intraOpThreads: 0 leaves the thread count to ONNX Runtime — the opposite of the
                    // conformance CLI, which pins it to 1. Pinning exists so a thread-count change cannot
                    // shift a reduction by ~1e-6 and flip an argmax; the service has no goldens to match
                    // and wants the throughput.
                    built.add(Instance(
                        attempt, opts.modelFormat, opts.ocrMode,
                        Recognizer(attempt, 0, ocrTierOf(opts.ocrMode)),
                    ))
                } catch (e: Throwable) {
                    buildError = e
                    break
                }
            }

            if (buildError != null) {
                // Partial builds are released before falling back: leaving them would hold a CUDA context
                // that the CPU attempt then competes with.
                built.forEach { runCatching { it.close() } }
                lastError = buildError
                val next = if (idx + 1 < attempts.size) {
                    "falling back to ${attempts[idx + 1].wire}"
                } else {
                    "no fallback left"
                }
                log("[RUNTIME] pipeline init FAILED on ${attempt.wire}: ${buildError.message} — $next")
                continue
            }

            val loadMs = ((System.nanoTime() - started) / 1_000_000).toInt()
            log("[RUNTIME] ${built.size} instance(s) constructed on ${attempt.wire} in $loadMs ms")

            var warmupMs: Int? = null
            if (sample != null) {
                var total = 0
                var warmed = 0
                for (instance in built) {
                    try {
                        total += warm(instance, sample)
                        warmed++
                    } catch (e: Exception) {
                        // A failed warmup is LOGGED, not fatal: the pipeline is built and works; the
                        // first document just pays the cold cost. The reference could not even report
                        // this — its warmup swallows the exception into a print.
                        log("[RUNTIME] warmup failed: ${e.message}")
                    }
                }
                if (warmed > 0) {
                    warmupMs = total / warmed
                }
            }

            val queue = ArrayBlockingQueue<Instance>(built.size)
            queue.addAll(built)
            pool = queue
            available = Semaphore(built.size)
            poolSize = built.size

            val first = built[0]
            val fellBack = wanted == "gpu" && attempt == Device.CPU
            // Recorded only now: the session built, so CUDA is not merely installed but working.
            val observed = if (attempt == Device.GPU) {
                listOf("CUDAExecutionProvider") + providers
            } else {
                providers
            }

            set {
                it.copy(
                    state = STATE_READY,
                    providers = observed,
                    device = first.device.wire,
                    ocrDevice = first.ocrDevice.wire,
                    fellBack = fellBack,
                    loadMs = loadMs,
                    warmupMs = warmupMs,
                    error = null,
                )
            }
            log("[RUNTIME] ready: device=${first.device.wire} ocr_device=${first.ocrDevice.wire} " +
                "load_ms=$loadMs instances=${built.size}")
            if (fellBack) {
                log("[RUNTIME] GPU was requested but only CPU worked — check CUDA/cuDNN. " +
                    "Recognition will be slower.")
            }
            return info()
        }

        log("[RUNTIME] recognition unavailable; the service will start and accept uploads, but " +
            "every document will fail: ${lastError?.message}")
        val message = lastError?.message ?: "unknown error"
        set { it.copy(state = STATE_ERROR, error = message) }
        return info()
    }

    /** Drops the pipelines. */
    override fun close() {
        var drained = 0
        val queue = pool
        if (queue != null) {
            while (true) {
                val instance = queue.poll() ?: break
                runCatching { instance.close() }
                drained++
            }
        }
        pool = null
        available = null
        poolSize = 0
        set { it.copy(state = STATE_INITIALIZING, device = null, ocrDevice = null) }
        log("[RUNTIME] released $drained pipeline instance(s)")
    }

    /**
     * Runs [body] with exclusive access to one instance.
     *
     * **The only way to reach a pipeline.** A higher-order function rather than acquire/release,
     * deliberately: it makes rule 3 — transform the result BEFORE releasing — structurally enforced
     * instead of a comment somebody deletes. Python expresses the same thing as a context manager, Go as a
     * callback, .NET as `Use<T>`.
     *
     * Throws `RUNTIME_NOT_READY` before the models finish loading and `PIPELINE_BUSY` if none becomes free
     * within the timeout. Both are TRANSIENT, so the caller requeues rather than failing the job.
     */
    public fun <T> use(timeoutMs: Long, body: (Instance) -> T): T {
        val snapshot = info()
        when (snapshot.state) {
            STATE_ERROR -> throw ServiceException.notReady(
                "Recognition runtime failed to start: ${snapshot.error}")
            STATE_READY -> Unit
            else -> throw ServiceException.notReady("Recognition runtime is still loading models")
        }

        val permits = available
        val queue = pool
        if (permits == null || queue == null ||
            !permits.tryAcquire(timeoutMs, TimeUnit.MILLISECONDS)
        ) {
            throw ServiceException.busy(
                "No pipeline became available within ${timeoutMs / 1000.0}s")
        }

        var instance: Instance? = null
        try {
            instance = queue.poll()
                // Cannot happen while the semaphore and the queue agree, and worth saying rather than
                // dereferencing null: the two are only ever changed together in init and close.
                ?: throw ServiceException.busy("Pipeline pool is inconsistent")
            return body(instance)
        } finally {
            // Returned in a finally so an exception in the body cannot leak the instance and wedge the
            // pool at zero available — which would look exactly like the hang the lease timeout exists to
            // report.
            if (instance != null) {
                queue.offer(instance)
            }
            permits.release()
        }
    }

    /** The per-document knobs. */
    public data class RecogniseOptions(
        val includeDebug: Boolean = false,
        val docconf: Double = 0.5,
        val imgSize: Int = 1500,
        val leaseTimeoutMs: Long = 0,
    )

    /**
     * What [recognise] produces.
     *
     * [canvas] is RGB and OWNED BY THE CALLER, who must close it. `null` when the document
     * short-circuited as unrecognised.
     */
    public data class RecogniseResult(
        val viewModel: Payload,
        val canvas: Image?,
    )

    /**
     * Processes one document. The whole public surface of this type.
     *
     * The canvas is RGB and the encoder writes BGR, so the artifact layer converts before writing — see
     * `Artifacts.saveCanvas`. Getting that wrong swaps red and blue in every stored document, and on a
     * passport it looks plausible enough to ship unnoticed.
     */
    public fun recognise(imagePath: String, opts: RecogniseOptions): RecogniseResult {
        val timeout = if (opts.leaseTimeoutMs > 0) opts.leaseTimeoutMs else LEASE_TIMEOUT_MS

        return use(timeout) { instance ->
            instance.pipeline.run(
                imagePath,
                RunOptions(
                    docconf = opts.docconf,
                    imgSize = opts.imgSize,
                    includeDebug = opts.includeDebug,
                ),
            ).use { results ->
                // Built INSIDE the lease — rule 3. Structural here rather than remembered, because there
                // is no way to hold `results` past the lambda.
                val payload = instance.pipeline.buildViewModel(results, opts.includeDebug)

                // **takeCanvas, not a bare field read.** The canvas must outlive the run, but every other
                // image the run allocated must not. In the Go port, reading the field and returning left
                // the intermediates — the fully decoded original among them — alive forever: 663 MB ->
                // 4018 MB across 230 documents, growing without bound. Nothing in the conformance suite
                // could catch it, because the CLI closes its Results after a single document.
                val canvas = results.takeCanvas()

                // The canvas ESCAPES the lease deliberately and is now owned by the caller: it is a
                // standalone Mat, not a view into pipeline state, so unlike the reference's `results` it
                // is safe to read after release.
                RecogniseResult(viewModel = payload, canvas = canvas)
            }
        }
    }

    /** Pays the cold-start cost once, up front. */
    private fun warm(instance: Instance, sample: String): Int {
        val started = System.nanoTime()
        instance.pipeline.run(sample, RunOptions(docconf = 0.5, imgSize = 1500)).use { }
        return ((System.nanoTime() - started) / 1_000_000).toInt()
    }

    /**
     * Resolves the configured image, else picks one from `samples/`.
     *
     * **Only anonymised repository samples are eligible.** Warmup re-reads this file at every start, so a
     * real document here would be read on every boot of every deployment — which is why the fallback
     * searches `samples/` and never the data directory.
     */
    private fun findWarmupSample(configured: String, repoRoot: String?): String? {
        if (configured.isNotEmpty()) {
            if (File(configured).isFile) {
                return configured
            }
            log("[RUNTIME] configured warmup image does not exist: $configured")
        }
        if (repoRoot == null) {
            return null
        }
        // A fixed preference order rather than "the first file found": the chosen sample decides which
        // parts of the graph get warmed, and a directory listing order is not a decision anybody made.
        for (relative in listOf(
            "samples/INTPASSPORT_2011/12_CR_INTPASSPORT_2011.jpg",
            "samples/DL_2011/1_CR_DL_2010.jpg",
        )) {
            val candidate = File(repoRoot, relative)
            if (candidate.isFile) {
                return candidate.path
            }
        }
        return null
    }
}

/**
 * What the recognition runtime actually ended up doing.
 *
 * Reported verbatim by `GET /status`. An operator needs the REAL answer, not the configured intent,
 * because the two differ whenever a GPU was asked for and not obtained.
 */
@Serializable
public data class RuntimeInfo(
    @SerialName("state") val state: String,
    @SerialName("providers") val providers: List<String>,
    /** What the detectors use. `ocr_device` differs from it BY DESIGN — rule 8. */
    @SerialName("device") val device: String? = null,
    @SerialName("ocr_device") val ocrDevice: String? = null,
    @SerialName("model_format") val modelFormat: String? = null,
    @SerialName("ocr_mode") val ocrMode: String? = null,
    @SerialName("requested_device") val requestedDevice: String? = null,
    /**
     * Records that a GPU was requested and CPU was used. The single most useful field on the status page,
     * because it is the difference between "slow" and "broken".
     */
    @SerialName("fell_back") val fellBack: Boolean = false,
    @SerialName("load_ms") val loadMs: Int? = null,
    @SerialName("warmup_ms") val warmupMs: Int? = null,
    @SerialName("library_version") val libraryVersion: String? = null,
    @SerialName("error") val error: String? = null,
    @SerialName("pool_size") val poolSize: Int = 0,
    @SerialName("pool_available") val poolAvailable: Int = 0,
)
