package net.russiandocs.service.worker

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.Future
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import net.russiandocs.docproc.viewmodel.Payload
import net.russiandocs.service.config.Settings
import net.russiandocs.service.errors.ErrorKind
import net.russiandocs.service.errors.ServiceException
import net.russiandocs.service.logging.ServiceLog
import net.russiandocs.service.ml.PipelineRuntime
import net.russiandocs.service.model.Document
import net.russiandocs.service.model.DocumentStatus
import net.russiandocs.service.repositories.Artifacts
import net.russiandocs.service.repositories.Documents
import net.russiandocs.service.repositories.SettingsRepository
import net.russiandocs.service.settings.SettingsSchema
import net.russiandocs.service.store.DocumentStore

/**
 * Live progress for one document.
 *
 * **Two honest steps rather than five interpolated ones.** The pipeline exposes no progress callbacks, so a
 * finer breakdown would be theatre — and at ~0.4 s per document a five-segment animated bar is a lie the
 * user can see through. `recognizing` self-calibrates from real completions instead of using a constant.
 */
@Serializable
public data class Progress(
    @SerialName("step") val step: String,
    @SerialName("label") val label: String,
    @SerialName("pct") val pct: Double = 0.0,
    @SerialName("eta_sec") val etaSec: Double = 0.0,
    @SerialName("queue_position") val queuePosition: Int? = null,
)

/**
 * The background recognition loop.
 *
 * One thread pulls queued documents and runs them through the pipeline. Several choices look arbitrary
 * until something breaks, so the reasoning is attached:
 *
 * - **Event-driven, not fixed-interval polling.** Recognition takes ~0.4 s; a ten-second poll would
 *   dominate the latency a user perceives. Uploads signal the loop, and a two-second timeout is only a
 *   safety net for anything that enqueues without signalling.
 * - **A dedicated thread, not the ambient pool.** Python's default executor sizes itself to min(32, cpu+4),
 *   and twenty threads racing for one pipeline lease is not useful. Here the bound is structural: ONE
 *   drain loop, so the invariant holds by construction rather than by pool sizing.
 * - **A timeout cannot kill work already inside the library.** This is the sharpest edge. In Python
 *   `asyncio.wait_for` cancels the coroutine while the executor thread keeps running `process_img` and
 *   keeps holding its lease; on the JVM `Future.cancel(true)` only sets an interrupt flag, which native
 *   ONNX code never checks. The job is marked failed and the loop moves on; the lease is released when
 *   that task finally finishes. Later jobs then get `PIPELINE_BUSY` — a BOUNDED wait — and requeue rather
 *   than blocking forever. A genuinely hung ONNX call needs a process restart, and the container's restart
 *   policy is the last line of defence.
 * - **Transient versus deterministic failures.** Retrying a corrupt JPEG forever is as wrong as giving up
 *   on a CUDA hiccup, so only transient failures consume a retry.
 *
 * Port of `service/worker.py`.
 */
public class RecognitionWorker(
    private val db: DocumentStore,
    private val runtime: PipelineRuntime,
    private val cfg: Settings,
    private val settings: SettingsRepository,
    private val log: ServiceLog,
) {

    private companion object {
        /**
         * The fallback interval. Normal flow is driven by the wake signal; this only catches anything that
         * enqueues without signalling.
         */
        const val QUEUE_POLL_MS = 2_000L

        val STEP_CONFIGS: Map<String, StepConfig> = mapOf(
            "loading" to StepConfig("Loading models", 0.0, 90.0, 20.0),
            "recognizing" to StepConfig("Recognising document", 5.0, 95.0, 0.6),
        )
    }

    private data class StepConfig(
        val label: String,
        val pctStart: Double,
        val pctEnd: Double,
        val duration: Double,
    )

    /**
     * The "there may be work" signal.
     *
     * A permit, not a queue: many uploads collapse into one wake-up and no producer can ever block on a
     * busy loop. The Go port uses a capacity-1 channel with a non-blocking send for the same reason.
     */
    private val wake = Semaphore(0)

    /** Gates the drain loop until model loading has finished (or failed). */
    private val runtimeReady = CountDownLatch(1)

    private val stopping = AtomicBoolean(false)

    /**
     * Two threads, both long-lived and both blocking for seconds inside native code.
     *
     * A dedicated executor rather than a shared pool: starving a shared pool would stall the HTTP side,
     * and the whole point of this design is that uploads stay responsive while a document is recognised.
     */
    private val threads = Executors.newCachedThreadPool { r ->
        Thread(r, "rdocs-worker").apply { isDaemon = true }
    }

    private val gate = Any()
    private val processing = LinkedHashMap<Int, Pair<String, Long>>()
    private var durationEma = 0.6

    /**
     * Wakes the drain loop. Called by the upload and reprocess endpoints.
     *
     * Non-blocking: if a wake is already pending the signal is dropped, which is correct because the loop
     * rescans the queue from scratch anyway.
     */
    public fun notifyNewWork() {
        if (wake.availablePermits() == 0) {
            wake.release()
        }
    }

    /** The current EMA, used by the status page. */
    public fun averageDurationSec(): Double = synchronized(gate) { durationEma }

    private fun setStep(id: Int, step: String) {
        synchronized(gate) { processing[id] = step to System.nanoTime() }
    }

    private fun clearStep(id: Int) {
        synchronized(gate) { processing.remove(id) }
    }

    private fun recordDuration(seconds: Double) {
        synchronized(gate) {
            durationEma = 0.7 * durationEma + 0.3 * maxOf(seconds, 0.05)
        }
    }

    /**
     * Live progress, or `null` when the document is not being processed.
     *
     * `null` is a real answer, not an error: the endpoint returns 200 with a JSON null for a document that
     * is queued or finished, and a 404 there would make the SPA treat a completed document as missing.
     */
    public fun documentProgress(id: Int): Progress? {
        val step: String
        val startedNanos: Long
        val ema: Double
        synchronized(gate) {
            val state = processing[id] ?: return null
            step = state.first
            startedNanos = state.second
            ema = durationEma
        }

        val config = STEP_CONFIGS[step] ?: STEP_CONFIGS.getValue("recognizing")
        val duration = if (step == "recognizing") ema else config.duration
        val elapsed = (System.nanoTime() - startedNanos) / 1_000_000_000.0
        // Capped below 1: a bar that reaches 100 % and then waits is worse than one that stalls at 95 and
        // then jumps, because the first looks broken.
        val fraction = minOf(elapsed / maxOf(duration, 0.05), 0.95)
        val pct = config.pctStart + fraction * (config.pctEnd - config.pctStart)

        return Progress(
            step = step,
            label = config.label,
            pct = round1(pct),
            etaSec = round1(maxOf(0.0, duration - elapsed)),
        )
    }

    private fun round1(v: Double): Double = Math.round(v * 10) / 10.0

    /**
     * Launches the runtime initialisation and the drain loop.
     *
     * **Runtime init runs on its OWN thread and startup does not wait for it**: 215 MB of sessions plus a
     * warmup document take seconds, and blocking startup would delay `/health` and fight Docker's
     * healthcheck. Uploads are accepted immediately and wait in the queue, which is exactly what the async
     * design is for.
     */
    public fun start() {
        val recovered = Documents.resetStaleProcessing(db)
        if (recovered > 0) {
            log.info("[WORKER] requeued $recovered document(s) left mid-processing")
        }
        threads.submit { initRuntime() }
        threads.submit { drainLoop() }
    }

    /** Stops the drain loop. The recognition in flight, if any, cannot be interrupted — see the type note. */
    public fun stop() {
        stopping.set(true)
        wake.release()
        threads.shutdown()
    }

    private fun initRuntime() {
        try {
            val device = SettingsSchema.typedString(
                "compute_device", settings.settingValue(db, "compute_device"), cfg.computeDevice)
            val mode = SettingsSchema.typedString(
                "ocr_mode", settings.settingValue(db, "ocr_mode"), cfg.ocrMode)

            val info = runtime.init(PipelineRuntime.Options(
                computeDevice = device,
                modelFormat = cfg.modelFormat,
                ocrMode = mode,
                warmupImage = cfg.warmupImage,
                poolSize = cfg.pipelinePoolSize,
                repoRoot = cfg.repoRoot(),
            ))
            if (info.state != PipelineRuntime.STATE_READY) {
                log.error("[WORKER] recognition runtime failed to start: ${info.error}")
            }
        } catch (e: Throwable) {
            log.error("[WORKER] runtime initialisation threw: ${e.message}", e)
        } finally {
            // Released EITHER WAY: with a broken runtime the drain loop still needs to run, so queued
            // documents fail with a clear message instead of sitting in 'queued' forever with no
            // explanation.
            runtimeReady.countDown()
            notifyNewWork()
        }
    }

    private fun drainLoop() {
        log.info("[WORKER] drain loop started")
        runtimeReady.await()

        while (!stopping.get()) {
            val id = Documents.nextQueued(db)
            if (id == null) {
                // Waits for a signal OR the fallback tick.
                wake.tryAcquire(QUEUE_POLL_MS, TimeUnit.MILLISECONDS)
                continue
            }
            try {
                processDocument(id)
            } catch (e: Throwable) {
                // The loop must survive anything: an exception escaping here would leave the queue
                // permanently unattended while the service still accepted uploads.
                log.error("[WORKER] drain loop caught an unexpected error on $id: ${e.message}", e)
            }
        }
        log.info("[WORKER] drain loop stopped")
    }

    /** The result of one recognition attempt. */
    private class Outcome(
        val result: PipelineRuntime.RecogniseResult?,
        val elapsedMs: Int,
        val error: Throwable?,
    )

    /**
     * Waits for abandoned recognition work and releases what it produced.
     *
     * Called when the timeout fires or the service is shutting down. The recognition itself CANNOT be
     * cancelled — native code does not observe an interrupt — so it will finish, and its canvas is an
     * `org.opencv.core.Mat`, memory OUTSIDE the Java heap. A missed `close()` here leaks it until the
     * OpenCV finalizer happens to run, which is not a schedule anybody can reason about, so this is the
     * only place that frees it deterministically.
     *
     * It also releases the pipeline lease as a side effect of the work completing, which is why a
     * subsequent job gets `PIPELINE_BUSY` (a bounded wait) rather than blocking forever.
     */
    private fun reap(id: Int, work: Future<Outcome>) {
        threads.submit {
            val got = try {
                work.get()
            } catch (e: Exception) {
                null
            }
            got?.result?.canvas?.close()
            log.warn("[WORKER] abandoned recognition for document $id finished after " +
                "${got?.elapsedMs ?: -1} ms; its canvas was released " +
                "(${got?.error?.message ?: "no error"})")
        }
    }

    /**
     * Claims one document and recognises it.
     *
     * The claim is a status transition, and re-reading the record first is what makes it safe: between the
     * queue scan and here the document may have been deleted or claimed, so a record that is no longer
     * `queued` is skipped rather than processed twice.
     */
    private fun processDocument(id: Int) {
        var record: Document? = Documents.getById(db, id)
        if (record == null || record.status != DocumentStatus.QUEUED) {
            return
        }
        record = Documents.updateStatus(db, record, DocumentStatus.PROCESSING, null, null)

        val timeoutSec = SettingsSchema.typedInt(
            "job_timeout_sec", settings.settingValue(db, "job_timeout_sec"), cfg.jobTimeoutSec)
        val maxRetries = SettingsSchema.typedInt(
            "max_retries", settings.settingValue(db, "max_retries"), cfg.maxRetries)
        val docconf = SettingsSchema.typedFloat(
            "docconf", settings.settingValue(db, "docconf"), cfg.docconf)
        val imgSize = SettingsSchema.typedInt(
            "img_size", settings.settingValue(db, "img_size"), cfg.imgSize)

        setStep(id, "recognizing")

        // The recognition runs on its own thread so the timeout can be observed. It CANNOT be cancelled —
        // see the type note.
        val work: Future<Outcome> = threads.submit<Outcome> {
            val started = System.nanoTime()
            try {
                Outcome(recognise(id, docconf, imgSize), elapsedMs(started), null)
            } catch (e: Throwable) {
                Outcome(null, elapsedMs(started), e)
            }
        }

        val got: Outcome = try {
            work.get(timeoutSec.toLong(), TimeUnit.SECONDS)
        } catch (e: TimeoutException) {
            // **The abandoned work still produces a canvas, and somebody has to free it.** The recognition
            // cannot be cancelled, so it will finish and deliver a result nothing is reading any more —
            // and that result owns a native Mat. Without this reaper every timed-out document holds a full
            // canvas until a finalizer runs, which shows up only in bulk and looks like a slow leak rather
            // than a timeout.
            reap(id, work)
            clearStep(id)
            handleFailure(id, TimeoutException("Recognition exceeded ${timeoutSec}s"), maxRetries)
            return
        } catch (e: InterruptedException) {
            // Shutdown. Same reasoning as the timeout path: the task is still running and its result must
            // be released rather than left to a finalizer.
            reap(id, work)
            clearStep(id)
            Thread.currentThread().interrupt()
            return
        }
        clearStep(id)

        if (got.error != null) {
            handleFailure(id, got.error, maxRetries)
            return
        }

        val result = got.result!!
        result.canvas.use { canvas ->
            recordDuration(got.elapsedMs / 1000.0)

            val current = Documents.getById(db, id) ?: return  // deleted while we were recognising

            if (canvas != null) {
                try {
                    Artifacts.saveCanvas(db, id, canvas)
                    Artifacts.saveThumbnail(db, id, canvas, 96)
                } catch (e: Exception) {
                    // A missing preview must NOT fail an otherwise good recognition: the fields are the
                    // product, the picture is a convenience.
                    log.error("[WORKER] canvas or thumbnail write failed for $id: ${e.message}")
                }
            }

            val payload: Payload = result.viewModel
            val node = try {
                SearchText.toElement(payload)
            } catch (e: Exception) {
                handleFailure(id, e, maxRetries)
                return
            }

            try {
                Documents.saveResult(db, current, node,
                    SearchText.build(current.filename, payload), got.elapsedMs)
            } catch (e: Exception) {
                log.error("[WORKER] cannot save result for $id: ${e.message}")
                return
            }

            log.info("[WORKER] done: doc=$id ms=${got.elapsedMs} type=${payload.docType} " +
                "fields=${payload.fields.size}")
        }
    }

    private fun elapsedMs(startedNanos: Long): Int =
        ((System.nanoTime() - startedNanos) / 1_000_000).toInt()

    /** Loads the stored original and runs the pipeline. */
    private fun recognise(id: Int, docconf: Double, imgSize: Int): PipelineRuntime.RecogniseResult {
        val artifact = Artifacts.openArtifact(db, id, "original")
        // Deterministic, so never retried: this is the symptom of the upload race that
        // Documents.reserveId exists to prevent, and retrying would only hide it.
            ?: throw ServiceException.unreadable("Document $id has no stored original")
        return runtime.recognise(artifact.first, PipelineRuntime.RecogniseOptions(
            docconf = docconf,
            imgSize = imgSize,
        ))
    }

    /**
     * Classifies the error and either requeues or fails the document.
     *
     * **Only TRANSIENT failures consume a retry.** A corrupt JPEG fails immediately and forever, because
     * the same bytes will fail the same way and a retry loop on them starves the queue.
     */
    private fun handleFailure(id: Int, error: Throwable, maxRetries: Int) {
        val (code, transient) = classify(error)
        log.warn("[WORKER] document $id failed: code=$code transient=$transient ${error.message}")

        val record = Documents.getById(db, id) ?: return
        val message = error.message ?: code
        if (transient && record.retryCount < maxRetries) {
            val nextRetry = record.retryCount + 1
            Documents.update(db, record) {
                it.copy(
                    status = DocumentStatus.QUEUED,
                    retryCount = nextRetry,
                    error = message,
                    errorCode = code,
                    startedAt = null,
                )
            }
            notifyNewWork()
            return
        }
        Documents.updateStatus(db, record, DocumentStatus.FAILED, message, code)
    }

    /**
     * Maps an error to a machine-readable code and a retry decision.
     *
     * The CODE is separate from the message because the UI is English while a message may not be, and
     * because a client should branch on a stable token rather than on prose.
     */
    private fun classify(error: Throwable): Pair<String, Boolean> = when {
        error is ServiceException && error.kind == ErrorKind.PIPELINE_BUSY ->
            "pipeline_busy" to true
        error is ServiceException && error.kind == ErrorKind.RUNTIME_NOT_READY ->
            "runtime_not_ready" to true
        error is ServiceException && error.kind == ErrorKind.IMAGE_UNREADABLE ->
            "image_unreadable" to false
        error is TimeoutException -> "timeout" to true
        // UNKNOWN IS NOT TRANSIENT. The safe direction: an unrecognised error retried forever stops the
        // queue making progress, and nothing in the log says why.
        else -> "error" to false
    }
}
