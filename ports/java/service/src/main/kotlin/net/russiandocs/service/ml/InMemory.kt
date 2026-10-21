package net.russiandocs.service.ml

import java.io.OutputStream
import java.io.PrintStream
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.pipeline.Results
import net.russiandocs.docproc.pipeline.RunOptions
import net.russiandocs.docproc.viewmodel.Payload

/**
 * What an in-memory recognition hands back: the view model and, when asked, the canvas as PNG bytes.
 *
 * [png] is a buffer the CALLER zeroes after the response body is built (ports/MEMORY.md §5).
 */
public class InMemoryResult(public val payload: Payload, public val png: ByteArray?)

/**
 * The part of a pipeline an in-memory recognition uses. An interface so the scrub in [InMemory.recogniseOn]
 * can be tested with a fake whose result holds images, without 215 MB of models — the reference's
 * `test_recognise_image_wipes_the_pipeline_result_inside_the_lease` does the same with a fake pipeline.
 */
public interface DocumentPipeline {
    /** Reads one decoded RGB frame, which stays the caller's. */
    public fun read(frame: Image, options: RunOptions): Results

    public fun viewModel(results: Results): Payload
}

/**
 * In-memory recognition: rule 11 of the reference runtime.
 *
 * **A pipeline result keeps the document until somebody clears it.** Everything a run produced — the
 * prepared frame, the turned frame, the pages, the canvas — is owned by its [Results]. The stored path hands
 * the canvas over and releases the rest; for an in-memory recognition "released" must mean "cleared", so
 * [recogniseOn] runs the pipeline with [RunOptions.wipeImages] (the library then zeroes those images even when
 * the run fails half-way and closes them itself) and, in `finally`, calls [Results.wipe] — zero every image,
 * then release — **before the lease is released**. Only the result: the other native memory a pipeline holds is
 * model weights and the blank-page templates, loaded before the first document and needed by the next one.
 *
 * Unlike the reference, this port keeps no per-document state on the pipeline object between calls
 * (`Recognizer.run` holds it in locals and returns it), so there is no `_mrz_zone` to clear: the [Results] is
 * the whole of it.
 */
public object InMemory {

    /**
     * Recognises [rgb] on [pipeline]; called INSIDE a lease. Returns the view model and the canvas PNG when
     * [withCanvas]. The result's images are zeroed and released in `finally`, on success and on failure alike.
     */
    public fun recogniseOn(
        pipeline: DocumentPipeline,
        rgb: Image,
        docconf: Double,
        imgSize: Int,
        withCanvas: Boolean,
    ): InMemoryResult {
        var results: Results? = null
        try {
            results = DiscardedOutput.during {
                pipeline.read(rgb, RunOptions(docconf = docconf, imgSize = imgSize, wipeImages = true))
            }
            val payload = pipeline.viewModel(results)
            val png = if (withCanvas) results.canvas?.let { Io.encodePngWiping(it) } else null
            return InMemoryResult(payload, png)
        } finally {
            results?.wipe()
            @Suppress("UNUSED_VALUE")
            results = null
        }
    }
}

/**
 * Discards what the library prints while an in-memory recognition runs, on that thread only.
 *
 * The reference redirects stdout during `process_img` because the Python library prints field values. This
 * port's library prints almost nothing — but it does write a line to stderr for a frame of unknown type, and
 * "nothing from the library reaches the log" is the rule, not "nothing harmful so far". The service's own log
 * does not go through `System.out`/`System.err` (LogRing writes to the file descriptor directly), so gating
 * the two streams per thread touches only stray library output.
 */
internal object DiscardedOutput {
    private val discarding = ThreadLocal.withInitial { false }

    @Volatile
    private var installed = false

    private fun install() {
        if (installed) {
            return
        }
        synchronized(this) {
            if (!installed) {
                System.setOut(gate(System.out))
                System.setErr(gate(System.err))
                installed = true
            }
        }
    }

    private fun gate(target: PrintStream): PrintStream = PrintStream(object : OutputStream() {
        override fun write(b: Int) {
            if (!discarding.get()) target.write(b)
        }

        override fun write(b: ByteArray, off: Int, len: Int) {
            if (!discarding.get()) target.write(b, off, len)
        }

        override fun flush() {
            target.flush()
        }
    }, true)

    fun <T> during(body: () -> T): T {
        install()
        val previous = discarding.get()
        discarding.set(true)
        try {
            return body()
        } finally {
            discarding.set(previous)
        }
    }
}

/** Decoded RGB pixels in, [InMemoryResult] out; the seam `POST /recognize` calls (see `ApiServer.memoryRecogniser`). */
public fun interface MemoryRecogniser {
    public fun recognise(
        rgb: Image,
        docconf: Double,
        imgSize: Int,
        withCanvas: Boolean,
        leaseTimeoutMs: Long,
    ): InMemoryResult
}
