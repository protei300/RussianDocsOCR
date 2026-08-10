package net.russiandocs.docproc.imaging

import net.russiandocs.docproc.tensors.NdArray
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect

/**
 * An owned image. Wraps an OpenCV [Mat] and must be closed.
 *
 * **The wrapper exists for ownership, not for abstraction.** A raw `Mat` holds native memory that the
 * JVM's garbage collector does not account for and will not reclaim on any useful timescale, and the
 * Go port proved what that costs: one path that read a result and returned without releasing it leaked
 * 12.7 MB per document, unbounded, 663 MB to 6932 MB over 460 documents — with the conformance suite
 * green throughout, because the CLI processes one document per process.
 *
 * On the JVM this matters MORE than in .NET, not less. `org.opencv.core.Mat` implements neither
 * `Closeable` nor `AutoCloseable`, so Kotlin's `use` does not work on it and nothing in the language
 * or the tooling will remind anybody. That is precisely why this type exists and is [AutoCloseable]:
 * it makes `use { }` available, and it makes a missing release visible as a missing `use`.
 *
 * Every function returning an [Image] transfers OWNERSHIP to the caller. Functions taking one only
 * borrow it. Where the reference aliases an array and lets the GC sort it out, this port clones — see
 * the unsplit-field fallback in word splitting, where a borrowed Mat inside a list the caller closes
 * is a double free that only shows up in bulk.
 */
public class Image private constructor(private var backing: Mat?) : AutoCloseable {

    /**
     * The underlying Mat, BORROWED. Callers must not release it, and must not keep it past the
     * lifetime of this [Image].
     */
    public val mat: Mat
        get() = backing ?: throw IllegalStateException("imaging: image has been closed")

    public val width: Int get() = mat.cols()
    public val height: Int get() = mat.rows()
    public val channels: Int get() = mat.channels()
    public val isEmpty: Boolean get() = backing?.empty() ?: true

    /** An independent copy. The caller owns the result. */
    public fun clone(): Image = Image(mat.clone())

    /**
     * Detaches the Mat and hands it over, leaving this instance closed.
     *
     * The analogue of the Go port's `Results.TakeCanvas`, for the same reason: exactly one image has
     * to outlive a pipeline run — the canvas the service stores — while every intermediate must be
     * released immediately. Without a way to say that, the only options are releasing what the caller
     * still needs or releasing nothing, and the Go port shipped the second one.
     */
    public fun take(): Mat {
        val m = mat
        backing = null
        return m
    }

    /**
     * Copies the pixels into an [NdArray] shaped `(H, W, C)`, uint8.
     *
     * Used only by the probe: this is how an image stage becomes a comparable payload.
     *
     * **A copy, and row by row.** `Mat` rows can be padded — `step1()` is not necessarily
     * `cols * channels` — and reading the buffer as if it were contiguous yields an image with a
     * diagonal skew that looks like a warp bug. `Mat.get(byte[])` on a non-continuous Mat throws
     * rather than compacting, so the rows are fetched individually.
     */
    public fun toArray(): NdArray {
        val m = mat
        check(CvType.depth(m.type()) == CvType.CV_8U) {
            "imaging: toArray expects 8-bit, got ${CvType.typeToString(m.type())}"
        }
        val h = m.rows()
        val w = m.cols()
        val c = m.channels()
        val data = ByteArray(h * w * c)
        if (m.isContinuous) {
            m.get(0, 0, data)
        } else {
            val rowBytes = w * c
            val row = ByteArray(rowBytes)
            for (y in 0 until h) {
                m.get(y, 0, row)
                row.copyInto(data, y * rowBytes)
            }
        }
        return NdArray.fromUInt8(data, h, w, c)
    }

    override fun close() {
        backing?.release()
        backing = null
    }

    public companion object {
        /** Takes ownership of an existing Mat. */
        public fun wrap(mat: Mat): Image = Image(mat)
    }
}

/**
 * Cropping. **The only sanctioned crop path in the port.**
 *
 * This is the single most dangerous divergence in the whole exercise, and it is worth spelling out
 * because the failure mode is a crash on a rare input rather than a wrong number on a common one.
 *
 * `img[y1:y2, x1:x2]` in Python does not validate: an upper bound past the edge is silently CLAMPED,
 * and a negative start counts from the end. `Mat.submat(rect)` on the JVM — like `Region` in gocv and
 * `new Mat(mat, rect)` in OpenCvSharp — THROWS. So the detectors, which routinely return a box a pixel
 * or two outside the image, produce a working crop in the reference and an exception in a port that
 * translates the slice literally.
 *
 * A port that "works" is therefore a port that clamps, and it must clamp the way the slice effectively
 * does. Hence one function, used everywhere, with unit tests per language.
 */
public object Crop {

    /**
     * The clamped equivalent of `img[y1:y2, x1:x2]`.
     *
     * Negative starts clamp to 0 rather than counting from the end. The reference's own coordinates
     * come from detector output already clipped to non-negative, so the count-from-the-end branch is
     * unreachable there; implementing it would add behaviour the reference does not exercise, which is
     * a worse kind of wrong than not implementing it.
     *
     * An empty intersection yields a zero-sized image rather than an error, because the reference's
     * slice does too — and the OCR path has a documented degenerate route for exactly that.
     */
    public fun clampedCrop(src: Image, x1: Int, y1: Int, x2: Int, y2: Int): Image {
        val w = src.width
        val h = src.height

        val left = x1.coerceIn(0, w)
        val top = y1.coerceIn(0, h)
        val right = x2.coerceIn(0, w)
        val bottom = y2.coerceIn(0, h)

        // A reversed range is empty in Python — not an error, and not a flipped crop.
        val cropW = maxOf(0, right - left)
        val cropH = maxOf(0, bottom - top)

        if (cropW == 0 || cropH == 0) {
            // Zero-sized, with the source's type so downstream code sees the expected channel count.
            return Image.wrap(Mat(cropH, cropW, src.mat.type()))
        }

        // Clone rather than return a view: a submat shares the parent's buffer, so the crop would
        // dangle the moment the parent is released — and the parent here is a pipeline intermediate
        // released as soon as the stage ends.
        val view = src.mat.submat(Rect(left, top, cropW, cropH))
        return try {
            Image.wrap(view.clone())
        } finally {
            // The view itself is a header over the parent's pixels; releasing it frees the header only,
            // which is exactly what should happen here.
            view.release()
        }
    }
}
