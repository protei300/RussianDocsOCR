package net.russiandocs.docproc.preprocess

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Interpolation
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.models.ModelInput
import net.russiandocs.docproc.tensors.Dtype
import net.russiandocs.docproc.tensors.NdArray
import net.russiandocs.docproc.tensors.Ops
import kotlin.math.max

/**
 * The OCR input: fixed height, DYNAMIC width, uint8 NHWC in the model's declared colour order.
 *
 * The dynamic width is the thing the whole port was gated on — the Go spike's T4 existed to prove one
 * long-lived session accepts a different width on every call, because there is no fallback if it does not.
 */
public class OcrV2(input: ModelInput) : Preprocessor {

    private val height: Int = input.height?.takeIf { it > 0 } ?: 32
    private val colorOrder: String = input.colorOrder?.takeIf { it.isNotEmpty() } ?: "BGR"

    override fun apply(image: Image): Pair<NdArray, Meta> {
        val h = image.height
        val w = image.width

        // A zero-sized crop returns a BLANK tensor of the minimum shape rather than throwing. The reference
        // does the same, and the degenerate crop is reachable: clampedCrop yields it when a detector box
        // lands entirely outside the patch. Failing here would turn a rare bad box into a failed document.
        if (h == 0 || w == 0) {
            val blank = NdArray.fromUInt8(ByteArray(height * MIN_WIDTH * 3), 1, height, MIN_WIDTH, 3)
            return blank to Meta(origH = h, origW = w, ratio = 1.0)
        }

        // Width scaled by the height ratio, rounded half-to-even, floored at the minimum.
        val newW = max(MIN_WIDTH, Ops.roundHalfEven(w.toDouble() * height / h, 0).toInt())

        // **The model wants BGR and the pipeline works in RGB**, so the conversion is real work, not a no-op
        // like the one in the quality tiles. Driven by the config's ColorOrder rather than hardcoded, because
        // that is where the reference reads it from.
        val source = if (colorOrder == "BGR") Io.toBgr(image) else image.clone()
        source.use {
            Io.resize(source, newW, height, Interpolation.LINEAR).use { resized ->
                val pixels = resized.toArray()
                val batched = NdArray(
                    pixels.data,
                    intArrayOf(1, resized.height, resized.width, resized.channels),
                    Dtype.UINT8,
                    1,
                )
                return batched to Meta(origH = h, origW = w, ratio = 1.0)
            }
        }
    }

    private companion object {
        /**
         * The minimum width the model will accept.
         *
         * 16, and it is load-bearing rather than defensive: a very narrow crop scaled by height alone can
         * round to 1 or 2 pixels, which the graph rejects. The reference clamps here too.
         */
        const val MIN_WIDTH = 16
    }
}
