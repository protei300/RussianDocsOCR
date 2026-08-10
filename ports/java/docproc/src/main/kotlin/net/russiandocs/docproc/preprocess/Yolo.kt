package net.russiandocs.docproc.preprocess

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Interpolation
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.models.ModelInput
import net.russiandocs.docproc.tensors.Dtype
import net.russiandocs.docproc.tensors.NdArray
import net.russiandocs.docproc.tensors.PyNum
import kotlin.math.min

/** The detector input: pad, letterbox to the declared square, hand over as uint8 NHWC. */
public class Yolo(input: ModelInput) : Preprocessor {

    private val height: Int
    private val width: Int
    private val paddingSize: List<Int>? = input.paddingSize
    private val paddingColour: List<Int>? = input.paddingColor

    init {
        val shape = input.shape
            ?: throw IllegalArgumentException("preprocess: YOLO input has no Shape")
        require(shape.size >= 2) {
            "preprocess: YOLO Shape needs at least 2 entries, got ${shape.size}"
        }
        height = shape[0]
        width = shape[1]
    }

    override fun apply(image: Image): Pair<NdArray, Meta> {
        val (padded, extra) = Padding.pad(image, paddingSize, paddingColour)
        padded.use {
            val paddedH = padded.height
            val paddedW = padded.width
            val (boxed, ratio, padLetter) = letterbox(padded, height, width)
            boxed.use {
                val pixels = boxed.toArray()
                val batched = NdArray(
                    pixels.data,
                    intArrayOf(1, boxed.height, boxed.width, boxed.channels),
                    Dtype.UINT8,
                    1,
                )
                return batched to Meta(
                    ratio = ratio,
                    padExtra = extra,
                    padLetter = padLetter,
                    paddedH = paddedH,
                    paddedW = paddedW,
                    origH = image.height,
                    origW = image.width,
                )
            }
        }
    }

    /**
     * Scales to fit and pads the remainder — the standard YOLO letterbox.
     *
     * **The asymmetric ±0.1 is not decoration.** With an odd amount of padding to distribute,
     * `round(dh - 0.1)` and `round(dh + 0.1)` put the extra row at the BOTTOM and the extra column at the
     * RIGHT. A "clean" implementation that halves the padding evenly shifts every returned box by a pixel,
     * and the failure surfaces as a coordinate mismatch with no obvious source.
     *
     * Every rounding here is half-to-even, matching `np.round`. On the JVM that means `Math.rint` — NOT
     * `Math.round` or `roundToInt`, both of which are half-up.
     *
     * The resize is SKIPPED when the size already matches, rather than run with a ratio of 1.0 — running
     * the interpolator needlessly can perturb pixels.
     */
    private fun letterbox(src: Image, targetH: Int, targetW: Int): Triple<Image, Double, DoubleArray> {
        val h = src.height
        val w = src.width

        val ratio = min(targetH.toDouble() / h, targetW.toDouble() / w)
        val newW = PyNum.roundHalfEvenToInt(w * ratio)
        val newH = PyNum.roundHalfEvenToInt(h * ratio)

        val dw = (targetW - newW) / 2.0
        val dh = (targetH - newH) / 2.0

        val scaled = if (w != newW || h != newH) {
            Io.resize(src, newW, newH, Interpolation.LINEAR)
        } else {
            src.clone()
        }

        scaled.use {
            val top = PyNum.roundHalfEvenToInt(dh - 0.1)
            val bottom = PyNum.roundHalfEvenToInt(dh + 0.1)
            val left = PyNum.roundHalfEvenToInt(dw - 0.1)
            val right = PyNum.roundHalfEvenToInt(dw + 0.1)

            val boxed = Io.copyMakeBorderConstant(scaled, top, bottom, left, right,
                FILL, FILL, FILL)
            return Triple(boxed, ratio, doubleArrayOf(dw, dh))
        }
    }

    private companion object {
        /**
         * The letterbox fill. Grey rather than black because that is what the reference uses, and the value
         * reaches the model — a different fill shifts every score slightly.
         */
        const val FILL = 114.0
    }
}
