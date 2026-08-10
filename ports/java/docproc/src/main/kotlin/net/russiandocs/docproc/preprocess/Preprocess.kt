package net.russiandocs.docproc.preprocess

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Interpolation
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.models.ModelInput
import net.russiandocs.docproc.tensors.Dtype
import net.russiandocs.docproc.tensors.NdArray

/**
 * What a preprocessor tells the postprocessor so it can undo the scaling.
 *
 * Travels through the pipeline instead of being recomputed, because recomputing it means duplicating the
 * letterbox arithmetic in two places and having them drift.
 */
public data class Meta(
    val ratio: Double = 1.0,
    val padExtra: IntArray = intArrayOf(0, 0),
    val padLetter: DoubleArray = doubleArrayOf(0.0, 0.0),
    val paddedH: Int = 0,
    val paddedW: Int = 0,
    val origH: Int = 0,
    val origW: Int = 0,
) {
    // Generated equals/hashCode on a data class with array members compare by REFERENCE, which is
    // useless and quietly so. Written out because a test comparing two Metas would otherwise pass or
    // fail for the wrong reason.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Meta) return false
        return ratio == other.ratio &&
            padExtra.contentEquals(other.padExtra) &&
            padLetter.contentEquals(other.padLetter) &&
            paddedH == other.paddedH && paddedW == other.paddedW &&
            origH == other.origH && origW == other.origW
    }

    override fun hashCode(): Int {
        var result = ratio.hashCode()
        result = 31 * result + padExtra.contentHashCode()
        result = 31 * result + padLetter.contentHashCode()
        result = 31 * result + paddedH
        result = 31 * result + paddedW
        result = 31 * result + origH
        result = 31 * result + origW
        return result
    }
}

/** One preprocessing step. An interface, never an abstract base class — see CONVENTIONS §5. */
public interface Preprocessor {
    public fun apply(image: Image): Pair<NdArray, Meta>
}

/**
 * A recognised-but-unimplemented input type.
 *
 * Wired rather than omitted (D-06). An omitted case reads as an oversight and gets "helpfully" filled in
 * differently by each port; a case that exists and refuses cannot.
 */
public class NotImplementedPreprocessor(private val tag: String) : Preprocessor {
    override fun apply(image: Image): Pair<NdArray, Meta> =
        throw NotImplementedError("preprocess: input type \"$tag\" is not implemented")
}

public object Padding {
    /**
     * The symmetric constant border from a config's `PaddingSize`.
     *
     * Every shipped `model.json` declares `[0, 0]`, so this is a no-op in practice — ported because it is
     * part of the contract and because the reference returns the applied offsets for the postprocessor to
     * undo.
     *
     * Note the halving: Python pads `pad_v // 2` top AND bottom, so a `PaddingSize` of `[4, 6]` adds 3
     * rows above and below, not 6 in total.
     */
    public fun pad(image: Image, size: List<Int>?, colour: List<Int>?): Pair<Image, IntArray> {
        if (size == null || size.size < 2 || (size[0] == 0 && size[1] == 0)) {
            // Clone, so the caller owns the result either way and the disposal rule has no special case.
            // One copy of a no-op is cheaper than an ownership question at every call site.
            return image.clone() to intArrayOf(0, 0)
        }

        val padH = size[0] / 2
        val padV = size[1] / 2
        val r = colour?.getOrNull(0)?.toDouble() ?: 0.0
        val g = colour?.getOrNull(1)?.toDouble() ?: 0.0
        val b = colour?.getOrNull(2)?.toDouble() ?: 0.0
        return Io.copyMakeBorderConstant(image, padV, padV, padH, padH, r, g, b) to
            intArrayOf(padH, padV)
    }
}

/**
 * The classification input: pad, resize to the declared size, hand over as uint8 NHWC.
 *
 * **No normalisation happens here.** The shipped graphs bake it in, and the reference's own
 * `BasePreprocessing.normalization` method is shadowed by an attribute of the same name and can never be
 * called — so the reference does not normalise either. Adding it would double-normalise every input,
 * which looks like a model problem rather than a preprocessing one.
 */
public class Classification(input: ModelInput) : Preprocessor {

    private val height: Int
    private val width: Int
    private val paddingSize: List<Int>? = input.paddingSize
    private val paddingColour: List<Int>? = input.paddingColor

    init {
        val shape = input.shape
            ?: throw IllegalArgumentException("preprocess: classification input has no Shape")
        require(shape.size >= 2) {
            "preprocess: classification Shape needs at least 2 entries, got ${shape.size}"
        }
        // **Shape is (h, w) and cv2.resize takes (w, h).** The shipped sizes are square, which hides a
        // swap — hence the deliberately non-square resize in the conformance suite.
        height = shape[0]
        width = shape[1]
    }

    override fun apply(image: Image): Pair<NdArray, Meta> {
        val (padded, extra) = Padding.pad(image, paddingSize, paddingColour)
        padded.use {
            Io.resize(padded, width, height, Interpolation.LINEAR).use { resized ->
                val pixels = resized.toArray()
                // Add the batch dimension: (H, W, C) becomes (1, H, W, C).
                val batched = NdArray(
                    pixels.data,
                    intArrayOf(1, resized.height, resized.width, resized.channels),
                    Dtype.UINT8,
                    1,
                )
                return batched to Meta(
                    padExtra = extra,
                    origH = image.height,
                    origW = image.width,
                )
            }
        }
    }
}
