package net.russiandocs.docproc.postprocess

import net.russiandocs.docproc.imaging.Contours
import net.russiandocs.docproc.imaging.FloatMask
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.tensors.NdArray
import net.russiandocs.docproc.tensors.Ops
import kotlin.math.min

public data class SegmentResult(val segments: List<List<Pt>>) : ModelResult

/**
 * Turns proto masks plus per-box coefficients into contours.
 *
 * Not reachable through [Postprocessor.apply]: it needs the detector's boxes as well as the proto tensor,
 * so the segmentation model calls [segment] directly. `apply` throws rather than returning something
 * plausible.
 */
public class YoloSegmentor(private val maskFilter: Double) : Postprocessor {

    override fun apply(output: NdArray, context: Context): ModelResult =
        throw IllegalStateException("postprocess: YOLOSegmentor needs segment(), not apply()")

    /** One contour per box. */
    public fun segment(
        proto: NdArray,
        boxes: List<Box>,
        extraPad: IntArray,
        origH: Int,
        origW: Int,
    ): List<List<Pt>> {
        if (boxes.isEmpty()) {
            return emptyList()
        }

        var shape = proto.shape
        if (shape.size == 4 && shape[0] == 1) {
            shape = shape.copyOfRange(1, shape.size)
        }
        require(shape.size == 3) {
            "postprocess: proto masks expect [h,w,chn], got ${NdArray.describe(proto.shape)}"
        }

        val imh = shape[0]
        val imw = shape[1]
        val chn = shape[2]
        val protoData = proto.asFloat32()

        val segments = ArrayList<List<Pt>>(boxes.size)
        for (box in boxes) {
            require(box.seg.size == chn) {
                "postprocess: ${box.seg.size} mask coefficients for $chn proto channels"
            }

            // masks @ proto.transpose(-1,0,1).reshape(chn,-1), then sigmoid — written as a direct loop over
            // pixels. The transpose exists in the reference only to make numpy's matmul line up, and
            // materialising it here would be pure copying.
            //
            // **Accumulated in float32**, matching the reference's dtype. Widening to double "for accuracy"
            // moves the mask boundary and therefore the contour.
            val mask = FloatArray(imh * imw)
            val coefficients = FloatArray(chn) { box.seg[it].toFloat() }
            for (y in 0 until imh) {
                for (x in 0 until imw) {
                    val at = (y * imw + x) * chn
                    var acc = 0.0f
                    for (c in 0 until chn) {
                        acc += coefficients[c] * protoData[at + c]
                    }
                    mask[y * imw + x] = Ops.sigmoid(acc)
                }
            }

            // Undo the letterbox INSIDE the proto resolution, before upscaling. gain is old/new, and the
            // padding is halved because it was split across both sides.
            val gain = min(imh.toDouble() / origH, imw.toDouble() / origW)
            val padX = (imw - origW * gain) / 2
            val padY = (imh - origH * gain) / 2
            val top = padY.toInt()
            val left = padX.toInt()

            // **TRUNCATE THE DIFFERENCE, do not subtract the truncated padding.** The reference writes
            // `int(imh - pad[1])`, and `imh - int(pad[1])` is a different number whenever the padding is
            // fractional: for imh=160 and pad=20.5 the two give 139 and 140. One row at proto resolution
            // becomes about nine rows after the upscale — which is exactly how the Go port caught it, with
            // borders.canvas reporting 868 rows against the golden's 877 and the width matching exactly.
            val bottom = (imh - padY).toInt()
            val right = (imw - padX).toInt()
            require(top >= 0 && left >= 0 && bottom <= imh && right <= imw &&
                bottom > top && right > left) {
                "postprocess: degenerate mask crop [$top:$bottom, $left:$right] in ${imh}x$imw"
            }

            buildFull(mask, imh, imw, top, bottom, left, right, origH, origW, extraPad).use { full ->
                // Zero everything outside the instance's own box, so two adjacent documents cannot bleed
                // into each other's contour. The box is expressed in original pixels, so the extra padding
                // comes off first.
                full.zeroOutsideBox(
                    box.x1 - extraPad[0], box.y1 - extraPad[1],
                    box.x2 - extraPad[0], box.y2 - extraPad[1],
                )

                full.threshold(maskFilter).use { binary ->
                    val contours = Contours.findExternalContours(binary)
                    if (contours.isEmpty()) {
                        return@use
                    }
                    // The LARGEST contour by area. A mask can produce specks; the document is the big one.
                    segments += contours.maxByOrNull { Contours.contourArea(it) }!!
                }
            }
        }
        return segments
    }

    private fun buildFull(
        mask: FloatArray,
        imh: Int,
        imw: Int,
        top: Int,
        bottom: Int,
        left: Int,
        right: Int,
        origH: Int,
        origW: Int,
        extraPad: IntArray,
    ): FloatMask =
        FloatMask.fromValues(mask, imh, imw).use { atProto ->
            atProto.crop(top, bottom, left, right).use { cropped ->
                // Upscale to the PRE-letterbox size, then strip the extra padding — the reference's order,
                // and it matters because the extra padding is expressed in original pixels.
                cropped.resize(origW, origH).use { upscaled ->
                    upscaled.crop(extraPad[1], origH - extraPad[1],
                        extraPad[0], origW - extraPad[0])
                }
            }
        }
}
