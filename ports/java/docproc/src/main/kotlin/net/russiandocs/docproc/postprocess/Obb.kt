package net.russiandocs.docproc.postprocess

import net.russiandocs.docproc.tensors.NdArray
import net.russiandocs.docproc.tensors.PyNum
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.RotatedRect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * One oriented detection: the row `[cx, cy, w, h, angle, conf, cls_idx, label]` of
 * `YOLOOBBDetectorPostprocessing`, on the image the preprocessor was given. The angle is in RADIANS.
 *
 * Every number is a float32 carried in a Double, which is what `float(np.float32)` gives in the reference.
 */
public class ObbBox(
    public val cx: Double,
    public val cy: Double,
    public val w: Double,
    public val h: Double,
    public val angle: Double,
    public val conf: Double,
    public val cls: Int,
    public val label: String,
)

public data class ObbResult(val boxes: List<ObbBox>) : ModelResult

/**
 * Decodes the raw ultralytics OBB head and suppresses overlaps by rotated IoU. Port of
 * `processing/postprocessing.py::YOLOOBBDetectorPostprocessing`.
 *
 * Output layout (after squeeze) is `[4 + nc + 1, anchors]` or its transpose: `cx, cy, w, h`, one score per
 * class, then the angle in radians. The class set is read from `Labels`; the thresholds from `CLS` and `IOU`.
 *
 * **Everything numeric is float32, as in the reference.** The row comes out of a float32 array, so in numpy 2
 * `(cx - pad) / ratio - extra` stays float32 (a Python float is "weak" and adopts the array's type), and so does
 * the IoU test: `union` is a float32 scalar and `inter / union` is compared to the threshold in float32. A
 * double implementation would agree to ~1e-5, which is inside the tolerance — but a box sitting on the 0.45
 * threshold is exactly where the two would part ways, so it is written the way the reference computes it.
 */
public class YoloObbDetector(
    private val labels: List<String>,
    private val iou: Double,
    private val cls: Double,
) : Postprocessor {

    override fun apply(output: NdArray, context: Context): ModelResult =
        ObbResult(decode(output, context))

    public fun decode(output: NdArray, context: Context): List<ObbBox> {
        val data = output.asFloat32()
        if (data.isEmpty()) {
            return emptyList()
        }

        // np.squeeze: every axis of length 1 goes.
        val dims = output.shape.filter { it != 1 }
        require(dims.size == 2) {
            "postprocess: OBB detector expects a [features, anchors] matrix after squeeze, got " +
                NdArray.describe(output.shape)
        }
        val first = dims[0]
        val second = dims[1]
        // `if vector.shape[0] < vector.shape[1]: vector = vector.T`
        val transposed = first < second
        val anchors = if (transposed) second else first
        val features = if (transposed) first else second
        fun at(anchor: Int, feature: Int): Float =
            if (transposed) data[feature * second + anchor] else data[anchor * second + feature]

        val nc = labels.size
        require(features >= 4 + nc + 1) {
            "postprocess: OBB row of $features values is too small for $nc classes"
        }

        val threshold = cls.toFloat()
        val boxes = ArrayList<FloatArray>()    // cx, cy, w, h
        val angles = ArrayList<Float>()
        val confs = ArrayList<Float>()
        val classes = ArrayList<Int>()
        for (a in 0 until anchors) {
            // `det.max(axis=1)` and `det.argmax(axis=1)`: the first maximum wins.
            var best = 0
            var bestScore = at(a, 4)
            for (c in 1 until nc) {
                val score = at(a, 4 + c)
                if (score > bestScore) {
                    best = c
                    bestScore = score
                }
            }
            if (!(bestScore > threshold)) {
                continue
            }
            boxes += floatArrayOf(at(a, 0), at(a, 1), at(a, 2), at(a, 3))
            angles += at(a, 4 + nc)
            confs += bestScore
            classes += best
        }
        if (boxes.isEmpty()) {
            return emptyList()
        }

        val keep = rotatedNms(boxes, angles, confs, iou.toFloat())

        val padX = context.padLetter[0].toFloat()
        val padY = context.padLetter[1].toFloat()
        val ratio = context.ratio.toFloat()
        val extraX = context.padExtra[0].toFloat()
        val extraY = context.padExtra[1].toFloat()

        val results = ArrayList<ObbBox>(keep.size)
        for (i in keep) {
            val (cx0, cy0, w0, h0) = boxes[i]
            // rescale centre and size from the letterboxed space back to the original; all float32
            val cx = (cx0 - padX) / ratio - extraX
            val cy = (cy0 - padY) / ratio - extraY
            val w = w0 / ratio
            val h = h0 / ratio
            val label = if (labels.isNotEmpty()) labels[classes[i]] else classes[i].toString()
            results += ObbBox(cx.toDouble(), cy.toDouble(), w.toDouble(), h.toDouble(),
                angles[i].toDouble(), confs[i].toDouble(), classes[i], label)
        }

        // reading order: top-to-bottom, then left-to-right. `round(r[1] / 10)` is Python's builtin round: half to
        // even. The sort is stable.
        return results.sortedWith(
            compareBy<ObbBox>({ PyNum.roundHalfEven(it.cy / 10) }, { it.cx }),
        )
    }

    private companion object {
        /** `np.degrees` on a float32: `x * (180.0f / NPY_PIf)`. */
        val DEGREES_PER_RADIAN: Float = 180.0f / Math.PI.toFloat()

        /** Greedy NMS on rotated IoU, best score first. Returns the kept indices, in the order they were picked. */
        fun rotatedNms(
            boxes: List<FloatArray>,
            angles: List<Float>,
            confs: List<Float>,
            threshold: Float,
        ): List<Int> {
            // `np.argsort(-conf)`: descending score. Stable here; numpy's quicksort is not, but equal float32
            // scores on two overlapping boxes are not a case worth reproducing.
            var order = confs.indices.sortedByDescending { confs[it] }
            val kept = ArrayList<Int>()
            while (order.isNotEmpty()) {
                val i = order.first()
                kept += i
                order = order.drop(1).filter { j ->
                    rotatedIou(boxes[i], angles[i], boxes[j], angles[j]) < threshold
                }
            }
            return kept
        }

        /** Rotated-rectangle IoU through OpenCV's intersection, `rotated_iou`. float32 where the reference is. */
        fun rotatedIou(a: FloatArray, angleA: Float, b: FloatArray, angleB: Float): Float {
            val ra = RotatedRect(Point(a[0].toDouble(), a[1].toDouble()), Size(a[2].toDouble(), a[3].toDouble()),
                (angleA * DEGREES_PER_RADIAN).toDouble())
            val rb = RotatedRect(Point(b[0].toDouble(), b[1].toDouble()), Size(b[2].toDouble(), b[3].toDouble()),
                (angleB * DEGREES_PER_RADIAN).toDouble())
            val region = Mat()
            try {
                val kind = Imgproc.rotatedRectangleIntersection(ra, rb, region)
                // INTERSECT_NONE leaves nothing to measure: the area of an empty region is 0.
                val inter: Double = if (kind == 0 || region.empty()) 0.0 else Imgproc.contourArea(region)
                // float32 scalars all the way: `a[2] * a[3] + b[2] * b[3] - inter`, `inter` weak.
                val union = a[2] * a[3] + b[2] * b[3] - inter.toFloat()
                return if (union > 0) inter.toFloat() / union else 0.0f
            } finally {
                region.release()
            }
        }
    }
}
