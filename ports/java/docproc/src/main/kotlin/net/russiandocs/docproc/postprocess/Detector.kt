package net.russiandocs.docproc.postprocess

import net.russiandocs.docproc.tensors.NdArray
import net.russiandocs.docproc.tensors.Ops
import net.russiandocs.docproc.tensors.PyNum
import kotlin.math.max
import kotlin.math.min
import kotlin.math.truncate

/** One detection. Mutable, because the coordinates are rewritten in place when mapped back. */
public class Box {
    public var x1: Double = 0.0
    public var y1: Double = 0.0
    public var x2: Double = 0.0
    public var y2: Double = 0.0
    public var conf: Double = 0.0
    public var cls: Int = 0
    public var label: String = ""

    /** Mask coefficients, for a segmentation head. Empty for a plain detector. */
    public var seg: DoubleArray = DoubleArray(0)

    public fun copy(): Box = Box().also {
        it.x1 = x1; it.y1 = y1; it.x2 = x2; it.y2 = y2
        it.conf = conf; it.cls = cls; it.label = label
        it.seg = seg.copyOf()
    }
}

public data class DetectResult(val boxes: List<Box>) : ModelResult

/** Which suppression the model's output type asks for. */
public enum class NmsMode {
    CLASS_AGNOSTIC,
    PER_CLASS,
}

/**
 * Decodes an anchors-first YOLO output and suppresses overlaps.
 *
 * One type with an [NmsMode] field rather than a base class and a subclass overriding the suppression —
 * CONVENTIONS §5. The reference has `PerClassYOLODetectorPostprocessing` inheriting from the plain one;
 * flattening it removes a place where Go's non-virtual embedding would silently call the wrong method, and
 * reads the same in all four languages.
 */
public class YoloDetector(
    private val labels: List<String>,
    private val iou: Double,
    private val cls: Double,
    private val mode: NmsMode,
    /**
     * When set, the decode stops before truncation and labelling.
     *
     * The segmentation path needs the RAW float coordinates and the mask coefficients — truncating first
     * loses sub-pixel information the mask crop depends on.
     */
    public val numpyOnly: Boolean = false,
) : Postprocessor {

    /**
     * A copy with raw-coordinate mode on.
     *
     * Exists so the loader switch stays the ONE place a detector is constructed: the segmentation model
     * needs the same detector with one flag flipped, and building a second one there would duplicate the
     * argument list the switch already owns.
     */
    public fun withNumpyOnly(): YoloDetector = YoloDetector(labels, iou, cls, mode, numpyOnly = true)

    override fun apply(output: NdArray, context: Context): ModelResult =
        DetectResult(decode(output, context))

    public fun decode(output: NdArray, context: Context): List<Box> {
        val data = output.asFloat32()
        var shape = output.shape
        if (shape.size == 3 && shape[0] == 1) {
            shape = shape.copyOfRange(1, shape.size)
        }
        require(shape.size == 2) {
            "postprocess: detector expects [anchors, 4+nc(+seg)], got " +
                NdArray.describe(output.shape)
        }

        val anchors = shape[0]
        val stride = shape[1]
        val nc = labels.size
        require(stride >= 4 + nc) {
            "postprocess: row width $stride is too small for $nc classes"
        }
        val segLen = stride - 4 - nc

        val boxes = ArrayList<Box>(64)
        for (a in 0 until anchors) {
            val base = a * stride

            // Strict `>` for the best class, matching np.argmax's first-maximum rule.
            var best = 0
            var bestScore = Double.NEGATIVE_INFINITY
            for (c in 0 until nc) {
                val score = data[base + 4 + c].toDouble()
                if (score > bestScore) {
                    best = c
                    bestScore = score
                }
            }
            // `!(score > cls)` rather than `score <= cls`: identical for real numbers, and it keeps the
            // reference's own spelling, which also handles NaN the same way.
            if (!(bestScore > cls)) {
                continue
            }

            val cx = data[base].toDouble()
            val cy = data[base + 1].toDouble()
            val w = data[base + 2].toDouble()
            val h = data[base + 3].toDouble()
            val box = Box().also {
                it.x1 = cx - w / 2
                it.y1 = cy - h / 2
                it.x2 = cx + w / 2
                it.y2 = cy + h / 2
                it.conf = bestScore
                it.cls = best
            }
            if (segLen > 0) {
                box.seg = DoubleArray(segLen) { i -> data[base + 4 + nc + i].toDouble() }
            }
            boxes += box
        }

        if (boxes.isEmpty()) {
            return emptyList()
        }

        val keep = if (mode == NmsMode.PER_CLASS) {
            nmsPerClass(boxes, iou)
        } else {
            nms(boxes.indices.toList(), boxes, iou)
        }

        // **Stable sort, reading order.** `sortedWith` is stable on the JVM; `Collections.sort` on a
        // mutable list is too, but an unstable primitive sort is not. Two boxes with the same y1 must keep
        // the order suppression left them in, because that order decides which word comes first in a
        // joined field string.
        var kept = keep.map { boxes[it] }
        kept = kept.sortedWith(compareBy({ it.y1 }, { it.x1 }))

        if (context.resize) {
            mapBack(kept, context)
        }

        if (!numpyOnly) {
            for (b in kept) {
                // TRUNCATION, not rounding — the reference casts to int here, after having rounded
                // half-to-even in mapBack. Two different operations, in that order.
                b.x1 = truncate(b.x1)
                b.y1 = truncate(b.y1)
                b.x2 = truncate(b.x2)
                b.y2 = truncate(b.y2)
                b.label = if (b.cls in labels.indices) labels[b.cls] else "Unsupported class"
            }
        }
        return kept
    }

    /**
     * Undoes the letterbox and the extra padding, then clamps and rounds.
     *
     * The order matters and is the reference's: unpad, unscale, un-extra-pad, clamp negatives to zero,
     * THEN round half-to-even.
     */
    private fun mapBack(boxes: List<Box>, ctx: Context) {
        for (b in boxes) {
            b.x1 = (b.x1 - ctx.padLetter[0]) / ctx.ratio - ctx.padExtra[0]
            b.x2 = (b.x2 - ctx.padLetter[0]) / ctx.ratio - ctx.padExtra[0]
            b.y1 = (b.y1 - ctx.padLetter[1]) / ctx.ratio - ctx.padExtra[1]
            b.y2 = (b.y2 - ctx.padLetter[1]) / ctx.ratio - ctx.padExtra[1]

            b.x1 = max(0.0, b.x1)
            b.y1 = max(0.0, b.y1)
            b.x2 = max(0.0, b.x2)
            b.y2 = max(0.0, b.y2)
            b.conf = max(0.0, b.conf)
            for (j in b.seg.indices) {
                // **Clamps the WHOLE row, mask coefficients included.** It reads like an oversight — the
                // intent was surely to clamp coordinates — but the reference does
                // `detect_res[detect_res < 0] = 0` across every column, and those coefficients go straight
                // to the segmentor. A port that clamps only the coordinates gets different masks.
                b.seg[j] = max(0.0, b.seg[j])
            }

            b.x1 = PyNum.roundHalfEven(b.x1)
            b.y1 = PyNum.roundHalfEven(b.y1)
            b.x2 = PyNum.roundHalfEven(b.x2)
            b.y2 = PyNum.roundHalfEven(b.y2)
            b.conf = Ops.roundHalfEven(b.conf, 3)
        }
    }

    /**
     * Non-maximum suppression over a subset of indices.
     *
     * **The tie rule is load-bearing.** The candidates are sorted ASCENDING by score with a stable sort,
     * and the highest — the LAST element — is picked each round. On a score tie that keeps the one with the
     * greater original index, which is what `np.argsort` plus `order[-1]` does. Picking the first instead
     * changes which of two equally-confident boxes survives.
     *
     * `ratio < threshold` survives, matching `np.where(ratio < threshold)`.
     */
    private fun nms(indices: List<Int>, boxes: List<Box>, threshold: Double): List<Int> {
        val areas = indices.associateWith { i ->
            (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1)
        }

        // Stable ascending by score — np.argsort.
        var order = indices.sortedBy { boxes[it].conf }

        val picked = ArrayList<Int>()
        while (order.isNotEmpty()) {
            val index = order.last()
            picked += index
            order = order.dropLast(1)

            order = order.filter { j ->
                val x1 = max(boxes[index].x1, boxes[j].x1)
                val y1 = max(boxes[index].y1, boxes[j].y1)
                val x2 = min(boxes[index].x2, boxes[j].x2)
                val y2 = min(boxes[index].y2, boxes[j].y2)
                val inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                val ratio = inter / (areas.getValue(index) + areas.getValue(j) - inter)
                ratio < threshold
            }
        }
        return picked
    }

    /**
     * Suppression within each class independently.
     *
     * Classes are visited in ASCENDING order, matching `np.unique`. The resulting index order feeds the
     * reading-order sort afterwards, so a different visit order can break ties differently — which is why
     * this sorts the class list rather than iterating a set.
     */
    private fun nmsPerClass(boxes: List<Box>, threshold: Double): List<Int> {
        val classes = boxes.map { it.cls }.distinct().sorted()
        val keep = ArrayList<Int>()
        for (c in classes) {
            keep += nms(boxes.indices.filter { boxes[it].cls == c }, boxes, threshold)
        }
        return keep
    }
}
