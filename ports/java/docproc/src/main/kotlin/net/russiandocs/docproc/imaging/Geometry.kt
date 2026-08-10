package net.russiandocs.docproc.imaging

import net.russiandocs.docproc.tensors.PyNum
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/** How a multi-page spread is joined back together. */
public enum class StackDirection {
    /** Decide from the page geometry, which is what the reference does. */
    AUTO,
    HORIZONTAL,
    VERTICAL,
}

/** Quadrilateral geometry: ordering corners, expanding a margin, and the perspective correction. */
public object Geometry {

    /**
     * The outward cushion applied to a detected document quad.
     *
     * 0.01, from `DOC_MARGIN_FRAC` in the reference. It is a fraction of the document's OWN size that each
     * EDGE moves out by, so the applied scale is `1 + 2*margin`. The .NET port first used 0.005 from memory
     * and every single-page canvas came out about 1% large — 910 columns against the golden 901 — which is
     * the whole error, visible only because the shape is compared exactly.
     */
    public const val DOC_MARGIN_FRACTION: Double = 0.01

    /**
     * Orders four points as top-left, top-right, bottom-right, bottom-left.
     *
     * By coordinate SUM and DIFFERENCE, exactly as the reference: the smallest `x+y` is top-left, the
     * largest is bottom-right, and the extremes of `y-x` give the other two. It is not a sort by angle and
     * it does not generalise — but it is what produced the goldens.
     *
     * Ties resolve to the FIRST index reaching the extreme, because the comparisons are strict. On a
     * perfectly axis-aligned rectangle two corners can share a sum, and picking the later one rotates the
     * whole quad.
     */
    public fun orderPoints(points: List<Pt>): List<Pt>? {
        if (points.size != 4) {
            return null
        }
        var minSum = 0
        var maxSum = 0
        var minDiff = 0
        var maxDiff = 0
        for (i in points.indices) {
            val sum = points[i].x + points[i].y
            val diff = points[i].y - points[i].x
            if (sum < points[minSum].x + points[minSum].y) minSum = i
            if (sum > points[maxSum].x + points[maxSum].y) maxSum = i
            if (diff < points[minDiff].y - points[minDiff].x) minDiff = i
            if (diff > points[maxDiff].y - points[maxDiff].x) maxDiff = i
        }
        return listOf(points[minSum], points[minDiff], points[maxSum], points[maxDiff])
    }

    /**
     * Reduces a contour to four corners.
     *
     * Tries increasing Douglas-Peucker tolerances until one yields exactly four points, and falls back to
     * the minimum-area rectangle of the ORIGINAL contour — not of the hull. The fraction ladder is the
     * reference's and the order matters: a coarser tolerance can also produce four points, but different
     * ones.
     */
    public fun extractQuad(contour: List<Pt>): List<Pt>? {
        if (contour.size < 4) {
            return null
        }
        val hull = Contours.convexHull(contour)
        if (hull.isEmpty()) {
            return null
        }
        val perimeter = Contours.arcLength(hull)
        for (fraction in listOf(0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15)) {
            val approx = Contours.approxPolyDp(hull, fraction * perimeter)
            if (approx.size == 4) {
                return approx
            }
        }
        return Contours.minAreaRectPoints(contour)
    }

    /** Scales a quadrilateral outward from its centroid by a fraction of its size. */
    public fun expandQuad(quad: List<Pt>, margin: Double): List<Pt> {
        if (margin <= 0) {
            return quad.toList()
        }
        var cx = 0.0
        var cy = 0.0
        for (p in quad) {
            cx += p.x
            cy += p.y
        }
        cx /= quad.size
        cy /= quad.size
        val scale = 1.0 + 2.0 * margin
        return quad.map { Pt(cx + (it.x - cx) * scale, cy + (it.y - cy) * scale) }
    }

    /**
     * Warps a quadrilateral to an axis-aligned image.
     *
     * The output size comes from the LONGER of each opposing pair of edges, rounded HALF TO EVEN. Rounding
     * away from zero here — which `Math.round` and `roundToInt` both do — gives a canvas one pixel
     * different in each dimension, and every box downstream is then compared against a golden made on a
     * differently-sized canvas.
     */
    public fun fourPointTransform(image: Image, quad: List<Pt>): Pair<Image?, Boolean> {
        val rect = orderPoints(quad) ?: return null to false

        val (tl, tr, br, bl) = listOf(rect[0], rect[1], rect[2], rect[3])
        val width = PyNum.roundHalfEvenToInt(max(distance(br, bl), distance(tr, tl)))
        val height = PyNum.roundHalfEvenToInt(max(distance(tr, br), distance(tl, bl)))
        if (width < 2 || height < 2) {
            return null to false
        }
        return try {
            Contours.warpPerspectiveQuad(image, rect, width, height) to true
        } catch (e: Exception) {
            // A degenerate quad makes getPerspectiveTransform throw. The reference returns the original
            // image in that case rather than failing the document, so the caller needs a false here.
            null to false
        }
    }

    public fun distance(a: Pt, b: Pt): Double =
        sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y))

    /**
     * Corrects perspective for one or more detected pages and stitches them together.
     *
     * Single page: order, expand by the margin, clamp, warp. Two pages: warp each, then join —
     * HORIZONTALLY when the pages sit side by side and VERTICALLY when they are stacked, decided from the
     * centroids.
     */
    public fun fixPerspective(
        image: Image,
        segments: List<List<Pt>>,
        direction: StackDirection,
        margin: Double,
    ): Pair<Image?, Boolean> {
        val pages = mutableListOf<Pair<List<Pt>, Image>>()
        try {
            for (segment in segments) {
                val quad = extractQuad(segment) ?: continue

                // ORDER FIRST, then expand, then CLAMP to the image. All three steps and their order are
                // the reference's: expanding an unordered quad moves the corners about its centroid
                // correctly but hands fourPointTransform points it will reorder anyway, and skipping the
                // clamp lets the cushion push a corner outside the image, where the warp samples the
                // border colour and widens the canvas.
                val ordered = orderPoints(quad) ?: continue
                val expanded = expandQuad(ordered, margin).map { p ->
                    Pt(
                        p.x.coerceIn(0.0, image.width.toDouble()),
                        p.y.coerceIn(0.0, image.height.toDouble()),
                    )
                }

                val (warped, ok) = fourPointTransform(image, expanded)
                if (!ok || warped == null) {
                    continue
                }
                pages += expanded to warped
            }

            if (pages.isEmpty()) {
                return null to false
            }
            if (pages.size == 1) {
                val only = pages[0].second
                pages.clear() // ownership moves to the caller
                return only to true
            }

            // Direction from the FIRST TWO pages' centroids only, matching the reference. A wider
            // horizontal separation means the pages sit side by side.
            var resolved = direction
            if (direction == StackDirection.AUTO) {
                val c0 = centroid(pages[0].first)
                val c1 = centroid(pages[1].first)
                resolved = if (abs(c0.x - c1.x) >= abs(c0.y - c1.y)) {
                    StackDirection.HORIZONTAL
                } else {
                    StackDirection.VERTICAL
                }
            }

            // Ordered by the quad's MINIMUM coordinate, not its centroid: two pages of different sizes can
            // have centroids in the opposite order to their left edges. sortedBy is STABLE, which keeps
            // two equal minima in detection order.
            val horizontal = resolved == StackDirection.HORIZONTAL
            val inOrder = if (horizontal) {
                pages.sortedBy { pg -> pg.first.minOf { it.x } }
            } else {
                pages.sortedBy { pg -> pg.first.minOf { it.y } }
            }

            // **The pages are RESIZED to a common dimension before joining.** This is the step whose
            // absence produced a 727x528 canvas against the golden's 701x505 in the .NET port: hconcat and
            // vconcat require the shared dimension to match exactly, so the reference scales every page to
            // the SMALLEST of them and scales the other axis proportionally, rounding half to even.
            val common = if (horizontal) {
                inOrder.minOf { it.second.height }
            } else {
                inOrder.minOf { it.second.width }
            }

            val scaled = mutableListOf<Image>()
            try {
                for ((_, warped) in inOrder) {
                    val other = if (horizontal) {
                        max(1, PyNum.roundHalfEvenToInt(
                            warped.width.toDouble() * common / warped.height))
                    } else {
                        max(1, PyNum.roundHalfEvenToInt(
                            warped.height.toDouble() * common / warped.width))
                    }
                    scaled += if (horizontal) {
                        Io.resize(warped, other, common, Interpolation.LINEAR)
                    } else {
                        Io.resize(warped, common, other, Interpolation.LINEAR)
                    }
                }

                var joined = scaled[0].clone()
                for (k in 1 until scaled.size) {
                    val combined = if (horizontal) {
                        Contours.hStack(joined, scaled[k])
                    } else {
                        Contours.vStack(joined, scaled[k])
                    }
                    joined.close()
                    joined = combined
                }
                return joined to true
            } finally {
                scaled.forEach { it.close() }
            }
        } catch (e: IllegalArgumentException) {
            return null to false
        } finally {
            pages.forEach { it.second.close() }
        }
    }

    private fun centroid(quad: List<Pt>): Pt {
        var cx = 0.0
        var cy = 0.0
        for (p in quad) {
            cx += p.x
            cy += p.y
        }
        return Pt(cx / quad.size, cy / quad.size)
    }
}

/**
 * A single-channel float32 mask, at whatever resolution the proto masks came in.
 *
 * Kept as float until the very last step. The threshold that turns it binary is the ONLY place a decision
 * is made, so any rounding earlier would move the contour — which is the thing being compared.
 *
 * **float32 throughout, never widened.** The reference accumulates the mask in float32, and "improving"
 * that to double changes the mask boundary and therefore the extracted quadrilateral.
 */
public class FloatMask private constructor(private var backing: Mat?) : AutoCloseable {

    private val mat: Mat get() = backing ?: throw IllegalStateException("imaging: mask is closed")

    public val height: Int get() = mat.rows()
    public val width: Int get() = mat.cols()

    /** Crops by row/column bounds, exclusive at the far edge — a numpy slice. */
    public fun crop(top: Int, bottom: Int, left: Int, right: Int): FloatMask {
        val view = mat.submat(org.opencv.core.Rect(left, top, right - left, bottom - top))
        return try {
            FloatMask(view.clone())
        } finally {
            view.release()
        }
    }

    /** Resizes with bilinear interpolation, matching the reference's mask upscale. */
    public fun resize(width: Int, height: Int): FloatMask {
        val dst = Mat()
        Imgproc.resize(mat, dst, Size(width.toDouble(), height.toDouble()), 0.0, 0.0,
            Imgproc.INTER_LINEAR)
        return FloatMask(dst)
    }

    /**
     * Zeroes everything outside a box, so two adjacent documents cannot bleed into each other's contour.
     *
     * **The comparisons are STRICT**, so the boundary row and column are zeroed too — matching the
     * reference's `clip_boxes` exactly. Using inclusive bounds adds a one-pixel rim to every mask, which
     * survives thresholding and shifts the contour.
     *
     * Done by building the KEPT region as a submat copy rather than by per-pixel `put` calls: a JNI call
     * per pixel is roughly a million crossings on a 1000x1000 mask, and the JVM binding has no bulk
     * setter for a scattered pattern. Zero the whole thing, then copy the interior back.
     */
    public fun zeroOutsideBox(x1: Double, y1: Double, x2: Double, y2: Double) {
        val m = mat
        // Strict comparisons mean the kept range starts at the first index STRICTLY greater than x1.
        val left = kotlin.math.floor(x1 + 1.0).toInt().coerceIn(0, m.cols())
        val top = kotlin.math.floor(y1 + 1.0).toInt().coerceIn(0, m.rows())
        val right = kotlin.math.ceil(x2).toInt().coerceIn(0, m.cols())
        val bottom = kotlin.math.ceil(y2).toInt().coerceIn(0, m.rows())

        if (right <= left || bottom <= top) {
            m.setTo(org.opencv.core.Scalar(0.0))
            return
        }

        val keptRect = org.opencv.core.Rect(left, top, right - left, bottom - top)
        val kept = m.submat(keptRect).clone()
        try {
            m.setTo(org.opencv.core.Scalar(0.0))
            val target = m.submat(keptRect)
            try {
                kept.copyTo(target)
            } finally {
                target.release()
            }
        } finally {
            kept.release()
        }
    }

    /** Thresholds to an 8-bit binary mask, which is what `findContours` wants. */
    public fun threshold(value: Double): Image {
        val binary = Mat()
        try {
            Imgproc.threshold(mat, binary, value, 255.0, Imgproc.THRESH_BINARY)
            val eightBit = Mat()
            binary.convertTo(eightBit, CvType.CV_8UC1)
            return Image.wrap(eightBit)
        } finally {
            binary.release()
        }
    }

    override fun close() {
        backing?.release()
        backing = null
    }

    public companion object {
        public fun fromValues(values: FloatArray, height: Int, width: Int): FloatMask {
            val mat = Mat(height, width, CvType.CV_32FC1)
            mat.put(0, 0, values)
            return FloatMask(mat)
        }
    }
}
