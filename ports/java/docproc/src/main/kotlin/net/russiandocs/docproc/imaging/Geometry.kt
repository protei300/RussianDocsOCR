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

/**
 * Where one page landed on the stitched canvas: `(scale, dx, dy)` from `stitch_pages`.
 *
 * A box `(x, y)` found on that ORIGINAL (unresized) page maps onto the canvas as
 * `(x*scale + dx, y*scale + dy)` — `Pipeline._fields_from_pages` (pipeline.py:1256).
 */
public data class Placement(public val scale: Double, public val dx: Double, public val dy: Double)

/**
 * What [Geometry.fixPerspective] produces: the stitched canvas, plus — ONLY for a two-page spread —
 * the individual pre-resize pages and where each landed. Port of `DocDetector.predict_transform`'s
 * `pages`/`page_placements` (doc_detector.py:170-186).
 *
 * [pages] is empty for zero or one detected page, matching the reference: `_fields_detector` only takes
 * the per-page path when `len(pages) >= 2`, so a single-page document's `pages` list is never read and is
 * not worth carrying. [pages] and [placements] are the same length and same order — the ORIGINAL segment
 * order, not the left-to-right/top-to-bottom order used to decide the stitch direction.
 *
 * [pages] is owned by the caller once returned.
 */
public class PerspectiveResult(
    public val canvas: Image?,
    public val ok: Boolean,
    public val pages: List<Image> = emptyList(),
    public val placements: List<Placement> = emptyList(),
)

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
    ): PerspectiveResult {
        val pages = mutableListOf<Pair<List<Pt>, Image>>()
        // Ownership tracking: everything in `pages` is closed in the `finally` UNLESS it was handed to the
        // caller first — either as the single-page return or inside a two-page [PerspectiveResult]. `handedOff`
        // records which indices escaped, since `pages` itself is cleared only in the single-page branch.
        var handedOff = false
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
                return PerspectiveResult(null, false)
            }
            if (pages.size == 1) {
                val only = pages[0].second
                pages.clear() // ownership moves to the caller
                handedOff = true
                // A single detected page never takes the per-page field-detection path in the reference
                // (`len(pages) >= 2` gates it — pipeline.py:1238), so its own `pages`/`placements` are not
                // worth returning; the canvas IS the page.
                return PerspectiveResult(only, true)
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
            // two equal minima in detection order. `order` keeps the ORIGINAL index of each entry — the
            // reference's `placements` array is indexed by that original index (stitch_pages, doc_detector's
            // `image_transformation.py:371`), not by stitch position, and `_fields_from_pages` walks `pages`
            // in ITS OWN (original) order too.
            val horizontal = resolved == StackDirection.HORIZONTAL
            val order = if (horizontal) {
                pages.indices.sortedBy { i -> pages[i].first.minOf { it.x } }
            } else {
                pages.indices.sortedBy { i -> pages[i].first.minOf { it.y } }
            }

            // **The pages are RESIZED to a common dimension before joining.** This is the step whose
            // absence produced a 727x528 canvas against the golden's 701x505 in the .NET port: hconcat and
            // vconcat require the shared dimension to match exactly, so the reference scales every page to
            // the SMALLEST of them and scales the other axis proportionally, rounding half to even.
            val common = if (horizontal) {
                order.minOf { pages[it].second.height }
            } else {
                order.minOf { pages[it].second.width }
            }

            val scaled = mutableListOf<Image>()
            val placements = arrayOfNulls<Placement>(pages.size)
            var offset = 0.0
            try {
                for (i in order) {
                    val warped = pages[i].second
                    // `scale` computed FIRST, exactly as the reference's `scale = common / w.shape[...]`
                    // then `new_w/new_h = round(w.shape[...] * scale)` — multiply-then-divide in one
                    // expression is a DIFFERENT float64 operation order and can round half a bit
                    // differently (CONVENTIONS §6), which is why this is two statements, not one.
                    val scale = if (horizontal) {
                        common.toDouble() / warped.height
                    } else {
                        common.toDouble() / warped.width
                    }
                    val other = if (horizontal) {
                        max(1, PyNum.roundHalfEvenToInt(warped.width * scale))
                    } else {
                        max(1, PyNum.roundHalfEvenToInt(warped.height * scale))
                    }
                    scaled += if (horizontal) {
                        Io.resize(warped, other, common, Interpolation.LINEAR)
                    } else {
                        Io.resize(warped, common, other, Interpolation.LINEAR)
                    }
                    placements[i] = if (horizontal) {
                        Placement(scale, offset, 0.0)
                    } else {
                        Placement(scale, 0.0, offset)
                    }
                    offset += other
                }

                // `scaled` is built in `order` (stitch position), matching `np.hstack(resized)`/
                // `np.vstack(resized)` on the reference's `resized` list — joining follows POSITION.
                // The per-page pieces returned below follow ORIGINAL index instead; see `placements`.
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

                // Ownership of the original (unresized) per-page images transfers to the returned result —
                // `_fields_from_pages` reads them at full per-page resolution, not the joining-time scale.
                val originalPages = pages.map { it.second }
                pages.clear()
                handedOff = true
                // Every index was filled: `order` is a permutation of `pages.indices`.
                return PerspectiveResult(joined, true, originalPages, placements.map { it!! })
            } finally {
                scaled.forEach { it.close() }
            }
        } catch (e: IllegalArgumentException) {
            return PerspectiveResult(null, false)
        } finally {
            if (!handedOff) {
                pages.forEach { it.second.close() }
            }
        }
    }

    /**
     * Merges already-rectified pages into one canvas, reporting where each page landed. Port of
     * `image_transformation.py::stitch_pages`, as a STANDALONE function over pages the caller already
     * warped — `page_registration.py`'s `_register_pages` calls this with `stack='vertical'` explicitly
     * (never `AUTO`) on pages it built from template registration or a Borders quad, which is why this
     * takes [Image]s directly rather than segmentation contours the way [fixPerspective] does.
     *
     * Does NOT take ownership of [pages] — the caller warped them and the caller closes them; only the
     * joined result and its own resize intermediates are this function's to manage.
     */
    public fun stitchPages(pages: List<Image>, quads: List<List<Pt>>, direction: StackDirection): Pair<Image?, List<Placement>> {
        if (pages.isEmpty()) return null to emptyList()
        if (pages.size == 1) return pages[0].clone() to listOf(Placement(1.0, 0.0, 0.0))

        var resolved = direction
        if (direction == StackDirection.AUTO) {
            val c0 = centroid(quads[0])
            val c1 = centroid(quads[1])
            resolved = if (abs(c0.x - c1.x) >= abs(c0.y - c1.y)) StackDirection.HORIZONTAL else StackDirection.VERTICAL
        }
        val horizontal = resolved == StackDirection.HORIZONTAL
        val order = pages.indices.sortedBy { i -> if (horizontal) quads[i].minOf { it.x } else quads[i].minOf { it.y } }

        val common = if (horizontal) order.minOf { pages[it].height } else order.minOf { pages[it].width }
        val scaled = mutableListOf<Image>()
        val placements = arrayOfNulls<Placement>(pages.size)
        var offset = 0.0
        try {
            for (i in order) {
                val warped = pages[i]
                val scale = if (horizontal) common.toDouble() / warped.height else common.toDouble() / warped.width
                val other = if (horizontal) {
                    max(1, PyNum.roundHalfEvenToInt(warped.width * scale))
                } else {
                    max(1, PyNum.roundHalfEvenToInt(warped.height * scale))
                }
                scaled += if (horizontal) {
                    Io.resize(warped, other, common, Interpolation.LINEAR)
                } else {
                    Io.resize(warped, common, other, Interpolation.LINEAR)
                }
                placements[i] = if (horizontal) Placement(scale, offset, 0.0) else Placement(scale, 0.0, offset)
                offset += other
            }
            var joined = scaled[0].clone()
            for (k in 1 until scaled.size) {
                val combined = if (horizontal) Contours.hStack(joined, scaled[k]) else Contours.vStack(joined, scaled[k])
                joined.close()
                joined = combined
            }
            return joined to placements.map { it!! }
        } finally {
            scaled.forEach { it.close() }
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
