package net.russiandocs.docproc.modules

import net.russiandocs.docproc.imaging.Contours
import net.russiandocs.docproc.imaging.Geometry
import net.russiandocs.docproc.imaging.Pt
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point as CvPoint
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/**
 * Page quad from straight-line fits to the segmentation contour. Port of `page_registration/quad_fit.py`
 * (`fit_quad_lines`).
 *
 * The Borders mask usually follows the physical page edge well; what breaks the polygon-corner
 * `Geometry.extractQuad` is the CORNER choice: a thumb merged into the mask, a bent corner or a run of
 * stair-steps along a blurred edge moves one polygon vertex, and the whole side swings. A straight line
 * fitted to the hundred-odd contour points of that side survives the thumb (RANSAC drops it as an
 * outlier) where a polygon vertex does not.
 *
 * **RANSAC sampling here uses Kotlin's own PRNG, not a bit-exact port of NumPy's PCG64.** The consensus
 * search draws 200 random point PAIRS from routinely 100+ contour points on a genuinely straight edge —
 * the dominant line has a large majority of true inliers, and two rounds of `cv2.fitLine` least-squares
 * refit after the search pull ANY "good enough" starting pair to essentially the same fitted line. This
 * is a measured claim, not an assumption: see the conformance numbers this change was checked against in
 * the task card. If a future measurement finds a case where the RNG choice moves the fitted quad, that is
 * the place to revisit — most likely by porting PCG64's bit generator, not by guessing at a "closer"
 * substitute.
 */
public object QuadFit {

    public const val RESAMPLE_STEP_PX: Double = 3.0
    public const val ITERATIONS: Int = 3
    public const val RANSAC_SAMPLES: Int = 200
    public const val INLIER_FRAC: Double = 0.012
    public const val MIN_SIDE_POINTS: Int = 12
    public const val MIN_SUPPORT_FRAC: Double = 0.30
    public const val ASSIGN_OVERHANG: Double = 0.05
    public const val MIN_IOU_WITH_INIT: Double = 0.6
    public const val MAX_OUTSIDE_FRAC: Double = 0.35
    public const val FRAME_TOL_PX: Double = 3.0
    public const val EXTRAPOLATE_MIN_VISIBLE: Double = 0.5
    public const val EXTRAPOLATE_MAX_VISIBLE: Double = 0.97

    /** Everything `fit_quad_lines` reports alongside the quad. */
    public class Info {
        public var method: String = "none"
        public var refit: List<Boolean> = listOf(false, false, false, false)
        public var clipped: MutableList<Int> = mutableListOf()
        public var extrapolated: Int? = null
        public var visible: Double? = null
        public var reason: String? = null
        public var iouInit: Double? = null
    }

    /**
     * Quad of a page (TL, TR, BR, BL) from line fits to its segmentation contour.
     *
     * [imageShape] (height, width) enables the frame-clipping check and, with [aspectHOverW] (page
     * height/width) and [extrapolate], lets a side that runs out of the photo be pushed out to where the
     * page's own aspect ratio puts it. Returns null only when no quad at all can be made; `info.method` is
     * `"lines"` when the line fit was used, `"polygon"` when the polygon corners were kept instead
     * (`info.reason` says why).
     */
    public fun fitQuadLines(
        contour: List<Pt>,
        imageShape: Pair<Int, Int>?,
        aspectHOverW: Double?,
        extrapolate: Boolean = true,
    ): Pair<List<Pt>?, Info> {
        val info = Info()
        val initRaw = Geometry.extractQuad(contour) ?: return null to info.also { it.method = "none" }
        val init = Geometry.orderPoints(initRaw) ?: return null to info.also { it.method = "none" }
        info.method = "polygon"
        val pts = resample(contour, RESAMPLE_STEP_PX)
        if (pts.size < 4 * MIN_SIDE_POINTS) {
            info.reason = "few points"
            return init to info
        }
        val rng = Random(0)
        val usable = BooleanArray(pts.size) { true }
        if (imageShape != null) {
            val (h, w) = imageShape
            for (i in pts.indices) {
                val p = pts[i]
                usable[i] = !(p.x <= FRAME_TOL_PX || p.x >= w - 1 - FRAME_TOL_PX ||
                    p.y <= FRAME_TOL_PX || p.y >= h - 1 - FRAME_TOL_PX)
            }
        }
        var quad = init.toMutableList()
        var refit = listOf(false, false, false, false)
        for (iter in 0 until ITERATIONS) {
            val a = quad
            val b = listOf(quad[1], quad[2], quad[3], quad[0])
            val d = List(4) { k -> Pt(b[k].x - a[k].x, b[k].y - a[k].y) }
            val length = DoubleArray(4) { k -> hypot(d[k].x, d[k].y) }
            if (length.any { it < 2 }) {
                info.reason = "degenerate side"
                return init to info
            }
            val u = List(4) { k -> Pt(d[k].x / length[k], d[k].y / length[k]) }

            // Assign every resampled point to its nearest side (by perpendicular distance among sides
            // whose projection along the side falls inside [-overhang, 1+overhang]).
            val side = IntArray(pts.size) { -1 }
            for (pi in pts.indices) {
                var bestK = -1
                var bestDist = Double.POSITIVE_INFINITY
                for (k in 0 until 4) {
                    val relX = pts[pi].x - a[k].x
                    val relY = pts[pi].y - a[k].y
                    val proj = (relX * u[k].x + relY * u[k].y) / length[k]
                    if (proj < -ASSIGN_OVERHANG || proj > 1 + ASSIGN_OVERHANG) continue
                    val dist = abs(relX * (-u[k].y) + relY * u[k].x)
                    if (dist < bestDist) {
                        bestDist = dist
                        bestK = k
                    }
                }
                side[pi] = bestK
            }

            val refitFlags = BooleanArray(4)
            val lines = ArrayList<Pair<Pt, Pt>>(4)
            for (k in 0 until 4) {
                val sel = pts.indices.filter { side[it] == k && usable[it] }.map { pts[it] }
                var fit: Triple<Pt, Pt, BooleanArray>? = null
                if (sel.size >= MIN_SIDE_POINTS) {
                    fit = fitLineRansac(sel, max(2.0, INLIER_FRAC * length[k]), rng)
                    if (fit != null) {
                        val (p0, dv, inl) = fit
                        val proj = sel.indices.filter { inl[it] }
                            .map { (sel[it].x - p0.x) * dv.x + (sel[it].y - p0.y) * dv.y }
                        val span = if (proj.isEmpty()) 0.0 else (proj.max() - proj.min())
                        if (span < MIN_SUPPORT_FRAC * length[k]) fit = null
                    }
                }
                refitFlags[k] = fit != null
                lines += if (fit != null) fit.first to fit.second else a[k] to u[k]
            }
            refit = refitFlags.toList()

            val corners = ArrayList<Pt>(4)
            for (k in 0 until 4) {
                val c = intersectLines(lines[(k + 3) % 4], lines[k]) ?: run {
                    info.reason = "parallel sides"
                    return@fitQuadLines init to info
                }
                corners += c
            }
            quad = (Geometry.orderPoints(corners) ?: run {
                info.reason = "parallel sides"
                return init to info
            }).toMutableList()
        }
        info.refit = refit
        if (refit.none { it }) {
            info.reason = "no side fitted"
            return init to info
        }
        if (!isConvex(quad)) {
            info.reason = "not convex"
            return init to info
        }
        val iou = quadIou(quad, init)
        info.iouInit = round3(iou)
        if (iou < MIN_IOU_WITH_INIT) {
            info.reason = "disagrees with polygon"
            return init to info
        }
        if (imageShape != null) {
            val (h, w) = imageShape
            val slack = MAX_OUTSIDE_FRAC * dist(quad[1], quad[0])
            if (quad.any { it.x < -slack || it.x > w + slack || it.y < -slack || it.y > h + slack }) {
                info.reason = "corner outside frame"
                return init to info
            }
            info.clipped = clippedSides(quad, h, w).toMutableList()
            for (k in info.clipped) {
                for (idx in intArrayOf(k, (k + 1) % 4)) {
                    quad[idx] = Pt(quad[idx].x.coerceIn(0.0, (w - 1).toDouble()),
                        quad[idx].y.coerceIn(0.0, (h - 1).toDouble()))
                }
            }
            if (extrapolate && aspectHOverW != null && info.clipped.size == 1) {
                val (newQuad, visible) = extrapolateSide(quad, info.clipped[0], aspectHOverW)
                info.visible = round3(visible)
                if (newQuad != null) {
                    quad = newQuad.toMutableList()
                    info.extrapolated = info.clipped[0]
                }
            }
        }
        info.method = "lines"
        return quad to info
    }

    private fun round3(v: Double): Double = Math.round(v * 1000.0) / 1000.0

    private fun dist(a: Pt, b: Pt): Double = hypot(a.x - b.x, a.y - b.y)

    /** Arc-length resample of a CLOSED contour to uniform spacing `step`. `_resample`. */
    private fun resample(contour: List<Pt>, step: Double): List<Pt> {
        if (contour.size < 3) return contour
        val closed = contour + contour[0]
        val seg = DoubleArray(closed.size - 1) { i -> dist(closed[i], closed[i + 1]) }
        val cum = DoubleArray(seg.size + 1)
        for (i in seg.indices) cum[i + 1] = cum[i] + seg[i]
        val total = cum.last()
        if (total <= 0) return contour
        val n = max((total / step).toInt(), contour.size)
        val out = ArrayList<Pt>(n)
        var segIdx = 0
        for (k in 0 until n) {
            val t = total * k / n
            while (segIdx < seg.size - 1 && cum[segIdx + 1] < t) segIdx++
            val segLen = cum[segIdx + 1] - cum[segIdx]
            val frac = if (segLen > 0) (t - cum[segIdx]) / segLen else 0.0
            val p0 = closed[segIdx]
            val p1 = closed[segIdx + 1]
            out += Pt(p0.x + frac * (p1.x - p0.x), p0.y + frac * (p1.y - p0.y))
        }
        return out
    }

    /**
     * RANSAC line through `pts`, refined by two rounds of least-squares (`cv2.fitLine`, `DIST_L2`).
     * Returns (point on line, unit direction, inlier mask) or null. `_fit_line`.
     */
    private fun fitLineRansac(pts: List<Pt>, thr: Double, rng: Random): Triple<Pt, Pt, BooleanArray>? {
        val n = pts.size
        if (n < 2) return null
        var bestI = -1
        var bestJ = -1
        var bestCount = -1
        // The reference vectorises all 200 pairs at once and argmax's the inlier count; a plain loop over
        // 200 candidates is equivalent and simpler in a language without NumPy's broadcasting.
        repeat(RANSAC_SAMPLES) {
            val i = rng.nextInt(n)
            val j = rng.nextInt(n)
            val dx = pts[j].x - pts[i].x
            val dy = pts[j].y - pts[i].y
            val len = hypot(dx, dy)
            if (len <= 1e-6) return@repeat
            val ux = dx / len
            val uy = dy / len
            val nx = -uy
            val ny = ux
            var count = 0
            for (p in pts) {
                val d = abs((p.x - pts[i].x) * nx + (p.y - pts[i].y) * ny)
                if (d < thr) count++
            }
            if (count > bestCount) {
                bestCount = count
                bestI = i
                bestJ = j
            }
        }
        if (bestI < 0) return null
        var dx = pts[bestJ].x - pts[bestI].x
        var dy = pts[bestJ].y - pts[bestI].y
        var len = hypot(dx, dy)
        var ux = dx / len
        var uy = dy / len
        var inl = BooleanArray(n) {
            abs((pts[it].x - pts[bestI].x) * (-uy) + (pts[it].y - pts[bestI].y) * ux) < thr
        }
        var p0 = pts[bestI]
        var dv = Pt(ux, uy)
        repeat(2) {
            val inlierPts = pts.filterIndexed { idx, _ -> inl[idx] }
            if (inlierPts.size < 2) return null
            val (fp0, fdv) = fitLineL2(inlierPts)
            p0 = fp0
            dv = fdv
            inl = BooleanArray(n) {
                abs((pts[it].x - p0.x) * (-dv.y) + (pts[it].y - p0.y) * dv.x) < thr
            }
        }
        return Triple(p0, dv, inl)
    }

    /** `cv2.fitLine(pts, DIST_L2, 0, 0.01, 0.01)`: point + unit direction of the least-squares line. */
    private fun fitLineL2(pts: List<Pt>): Pair<Pt, Pt> {
        val mat = org.opencv.core.MatOfPoint2f(*pts.map { CvPoint(it.x, it.y) }.toTypedArray())
        val line = Mat()
        try {
            Imgproc.fitLine(mat, line, Imgproc.DIST_L2, 0.0, 0.01, 0.01)
            val v = FloatArray(4)
            line.get(0, 0, v)
            return Pt(v[2].toDouble(), v[3].toDouble()) to Pt(v[0].toDouble(), v[1].toDouble())
        } finally {
            mat.release()
            line.release()
        }
    }

    /** Intersection of two lines given as (point, direction). `_intersect`. */
    private fun intersectLines(a: Pair<Pt, Pt>, b: Pair<Pt, Pt>): Pt? {
        val (p, d) = a
        val (q, e) = b
        val den = d.x * e.y - d.y * e.x
        if (abs(den) < 1e-9) return null
        val t = ((q.x - p.x) * e.y - (q.y - p.y) * e.x) / den
        return Pt(p.x + t * d.x, p.y + t * d.y)
    }

    /**
     * `cv2.isContourConvex(quad.astype(np.float32))` — the reference checks convexity at FLOAT32
     * precision. The Java binding's `isContourConvex` has only ONE overload, `(MatOfPoint)`, which is
     * INTEGER points — there is no float-precision path available at all, unlike the Python binding.
     * Rounded to the nearest pixel here; a real page quad is nowhere near degenerate enough for
     * sub-pixel rounding to flip a convexity verdict, but it is a platform-forced approximation, not an
     * oversight, and belongs in this comment the way every such asymmetry does (CONVENTIONS §6).
     */
    private fun isConvex(quad: List<Pt>): Boolean {
        val mat = org.opencv.core.MatOfPoint(
            *quad.map { CvPoint(Math.round(it.x).toDouble(), Math.round(it.y).toDouble()) }.toTypedArray())
        return try {
            Imgproc.isContourConvex(mat)
        } finally {
            mat.release()
        }
    }

    /** IoU of two convex quads via `cv2.intersectConvexConvex`. `_quad_iou`. */
    private fun quadIou(a: List<Pt>, b: List<Pt>): Double {
        val ao = Geometry.orderPoints(a) ?: return 0.0
        val bo = Geometry.orderPoints(b) ?: return 0.0
        val am = org.opencv.core.MatOfPoint2f(*ao.map { CvPoint(it.x, it.y) }.toTypedArray())
        val bm = org.opencv.core.MatOfPoint2f(*bo.map { CvPoint(it.x, it.y) }.toTypedArray())
        val inter = Mat()
        return try {
            val interArea = Imgproc.intersectConvexConvex(am, bm, inter)
            val union = Contours.contourArea(ao) + Contours.contourArea(bo) - interArea
            if (union > 0) interArea / union else 0.0
        } finally {
            am.release(); bm.release(); inter.release()
        }
    }

    /** Indices of quad sides (0 top, 1 right, 2 bottom, 3 left) lying on the photo frame. `_clipped_sides`. */
    private fun clippedSides(quad: List<Pt>, h: Int, w: Int): List<Int> {
        val out = ArrayList<Int>()
        for (k in 0 until 4) {
            val a = quad[k]
            val b = quad[(k + 1) % 4]
            val onFrame = (abs(a.x) <= FRAME_TOL_PX && abs(b.x) <= FRAME_TOL_PX) ||
                (abs(a.x - (w - 1)) <= FRAME_TOL_PX && abs(b.x - (w - 1)) <= FRAME_TOL_PX) ||
                (abs(a.y) <= FRAME_TOL_PX && abs(b.y) <= FRAME_TOL_PX) ||
                (abs(a.y - (h - 1)) <= FRAME_TOL_PX && abs(b.y - (h - 1)) <= FRAME_TOL_PX)
            if (onFrame) out += k
        }
        return out
    }

    /** Moves a clipped side out to where the page's own aspect ratio puts it. `_extrapolate`. */
    private fun extrapolateSide(quad: List<Pt>, side: Int, aspectHOverW: Double): Pair<List<Pt>?, Double> {
        val k0 = side
        val k1 = (side + 1) % 4
        val o0 = (side + 3) % 4
        val o1 = (side + 2) % 4
        val oppLen = dist(quad[o1], quad[o0])
        if (oppLen < 1) return null to 0.0
        val expect = oppLen * (if (side == 0 || side == 2) aspectHOverW else 1.0 / aspectHOverW)
        val u0 = Pt(quad[k0].x - quad[o0].x, quad[k0].y - quad[o0].y)
        val u1 = Pt(quad[k1].x - quad[o1].x, quad[k1].y - quad[o1].y)
        val l0 = hypot(u0.x, u0.y)
        val l1 = hypot(u1.x, u1.y)
        if (l0 < 1 || l1 < 1) return null to 0.0
        val visible = 0.5 * (l0 + l1) / expect
        if (visible < EXTRAPOLATE_MIN_VISIBLE || visible > EXTRAPOLATE_MAX_VISIBLE) return null to visible
        val out = quad.toMutableList()
        out[k0] = Pt(quad[o0].x + u0.x / l0 * expect, quad[o0].y + u0.y / l0 * expect)
        out[k1] = Pt(quad[o1].x + u1.x / l1 * expect, quad[o1].y + u1.y / l1 * expect)
        return out to visible
    }
}
