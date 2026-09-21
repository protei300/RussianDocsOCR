package net.russiandocs.docproc.modules

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.tensors.LevenbergMarquardt
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point as CvPoint
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

/**
 * Blur-tolerant straightening of a rectified page by its own straight structures. Port of
 * `page_registration/line_refine.py`.
 *
 * A 4-parameter homography model (rotation, shear, two perspective terms) is fitted to LSD line-segment
 * angles and horizontal-band projection-profile tilts with a robust (Cauchy-weighted IRLS) loss. The
 * nonlinear fit itself (`_fit`) uses [LevenbergMarquardt] — the reference calls `scipy.optimize.
 * least_squares`, whose default method is `trf` (trust-region reflective), not Levenberg-Marquardt; this
 * is a DIFFERENT algorithm on the SAME minimisation problem, not a translation of `trf`'s internals. For
 * this residual (smooth, 4 parameters, no bounds, a well-separated minimum near the zero start) both
 * converge to the same optimum — checked by measuring the actual straightened canvas against the
 * reference's, not assumed; see the task card for the numbers this claim rests on.
 */
public object LineRefine {

    public const val LSD_MIN_LEN_FRAC: Double = 0.04
    public const val LSD_MAX_WEIGHT: Double = 3.0
    public const val ANGLE_TOL_DEG: Double = 12.0
    public const val PROFILE_SCALE: Double = 0.5
    public const val PROFILE_FINE_STEP: Double = 0.25
    public const val PROFILE_MIN_PEAK: Double = 1.15
    public const val PROFILE_WEIGHT: Double = 3.0
    public const val MIN_HORIZ_WEIGHT: Double = 4.0
    public const val MAX_ROT_DEG: Double = 5.0
    public const val MAX_CORNER_SHIFT_FRAC: Double = 0.08
    public const val MIN_GAIN: Double = 0.25
    public const val MIN_RESIDUAL_DEG: Double = 0.3
    public val PRIOR_SCALE: DoubleArray = doubleArrayOf(0.03, 0.05, 0.03)
    public const val IRLS_ITERS: Int = 3
    public const val IRLS_SCALE_DEG: Double = 1.0
    public const val VERT_MIN_LEN_FRAC: Double = 0.15
    public const val BLOB_MAX_FRAC: Double = 0.2
    public const val N_BANDS: Int = 5

    /** -6..6 step 1, inclusive — `PROFILE_COARSE`. */
    public val PROFILE_COARSE: DoubleArray = DoubleArray(13) { -6.0 + it }

    /** One row of straightness evidence: `kind` 0=horizontal,1=vertical; `source` 0=LSD,1=band. */
    public class Measurement(
        public val x1: Double, public val y1: Double, public val x2: Double, public val y2: Double,
        public val kind: Int, public val weight: Double, public val source: Int,
    )

    /** What [refineByLines] found. */
    public class Info {
        public var nSeg: Int = 0
        public var nBand: Int = 0
        public var applied: Boolean = false
        public var reason: String? = null
        public var horizWeight: Double? = null
        public var before: Double? = null
        public var after: Double? = null
        public var rotDeg: Double? = null
        public var shear: Double? = null
        public var p1: Double? = null
        public var p2: Double? = null
        public var cornerShiftPx: Double? = null
    }

    private var lsd: org.opencv.imgproc.LineSegmentDetector? = null
    private fun lsd(): org.opencv.imgproc.LineSegmentDetector {
        var l = lsd
        if (l == null) {
            l = Imgproc.createLineSegmentDetector(Imgproc.LSD_REFINE_STD)
            lsd = l
        }
        return l
    }

    internal class Segment(val x1: Double, val y1: Double, val x2: Double, val y2: Double)

    /** LSD segments at least `minLen` long. `_segments`. */
    internal fun segments(gray: Mat, minLen: Double): List<Segment> {
        val lines = Mat()
        try {
            lsd().detect(gray, lines)
            if (lines.empty()) return emptyList()
            val n = lines.rows()
            val buf = FloatArray(4)
            val out = ArrayList<Segment>(n)
            for (i in 0 until n) {
                lines.get(i, 0, buf)
                val len = kotlin.math.hypot((buf[2] - buf[0]).toDouble(), (buf[3] - buf[1]).toDouble())
                if (len >= minLen) {
                    out += Segment(buf[0].toDouble(), buf[1].toDouble(), buf[2].toDouble(), buf[3].toDouble())
                }
            }
            return out
        } finally {
            lines.release()
        }
    }

    /** Segment angle in `(-90, 90]` degrees. `_seg_angle`. */
    internal fun segAngle(s: Segment): Double {
        val ang = Math.toDegrees(atan2(s.y2 - s.y1, s.x2 - s.x1))
        return ((ang + 90.0).mod(180.0)) - 90.0
    }

    /**
     * Tilt (degrees) that makes the row (axis=1) or column (axis=0) projection profile of `region`
     * sharpest, and the peak-to-median ratio. `_profile_tilt`.
     */
    private fun profileTilt(region: Mat, axis: Int): Pair<Double, Double> {
        val small = Mat()
        try {
            if (PROFILE_SCALE != 1.0) {
                Imgproc.resize(region, small, Size(), PROFILE_SCALE, PROFILE_SCALE, Imgproc.INTER_AREA)
            } else {
                region.copyTo(small)
            }
            val binary8 = Mat()
            Imgproc.threshold(small, binary8, 0.0, 255.0,
                Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)
            val h = binary8.rows()
            val w = binary8.cols()
            // Drop ink blobs taller (axis=1) / wider (axis=0) than BLOB_MAX_FRAC of the region: a photo,
            // stamp or thumb would otherwise own the profile.
            val labels = Mat()
            val stats = Mat()
            val centroids = Mat()
            val n = Imgproc.connectedComponentsWithStats(binary8, labels, stats, centroids, 8)
            if (n > 1) {
                val statBuf = IntArray(5)
                val bigLabels = HashSet<Int>()
                for (lbl in 1 until n) {
                    stats.get(lbl, 0, statBuf)
                    val dim = if (axis == 1) statBuf[3] else statBuf[2] // HEIGHT=3, WIDTH=2
                    val limit = if (axis == 1) BLOB_MAX_FRAC * h else BLOB_MAX_FRAC * w
                    if (dim > limit) bigLabels += lbl
                }
                if (bigLabels.isNotEmpty()) {
                    val lab = IntArray(h * w)
                    labels.get(0, 0, lab)
                    val bin = ByteArray(h * w)
                    binary8.get(0, 0, bin)
                    for (i in lab.indices) {
                        if (lab[i] in bigLabels) bin[i] = 0
                    }
                    binary8.put(0, 0, bin)
                }
            }
            labels.release(); stats.release(); centroids.release()

            val binF = FloatArray(h * w)
            run {
                val bin = ByteArray(h * w)
                binary8.get(0, 0, bin)
                for (i in bin.indices) binF[i] = if (bin[i].toInt() != 0) 1.0f else 0.0f
            }
            binary8.release()
            val centre = CvPoint(w / 2.0, h / 2.0)

            fun score(angles: DoubleArray): DoubleArray {
                val out = DoubleArray(angles.size)
                val binMat = Mat(h, w, CvType.CV_32F)
                binMat.put(0, 0, binF)
                val validMat = Mat(h, w, CvType.CV_32F, org.opencv.core.Scalar(1.0))
                try {
                    for (ai in angles.indices) {
                        val m = Imgproc.getRotationMatrix2D(centre, angles[ai], 1.0)
                        val rot = Mat()
                        val cnt = Mat()
                        try {
                            Imgproc.warpAffine(binMat, rot, m, Size(w.toDouble(), h.toDouble()),
                                Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, org.opencv.core.Scalar(0.0))
                            Imgproc.warpAffine(validMat, cnt, m, Size(w.toDouble(), h.toDouble()),
                                Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, org.opencv.core.Scalar(0.0))
                            val rotF = FloatArray(h * w)
                            val cntF = FloatArray(h * w)
                            rot.get(0, 0, rotF)
                            cnt.get(0, 0, cntF)
                            val lines = if (axis == 1) h else w
                            val perLine = if (axis == 1) w else h
                            val ink = DoubleArray(lines)
                            val count = DoubleArray(lines)
                            for (li in 0 until lines) {
                                var si = 0.0
                                var sc = 0.0
                                for (pj in 0 until perLine) {
                                    val idx = if (axis == 1) li * w + pj else pj * w + li
                                    si += rotF[idx]
                                    sc += cntF[idx]
                                }
                                ink[li] = si
                                count[li] = sc
                            }
                            val cmax = count.maxOrNull() ?: 0.0
                            val prof = ArrayList<Double>()
                            for (li in 0 until lines) {
                                if (count[li] >= 0.6 * cmax) {
                                    prof += ink[li] / max(count[li], 1.0)
                                }
                            }
                            out[ai] = if (prof.size > 2) variance(prof) else 0.0
                        } finally {
                            rot.release(); cnt.release(); m.release()
                        }
                    }
                } finally {
                    binMat.release(); validMat.release()
                }
                return out
            }

            val coarse = score(PROFILE_COARSE)
            val ib = argmax(coarse)
            if (ib == 0 || ib == PROFILE_COARSE.size - 1 || coarse[ib] <= 0) return 0.0 to 0.0
            val fine = fineRange(PROFILE_COARSE[ib] - 1.0, PROFILE_COARSE[ib] + 1.001, PROFILE_FINE_STEP)
            val fs = score(fine)
            val jb = argmax(fs)
            var best = fine[jb]
            if (jb in 1 until fine.size - 1) {
                val y0 = fs[jb - 1]; val y1 = fs[jb]; val y2 = fs[jb + 1]
                val den = y0 - 2 * y1 + y2
                if (den < 0) best += PROFILE_FINE_STEP * 0.5 * (y0 - y2) / den
            }
            val med = median(coarse)
            val ratio = if (med > 0) fs[jb] / med else 0.0
            return best to ratio
        } finally {
            small.release()
        }
    }

    private fun fineRange(from: Double, to: Double, step: Double): DoubleArray {
        val n = ((to - from) / step).toInt() + 1
        return DoubleArray(n) { from + it * step }
    }

    private fun argmax(v: DoubleArray): Int {
        var best = 0
        for (i in 1 until v.size) if (v[i] > v[best]) best = i
        return best
    }

    private fun variance(v: List<Double>): Double {
        val mean = v.sum() / v.size
        var s = 0.0
        for (x in v) s += (x - mean) * (x - mean)
        return s / v.size
    }

    private fun median(v: DoubleArray): Double {
        val sorted = v.sorted()
        val n = sorted.size
        return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }

    /** Straightness evidence of a rectified page. `measure`. */
    public fun measure(gray: Image, inset: Int): List<Measurement> {
        val h = gray.height
        val w = gray.width
        val rows = ArrayList<Measurement>()
        val segs = segments(gray.mat, LSD_MIN_LEN_FRAC * w)
        for (s in segs) {
            val a = segAngle(s)
            val len = kotlin.math.hypot(s.x2 - s.x1, s.y2 - s.y1)
            val wgt = min(len / (0.1 * w), LSD_MAX_WEIGHT)
            if (abs(a) < ANGLE_TOL_DEG) {
                rows += Measurement(s.x1, s.y1, s.x2, s.y2, 0, wgt, 0)
            } else if (abs(abs(a) - 90.0) < ANGLE_TOL_DEG && len >= VERT_MIN_LEN_FRAC * h) {
                rows += Measurement(s.x1, s.y1, s.x2, s.y2, 1, wgt, 0)
            }
        }
        val x0 = inset.toDouble(); val y0 = inset.toDouble()
        val x1 = w - inset.toDouble(); val y1 = h - inset.toDouble()
        if (x1 - x0 < 60 || y1 - y0 < 60) return rows
        val bh = (y1 - y0) / 3.0
        val bandYs = linspace(y0, y1 - bh, N_BANDS)
        for (yb in bandYs) {
            val band = gray.mat.submat(yb.toInt(), (yb + bh).toInt(), x0.toInt(), x1.toInt())
            try {
                val (tilt, ratio) = profileTilt(band, 1)
                if (ratio >= PROFILE_MIN_PEAK) {
                    val yc = yb + 0.5 * bh
                    val half = 0.5 * (x1 - x0)
                    val dy = tan(Math.toRadians(tilt)) * half
                    rows += Measurement(x0, yc - dy, x1, yc + dy, 0, PROFILE_WEIGHT * min(1.0, ratio - 1.0), 1)
                }
            } finally {
                band.release()
            }
        }
        return rows
    }

    private fun linspace(from: Double, to: Double, n: Int): DoubleArray {
        if (n == 1) return doubleArrayOf(from)
        val step = (to - from) / (n - 1)
        return DoubleArray(n) { from + it * step }
    }

    /** Pixel homography for (rotation rad, shear, p1, p2) about the page centre. `_model`. */
    private fun model(params: DoubleArray, w: Int, h: Int): Array<DoubleArray> {
        val (th, s, p1, p2) = params
        val k = 2.0 / w
        // T, Tinv, R, S, P as 3x3; compute Tinv * P * R * S * T directly.
        val t = arrayOf(
            doubleArrayOf(k, 0.0, -1.0),
            doubleArrayOf(0.0, k, -h.toDouble() / w),
            doubleArrayOf(0.0, 0.0, 1.0),
        )
        val tInv = invert3x3(t)
        val r = arrayOf(
            doubleArrayOf(cos(th), -sin(th), 0.0),
            doubleArrayOf(sin(th), cos(th), 0.0),
            doubleArrayOf(0.0, 0.0, 1.0),
        )
        val sh = arrayOf(
            doubleArrayOf(1.0, s, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 1.0),
        )
        val p = arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1.0, 0.0),
            doubleArrayOf(p1, p2, 1.0),
        )
        return matMul(matMul(matMul(tInv, p), matMul(r, sh)), t)
    }

    private operator fun DoubleArray.component1() = this[0]
    private operator fun DoubleArray.component2() = this[1]
    private operator fun DoubleArray.component3() = this[2]
    private operator fun DoubleArray.component4() = this[3]

    private fun matMul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val out = Array(3) { DoubleArray(3) }
        for (i in 0 until 3) for (j in 0 until 3) {
            var s = 0.0
            for (k in 0 until 3) s += a[i][k] * b[k][j]
            out[i][j] = s
        }
        return out
    }

    private fun invert3x3(m: Array<DoubleArray>): Array<DoubleArray> {
        val a = m[0][0]; val b = m[0][1]; val c = m[0][2]
        val d = m[1][0]; val e = m[1][1]; val f = m[1][2]
        val g = m[2][0]; val h = m[2][1]; val i = m[2][2]
        val det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        val inv = 1.0 / det
        return arrayOf(
            doubleArrayOf((e * i - f * h) * inv, (c * h - b * i) * inv, (b * f - c * e) * inv),
            doubleArrayOf((f * g - d * i) * inv, (a * i - c * g) * inv, (c * d - a * f) * inv),
            doubleArrayOf((d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv),
        )
    }

    private fun applyH(hM: Array<DoubleArray>, x: Double, y: Double): Pair<Double, Double> {
        val wv = hM[2][0] * x + hM[2][1] * y + hM[2][2]
        val xv = (hM[0][0] * x + hM[0][1] * y + hM[0][2]) / wv
        val yv = (hM[1][0] * x + hM[1][1] * y + hM[1][2]) / wv
        return xv to yv
    }

    /** Angle deviation (deg) of every measurement's segment after applying homography `hM`. `_angles_after`. */
    private fun anglesAfter(hM: Array<DoubleArray>, meas: List<Measurement>): DoubleArray {
        return DoubleArray(meas.size) { i ->
            val m = meas[i]
            val (x0, y0) = applyH(hM, m.x1, m.y1)
            val (x1, y1) = applyH(hM, m.x2, m.y2)
            var ang = Math.toDegrees(atan2(y1 - y0, x1 - x0))
            ang = ((ang + 90.0).mod(180.0)) - 90.0
            if (m.kind == 0) ang else abs(ang) - 90.0
        }
    }

    /** `_residuals`: sqrt-weighted angle deviations, plus the quadratic prior on shear/p1/p2. */
    private fun residuals(params: DoubleArray, meas: List<Measurement>, w: Int, h: Int, wgt: DoubleArray): DoubleArray {
        val dev = anglesAfter(model(params, w, h), meas)
        val out = DoubleArray(dev.size + 3)
        for (i in dev.indices) out[i] = dev[i] * sqrt(wgt[i])
        for (i in 0 until 3) out[dev.size + i] = params[i + 1] / PRIOR_SCALE[i]
        return out
    }

    /** Robust IRLS (Cauchy weights) fit of the 4 parameters, via [LevenbergMarquardt]. `_fit`. */
    private fun fit(meas: List<Measurement>, w: Int, h: Int): DoubleArray {
        var x = doubleArrayOf(0.0, 0.0, 0.0, 0.0)
        var wgt = DoubleArray(meas.size) { meas[it].weight }
        repeat(IRLS_ITERS) {
            val wgtCopy = wgt
            val sol = LevenbergMarquardt.solve(x, { p -> residuals(p, meas, w, h, wgtCopy) }, maxIterations = 100)
            x = sol.params
            val r = anglesAfter(model(x, w, h), meas)
            wgt = DoubleArray(meas.size) { meas[it].weight / (1.0 + (r[it] / IRLS_SCALE_DEG) * (r[it] / IRLS_SCALE_DEG)) }
        }
        return x
    }

    internal fun wmedian(values: DoubleArray, weights: DoubleArray): Double {
        if (values.isEmpty()) return 0.0
        val order = values.indices.sortedBy { values[it] }
        val v = DoubleArray(order.size) { values[order[it]] }
        val w = DoubleArray(order.size) { weights[order[it]] }
        val c = DoubleArray(w.size)
        var running = 0.0
        for (i in w.indices) {
            running += w[i]
            c[i] = running
        }
        val half = 0.5 * c.last()
        var idx = c.indexOfFirst { it >= half }
        if (idx < 0) idx = c.size - 1
        return v[idx]
    }

    /**
     * Homography (pixel, src->dst) that straightens a rectified page by its own lines, or null when there
     * is not enough evidence or the correction is not warranted. `refine_by_lines`.
     */
    public fun refineByLines(gray: Image, inset: Int = 0, minResidualDeg: Double = MIN_RESIDUAL_DEG): Pair<Array<DoubleArray>?, Info> {
        val h = gray.height
        val w = gray.width
        val info = Info()
        val meas = measure(gray, inset)
        info.nSeg = meas.count { it.source == 0 }
        info.nBand = meas.count { it.source == 1 }
        if (meas.isEmpty()) {
            info.reason = "no evidence"
            return null to info
        }
        val horizW = meas.filter { it.kind == 0 }.sumOf { it.weight }
        info.horizWeight = round2(horizW)
        if (horizW < MIN_HORIZ_WEIGHT) {
            info.reason = "too little horizontal evidence"
            return null to info
        }
        val weights = DoubleArray(meas.size) { meas[it].weight }
        val before = anglesAfter(identity3(), meas).map { abs(it) }.toDoubleArray()
        info.before = round2(wmedian(before, weights))
        val x = fit(meas, w, h)
        val hM = model(x, w, h)
        val after = anglesAfter(hM, meas).map { abs(it) }.toDoubleArray()
        info.after = round2(wmedian(after, weights))
        info.rotDeg = round2(Math.toDegrees(x[0]))
        info.shear = round4(x[1])
        info.p1 = round4(x[2])
        info.p2 = round4(x[3])
        if (info.before!! < minResidualDeg) {
            info.reason = "already straight"
            return null to info
        }
        if (info.after!! > (1.0 - MIN_GAIN) * info.before!!) {
            info.reason = "no gain"
            return null to info
        }
        if (abs(info.rotDeg!!) > MAX_ROT_DEG) {
            info.reason = "rotation too large"
            return null to info
        }
        val corners = arrayOf(
            doubleArrayOf(inset.toDouble(), inset.toDouble()),
            doubleArrayOf((w - inset).toDouble(), inset.toDouble()),
            doubleArrayOf((w - inset).toDouble(), (h - inset).toDouble()),
            doubleArrayOf(inset.toDouble(), (h - inset).toDouble()),
        )
        var maxShift = 0.0
        for (c in corners) {
            val (mx, my) = applyH(hM, c[0], c[1])
            val d = kotlin.math.hypot(mx - c[0], my - c[1])
            if (d > maxShift) maxShift = d
        }
        info.cornerShiftPx = round1(maxShift)
        if (maxShift > MAX_CORNER_SHIFT_FRAC * w) {
            info.reason = "correction too large"
            return null to info
        }
        info.applied = true
        return hM to info
    }

    private fun identity3(): Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0), doubleArrayOf(0.0, 1.0, 0.0), doubleArrayOf(0.0, 0.0, 1.0))

    private fun round1(v: Double) = Math.round(v * 10.0) / 10.0
    private fun round2(v: Double) = Math.round(v * 100.0) / 100.0
    private fun round4(v: Double) = Math.round(v * 10000.0) / 10000.0

    /** Warps `page` by homography `hM` (photo pixel, src->dst). `apply_refinement`. */
    public fun applyRefinement(page: Image, hM: Array<DoubleArray>): Image {
        val hMat = Mat(3, 3, CvType.CV_64F)
        for (i in 0 until 3) hMat.put(i, 0, *hM[i])
        val out = Mat()
        try {
            Imgproc.warpPerspective(page.mat, out, hMat,
                Size(page.width.toDouble(), page.height.toDouble()),
                Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
            return Image.wrap(out)
        } finally {
            hMat.release()
        }
    }
}
