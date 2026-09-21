package net.russiandocs.docproc.modules

import net.russiandocs.docproc.imaging.Image
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point as CvPoint
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * Bend correction of a rectified page from the LOCAL tilt of its text lines. Port of
 * `page_registration/line_dewarp.py`.
 *
 * [LineRefine]'s homography makes a FLAT page straight; a booklet page bent near the spine is not flat —
 * its text lines curve, and their local tilt changes with x, which a homography cannot express. This
 * measures the local tilt on a grid of overlapping cells (projection profiles) plus LSD segments, fits a
 * low-order polynomial `t(x,y) = c0 + c1*x + c2*y + c3*x^2 + c4*x*y` by WEIGHTED LINEAR least squares with
 * a small ridge term (genuinely linear in the coefficients — no LM needed here, unlike [LineRefine]) inside
 * a 3-round Cauchy-IRLS loop, and integrates it along x into a vertical displacement map, zero on the
 * page's centre column.
 */
public object LineDewarp {

    public const val GRID: Int = 5
    public const val MIN_CELLS: Int = 8
    public const val MIN_DISP_PX: Double = 3.0
    public const val MAX_DISP_FRAC: Double = 0.04
    public const val MIN_GAIN: Double = 0.25
    public const val IRLS_ITERS: Int = 3
    public const val IRLS_SCALE_DEG: Double = 1.0
    public const val RIDGE: Double = 1e-3

    /** (x, y, tiltDeg, weight, source) — source 0 = LSD horizontal segment, 1 = profile cell. */
    public class CellMeasurement(
        public val x: Double, public val y: Double, public val tiltDeg: Double,
        public val weight: Double, public val source: Int,
    )

    public class Info {
        public var nSeg: Int = 0
        public var nCell: Int = 0
        public var applied: Boolean = false
        public var reason: String? = null
        public var before: Double? = null
        public var after: Double? = null
        public var maxDispPx: Double? = null
        public var coef: DoubleArray? = null
    }

    /** Local tilt evidence: LSD segments (half scale) plus a `GRID x GRID` overlapping-cell profile scan. */
    public fun measureCells(gray: Image, inset: Int): List<CellMeasurement> {
        val h = gray.height
        val w = gray.width
        val rows = ArrayList<CellMeasurement>()
        val half = Mat()
        try {
            Imgproc.resize(gray.mat, half, Size(), 0.5, 0.5, Imgproc.INTER_AREA)
            val segs = LineRefine.segments(half, LineRefine.LSD_MIN_LEN_FRAC * w * 0.5)
            for (s in segs) {
                val full = LineRefine.Segment(s.x1 * 2.0, s.y1 * 2.0, s.x2 * 2.0, s.y2 * 2.0)
                val a = LineRefine.segAngle(full)
                val len = kotlin.math.hypot(full.x2 - full.x1, full.y2 - full.y1)
                val wgt = min(len / (0.1 * w), LineRefine.LSD_MAX_WEIGHT)
                if (abs(a) < LineRefine.ANGLE_TOL_DEG) {
                    rows += CellMeasurement(0.5 * (full.x1 + full.x2), 0.5 * (full.y1 + full.y2), a, wgt, 0)
                }
            }
        } finally {
            half.release()
        }
        val x0 = inset.toDouble(); val y0 = inset.toDouble()
        val x1 = w - inset.toDouble(); val y1 = h - inset.toDouble()
        if (x1 - x0 < 90 || y1 - y0 < 90) return rows
        val cw = (x1 - x0) / 3.0
        val ch = (y1 - y0) / 3.0
        val xs = linspace(x0, x1 - cw, GRID)
        val ys = linspace(y0, y1 - ch, GRID)
        val cells = ArrayList<Pair<Double, Double>>()
        for (yb in ys) for (xb in xs) cells += xb to yb
        val region = gray.mat.submat(y0.toInt(), y1.toInt(), x0.toInt(), x1.toInt())
        try {
            val localCells = cells.map { (xb, yb) -> doubleArrayOf(xb - x0, yb - y0, cw, ch) }
            val tilts = cellTilts(region, localCells)
            for (i in cells.indices) {
                val (xb, yb) = cells[i]
                val (tilt, ratio) = tilts[i]
                if (ratio >= LineRefine.PROFILE_MIN_PEAK) {
                    rows += CellMeasurement(xb + 0.5 * cw, yb + 0.5 * ch, tilt,
                        LineRefine.PROFILE_WEIGHT * min(1.0, ratio - 1.0), 1)
                }
            }
        } finally {
            region.release()
        }
        return rows
    }

    private fun linspace(from: Double, to: Double, n: Int): DoubleArray {
        if (n == 1) return doubleArrayOf(from)
        val step = (to - from) / (n - 1)
        return DoubleArray(n) { from + it * step }
    }

    /**
     * Tilt and peak ratio of the row profile in every cell, rotating the WHOLE region once per angle and
     * slicing cells out via a cumulative sum — 25 cells x ~22 angles of per-cell warps was measured as the
     * slowest step of a page. `_cell_tilts`.
     */
    private fun cellTilts(region: Mat, cells: List<DoubleArray>): List<Pair<Double, Double>> {
        val small = Mat()
        try {
            Imgproc.resize(region, small, Size(), LineRefine.PROFILE_SCALE, LineRefine.PROFILE_SCALE,
                Imgproc.INTER_AREA)
            val binary8 = Mat()
            Imgproc.threshold(small, binary8, 0.0, 255.0, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)
            val h = binary8.rows()
            val w = binary8.cols()
            val n = run {
                val labels = Mat(); val stats = Mat(); val centroids = Mat()
                val count = Imgproc.connectedComponentsWithStats(binary8, labels, stats, centroids, 8)
                if (count > 1) {
                    val limit = LineRefine.BLOB_MAX_FRAC * (cells[0][3] * LineRefine.PROFILE_SCALE)
                    val statBuf = IntArray(5)
                    val bigLabels = HashSet<Int>()
                    for (lbl in 1 until count) {
                        stats.get(lbl, 0, statBuf)
                        if (statBuf[3] > limit) bigLabels += lbl // HEIGHT
                    }
                    if (bigLabels.isNotEmpty()) {
                        val lab = IntArray(h * w); labels.get(0, 0, lab)
                        val bin = ByteArray(h * w); binary8.get(0, 0, bin)
                        for (i in lab.indices) if (lab[i] in bigLabels) bin[i] = 0
                        binary8.put(0, 0, bin)
                    }
                }
                labels.release(); stats.release(); centroids.release()
                count
            }
            val binF = FloatArray(h * w)
            run {
                val bin = ByteArray(h * w); binary8.get(0, 0, bin)
                for (i in bin.indices) binF[i] = if (bin[i].toInt() != 0) 1.0f else 0.0f
            }
            binary8.release()
            val centre = CvPoint(w / 2.0, h / 2.0)
            val boxes = cells.map { (x, y, cwv, chv) ->
                intArrayOf((x * LineRefine.PROFILE_SCALE).toInt(), (y * LineRefine.PROFILE_SCALE).toInt(),
                    max(2, (cwv * LineRefine.PROFILE_SCALE).toInt()), max(2, (chv * LineRefine.PROFILE_SCALE).toInt()))
            }

            fun score(angles: DoubleArray): Array<DoubleArray> {
                val out = Array(angles.size) { DoubleArray(boxes.size) }
                val binMat = Mat(h, w, CvType.CV_32F); binMat.put(0, 0, binF)
                val validMat = Mat(h, w, CvType.CV_32F, org.opencv.core.Scalar(1.0))
                try {
                    for (ai in angles.indices) {
                        val m = Imgproc.getRotationMatrix2D(centre, angles[ai], 1.0)
                        val rot = Mat(); val cnt = Mat()
                        try {
                            Imgproc.warpAffine(binMat, rot, m, Size(w.toDouble(), h.toDouble()),
                                Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, org.opencv.core.Scalar(0.0))
                            Imgproc.warpAffine(validMat, cnt, m, Size(w.toDouble(), h.toDouble()),
                                Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, org.opencv.core.Scalar(0.0))
                            // row-wise cumulative sums along x, so each cell's row sum is a difference of
                            // two lookups instead of a fresh scan — `rot_c`/`cnt_c`.
                            val rotF = FloatArray(h * w); rot.get(0, 0, rotF)
                            val cntF = FloatArray(h * w); cnt.get(0, 0, cntF)
                            val rotC = Array(h) { DoubleArray(w) }
                            val cntC = Array(h) { DoubleArray(w) }
                            for (row in 0 until h) {
                                var rs = 0.0; var cs = 0.0
                                for (col in 0 until w) {
                                    rs += rotF[row * w + col]; cs += cntF[row * w + col]
                                    rotC[row][col] = rs; cntC[row][col] = cs
                                }
                            }
                            for (j in boxes.indices) {
                                val (bx, by, bw, bh) = boxes[j]
                                val x2 = min(bx + bw, w) - 1
                                val yTop = by; val yBot = min(by + bh, h)
                                var cmax = 0.0
                                val ink = DoubleArray(yBot - yTop)
                                val cnt2 = DoubleArray(yBot - yTop)
                                for (row in yTop until yBot) {
                                    val inkV = rotC[row][x2] - (if (bx > 0) rotC[row][bx - 1] else 0.0)
                                    val cntV = cntC[row][x2] - (if (bx > 0) cntC[row][bx - 1] else 0.0)
                                    ink[row - yTop] = inkV
                                    cnt2[row - yTop] = cntV
                                    if (cntV > cmax) cmax = cntV
                                }
                                if (cmax <= 0) continue
                                val prof = ArrayList<Double>()
                                for (row in ink.indices) {
                                    if (cnt2[row] >= 0.6 * cmax) prof += ink[row] / max(cnt2[row], 1.0)
                                }
                                out[ai][j] = if (prof.size > 2) variance(prof) else 0.0
                            }
                        } finally {
                            rot.release(); cnt.release(); m.release()
                        }
                    }
                } finally {
                    binMat.release(); validMat.release()
                }
                return out
            }

            val coarse = score(LineRefine.PROFILE_COARSE)
            val fineAngles = DoubleArray((2.0 / LineRefine.PROFILE_FINE_STEP).toInt() + 1) {
                -1.0 + it * LineRefine.PROFILE_FINE_STEP
            }
            val peaks = IntArray(boxes.size) { -1 }
            for (j in boxes.indices) {
                var ib = 0
                for (i in 1 until coarse.size) if (coarse[i][j] > coarse[ib][j]) ib = i
                peaks[j] = if (ib == 0 || ib == coarse.size - 1 || coarse[ib][j] <= 0) -1 else ib
            }
            val fineCache = HashMap<Int, Array<DoubleArray>>()
            for (ib in peaks.toSet()) {
                if (ib < 0) continue
                val angles = DoubleArray(fineAngles.size) { LineRefine.PROFILE_COARSE[ib] + fineAngles[it] }
                fineCache[ib] = score(angles)
            }
            val results = ArrayList<Pair<Double, Double>>(boxes.size)
            for (j in boxes.indices) {
                val ib = peaks[j]
                if (ib < 0) {
                    results += 0.0 to 0.0
                    continue
                }
                val fs = DoubleArray(fineAngles.size) { fineCache[ib]!![it][j] }
                var jb = 0
                for (i in 1 until fs.size) if (fs[i] > fs[jb]) jb = i
                var best = LineRefine.PROFILE_COARSE[ib] + fineAngles[jb]
                if (jb in 1 until fs.size - 1) {
                    val y0v = fs[jb - 1]; val y1v = fs[jb]; val y2v = fs[jb + 1]
                    val den = y0v - 2 * y1v + y2v
                    if (den < 0) best += LineRefine.PROFILE_FINE_STEP * 0.5 * (y0v - y2v) / den
                }
                var med = 0.0
                run {
                    val col = DoubleArray(coarse.size) { coarse[it][j] }
                    col.sort()
                    med = if (col.size % 2 == 1) col[col.size / 2] else (col[col.size / 2 - 1] + col[col.size / 2]) / 2.0
                }
                results += best to (if (med > 0) fs[jb] / med else 0.0)
            }
            return results
        } finally {
            small.release()
        }
    }

    private operator fun IntArray.component1() = this[0]
    private operator fun IntArray.component2() = this[1]
    private operator fun IntArray.component3() = this[2]
    private operator fun IntArray.component4() = this[3]

    private fun variance(v: List<Double>): Double {
        val mean = v.sum() / v.size
        var s = 0.0
        for (x in v) s += (x - mean) * (x - mean)
        return s / v.size
    }

    private fun basis(xn: Double, yn: Double): DoubleArray =
        doubleArrayOf(1.0, xn, yn, xn * xn, xn * yn)

    /** Robust weighted least squares (IRLS, Cauchy) of the tilt polynomial. Linear — a direct solve, no LM. */
    private fun fitTilt(meas: List<CellMeasurement>, w: Int, h: Int): DoubleArray {
        val n = meas.size
        val xn = DoubleArray(n) { (meas[it].x - w / 2.0) / (w / 2.0) }
        val yn = DoubleArray(n) { (meas[it].y - h / 2.0) / (w / 2.0) }
        val a = Array(n) { basis(xn[it], yn[it]) }
        val t = DoubleArray(n) { meas[it].tiltDeg }
        var weight = DoubleArray(n) { meas[it].weight }
        var c = DoubleArray(5)
        repeat(IRLS_ITERS) {
            // Normal equations: (A^T W A + ridge*I) c = A^T (W t)
            val ata = Array(5) { DoubleArray(5) }
            val atb = DoubleArray(5)
            for (i in 0 until n) {
                val wi = weight[i]
                for (p in 0 until 5) {
                    atb[p] += a[i][p] * wi * t[i]
                    for (q in 0 until 5) ata[p][q] += a[i][p] * wi * a[i][q]
                }
            }
            for (p in 0 until 5) ata[p][p] += RIDGE
            c = solve5(ata, atb)
            val r = DoubleArray(n) { t[it] - dot(a[it], c) }
            weight = DoubleArray(n) { meas[it].weight / (1.0 + (r[it] / IRLS_SCALE_DEG) * (r[it] / IRLS_SCALE_DEG)) }
        }
        return c
    }

    private fun dot(a: DoubleArray, b: DoubleArray): Double {
        var s = 0.0
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    /** Gaussian elimination with partial pivoting for a small (5x5) symmetric positive-definite system. */
    private fun solve5(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val n = b.size
        val m = Array(n) { i -> DoubleArray(n + 1) { j -> if (j < n) a[i][j] else b[i] } }
        for (col in 0 until n) {
            var pivot = col
            for (row in col + 1 until n) if (abs(m[row][col]) > abs(m[pivot][col])) pivot = row
            val tmp = m[col]; m[col] = m[pivot]; m[pivot] = tmp
            for (row in 0 until n) {
                if (row == col) continue
                val factor = m[row][col] / m[col][col]
                for (c in col..n) m[row][c] -= factor * m[col][c]
            }
        }
        return DoubleArray(n) { m[it][n] / m[it][it] }
    }

    /** Vertical displacement map v(x, y), the x-integral of the tilt polynomial, zero on the centre column. */
    public fun displacement(c: DoubleArray, w: Int, h: Int): Mat {
        val gw = max(2, w / 8)
        val gh = max(2, h / 8)
        val xs = DoubleArray(gw) { ((it * (w - 1).toDouble() / (gw - 1)) - w / 2.0) / (w / 2.0) }
        val ys = DoubleArray(gh) { ((it * (h - 1).toDouble() / (gh - 1)) - h / 2.0) / (w / 2.0) }
        val c0 = Math.toRadians(c[0]); val c1 = Math.toRadians(c[1]); val c2 = Math.toRadians(c[2])
        val c3 = Math.toRadians(c[3]); val c4 = Math.toRadians(c[4])
        val grid = Mat(gh, gw, CvType.CV_32F)
        val buf = FloatArray(gw)
        for (gy in 0 until gh) {
            val y = ys[gy]
            for (gx in 0 until gw) {
                val x = xs[gx]
                val v = c0 * x + c1 * x * x / 2 + c2 * x * y + c3 * x * x * x / 3 + c4 * x * x * y / 2
                buf[gx] = (v * (w / 2.0)).toFloat()
            }
            grid.put(gy, 0, buf)
        }
        val out = Mat()
        Imgproc.resize(grid, out, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        grid.release()
        return out
    }

    /**
     * Displacement map (H, W) float32 that unbends the page (`y_src = y + v`), or null with the reason.
     * `dewarp_by_lines`.
     */
    public fun dewarpByLines(gray: Image, inset: Int = 0, minDispPx: Double = MIN_DISP_PX): Pair<Mat?, Info> {
        val h = gray.height
        val w = gray.width
        val info = Info()
        val meas = measureCells(gray, inset)
        val cells = meas.filter { it.source == 1 }
        info.nSeg = meas.count { it.source == 0 }
        info.nCell = cells.size
        if (cells.size < MIN_CELLS) {
            info.reason = "too few cells"
            return null to info
        }
        info.before = round2(LineRefine.wmedian(
            DoubleArray(cells.size) { abs(cells[it].tiltDeg) }, DoubleArray(cells.size) { cells[it].weight }))
        val c = fitTilt(meas, w, h)
        info.coef = DoubleArray(5) { round3(c[it]) }
        val v = displacement(c, w, h)
        val vArr = FloatArray(w * h)
        v.get(0, 0, vArr)
        var vmax = 0.0
        for (x in vArr) if (abs(x.toDouble()) > vmax) vmax = abs(x.toDouble())
        info.maxDispPx = round1(vmax)
        if (vmax < minDispPx) {
            v.release()
            info.reason = "flat enough"
            return null to info
        }
        if (vmax > MAX_DISP_FRAC * h) {
            v.release()
            info.reason = "bend too large"
            return null to info
        }
        val xn = DoubleArray(meas.size) { (meas[it].x - w / 2.0) / (w / 2.0) }
        val yn = DoubleArray(meas.size) { (meas[it].y - h / 2.0) / (w / 2.0) }
        val afterAll = DoubleArray(meas.size) { abs(meas[it].tiltDeg - dot(basis(xn[it], yn[it]), c)) }
        val beforeAll = DoubleArray(meas.size) { abs(meas[it].tiltDeg) }
        val cellIdx = meas.indices.filter { meas[it].source == 1 }
        val afterCells = DoubleArray(cellIdx.size) { afterAll[cellIdx[it]] }
        val cellW = DoubleArray(cellIdx.size) { meas[cellIdx[it]].weight }
        val allW = DoubleArray(meas.size) { meas[it].weight }
        val after = round2(LineRefine.wmedian(afterCells, cellW))
        val beforeAllW = round2(LineRefine.wmedian(beforeAll, allW))
        val afterAllW = round2(LineRefine.wmedian(afterAll, allW))
        info.after = after
        if (after > (1.0 - MIN_GAIN) * info.before!! || afterAllW > beforeAllW) {
            v.release()
            info.reason = "no gain"
            return null to info
        }
        info.applied = true
        return v to info
    }

    private fun round1(v: Double) = Math.round(v * 10.0) / 10.0
    private fun round2(v: Double) = Math.round(v * 100.0) / 100.0
    private fun round3(v: Double) = Math.round(v * 1000.0) / 1000.0

    /** Remaps `page` by `y' = y + v(x, y)`. `apply_dewarp`. */
    public fun applyDewarp(page: Image, v: Mat): Image {
        val h = page.height
        val w = page.width
        val mapX = Mat(h, w, CvType.CV_32F)
        val mapY = Mat(h, w, CvType.CV_32F)
        val vArr = FloatArray(w * h)
        v.get(0, 0, vArr)
        val xBuf = FloatArray(w)
        val yBuf = FloatArray(w)
        for (y in 0 until h) {
            for (x in 0 until w) {
                xBuf[x] = x.toFloat()
                yBuf[x] = y + vArr[y * w + x]
            }
            mapX.put(y, 0, xBuf)
            mapY.put(y, 0, yBuf)
        }
        val out = Mat()
        try {
            Imgproc.remap(page.mat, out, mapX, mapY, Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
            return Image.wrap(out)
        } finally {
            mapX.release(); mapY.release()
        }
    }
}
