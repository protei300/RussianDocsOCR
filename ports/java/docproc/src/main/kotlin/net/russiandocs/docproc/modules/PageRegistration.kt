package net.russiandocs.docproc.modules

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.imaging.Pt
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point as CvPoint
import org.opencv.core.Size
import org.opencv.features2d.BFMatcher
import org.opencv.features2d.SIFT
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Template-based page registration for booklet documents (internal passport). Port of
 * `page_registration/page_registration.py`.
 *
 * Aligns the printed BLANK of a page to a canonical, person-free template (SIFT features + a MAGSAC
 * homography), refines the match in the canonical frame, and — only on success — replaces the Borders
 * warp for that page. `use_ecc` is FALSE by the reference's own default (`Pipeline`'s `PageRegistrar()`
 * call site never turns it on), so the dense ECC polish (`_ecc`, `cv2.findTransformECC`) is intentionally
 * NOT ported: porting an unused code path would be exactly the kind of dead weight `MAPPING.md` warns
 * against carrying into a new language.
 */
public class PageTemplate(
    public val name: String,
    imagePath: String,
    maskPath: String,
    sift: SIFT,
) : AutoCloseable {
    // `PageTemplate.__init__` (page_registration.py): reads COLOR then converts BGR->gray via
    // cvtColor - NOT a direct grayscale decode. The two are not bit-identical (a direct
    // IMREAD_GRAYSCALE decode can round differently than IMREAD_COLOR + cvtColor's fixed luma
    // formula), and on these template prints the gap was enough to shift SIFT's contrast-threshold
    // boundary and change which keypoints get kept (measured: 521/515 kp here vs the reference's
    // 519/506 on the same two page2 PNGs, while the PHOTO side - already loaded via cvtColor - was
    // bit-identical, 865 kp on both). Match the reference's path exactly.
    public val gray: Mat = run {
        val bgr = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_COLOR)
            .also { if (it.empty()) throw java.io.FileNotFoundException(imagePath) }
        val g = Mat()
        Imgproc.cvtColor(bgr, g, Imgproc.COLOR_BGR2GRAY)
        bgr.release()
        g
    }
    public val height: Int = gray.rows()
    public val width: Int = gray.cols()
    public val mask: Mat
    public val keypoints: MatOfKeyPoint = MatOfKeyPoint()
    public val descriptors: Mat = Mat()
    public val pts: Array<Pt>
    public val corners: Array<Pt> = arrayOf(
        Pt(0.0, 0.0), Pt(width.toDouble(), 0.0),
        Pt(width.toDouble(), height.toDouble()), Pt(0.0, height.toDouble()))

    init {
        val rawMask = Imgcodecs.imread(maskPath, Imgcodecs.IMREAD_GRAYSCALE)
            .also { if (it.empty()) throw java.io.FileNotFoundException(maskPath) }
        val mask255 = Mat()
        Imgproc.threshold(rawMask, mask255, 127.0, 255.0, Imgproc.THRESH_BINARY)
        rawMask.release()
        mask = mask255
        sift.detectAndCompute(gray, mask255, keypoints, descriptors)
        val kpArray = keypoints.toArray()
        pts = Array(kpArray.size) { Pt(kpArray[it].pt.x, kpArray[it].pt.y) }
    }

    /** Page corners mapped back into the photo. `quad_in_image`. */
    public fun quadInImage(hImgToPage: Array<DoubleArray>): Array<Pt> {
        val inv = Homography.invert(hImgToPage)
        return Array(4) { Homography.apply(inv, corners[it]) }
    }

    override fun close() {
        gray.release(); mask.release(); keypoints.release(); descriptors.release()
    }
}

/** Result for one page. `H` maps photo pixels to canonical page pixels. */
public class PageRegistration(
    public var name: String,
    public val H: Array<DoubleArray>? = null,
    public val inliers: Int = 0,
    public var method: String = "none",
    public val quad: Array<Pt>? = null,
    public var ref: Int? = null,
) {
    public val ok: Boolean get() = H != null
}

/** 3x3 homography helpers shared by this file. */
internal object Homography {
    fun identity(): Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0), doubleArrayOf(0.0, 1.0, 0.0), doubleArrayOf(0.0, 0.0, 1.0))

    fun apply(h: Array<DoubleArray>, p: Pt): Pt {
        val w = h[2][0] * p.x + h[2][1] * p.y + h[2][2]
        return Pt((h[0][0] * p.x + h[0][1] * p.y + h[0][2]) / w, (h[1][0] * p.x + h[1][1] * p.y + h[1][2]) / w)
    }

    fun mul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val out = Array(3) { DoubleArray(3) }
        for (i in 0 until 3) for (j in 0 until 3) {
            var s = 0.0
            for (k in 0 until 3) s += a[i][k] * b[k][j]
            out[i][j] = s
        }
        return out
    }

    fun invert(m: Array<DoubleArray>): Array<DoubleArray> {
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

    fun fromMat(m: Mat): Array<DoubleArray> {
        val out = Array(3) { DoubleArray(3) }
        val buf = DoubleArray(3)
        for (i in 0 until 3) { m.get(i, 0, buf); out[i] = buf.copyOf() }
        return out
    }

    fun toMat(h: Array<DoubleArray>): Mat {
        val m = Mat(3, 3, CvType.CV_64F)
        for (i in 0 until 3) m.put(i, 0, *h[i])
        return m
    }

    fun translateY(dy: Double): Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0), doubleArrayOf(0.0, 1.0, -dy), doubleArrayOf(0.0, 0.0, 1.0))
}

/**
 * Registers the pages of one document type against its templates. Port of `PageRegistrar`.
 *
 * Templates are read directly from `document_processing/pipeline_modules/page_registration/templates/`
 * (via [ModelPaths][net.russiandocs.docproc.config.ModelPaths]'s repo-root resolution) rather than
 * copied into `ports/java` — the same "single source of truth" the model weights already follow, and
 * confirmed by checking: neither the Go nor the .NET port carries its own copy either.
 */
public class PageRegistrar(
    docType: String = "INTPASSPORT",
    templatesDir: String? = null,
    nfeatures: Int = 6000,
    public val lineQuads: Boolean = true,
    public val lineRefine: Boolean = true,
    public val lineDewarp: Boolean = true,
) : AutoCloseable {

    public companion object {
        public const val MIN_COARSE_INLIERS: Int = 8
        public const val MIN_REFINED_INLIERS: Int = 18
        public const val MIN_REFINED_INLIERS_CHAIN: Int = 10
        public const val MIN_SPREAD_PX_CHAIN: Double = 140.0
        public const val MIN_SPREAD_PX: Double = 120.0
        public const val RATIO_TEST: Double = 0.8
        public const val COARSE_REPROJ_PX: Double = 4.0
        public val REFINE_REPROJ_PX: DoubleArray = doubleArrayOf(3.0, 2.0)
        public val REFINE_RADIUS_PX: DoubleArray = doubleArrayOf(45.0, 15.0)
        public val CHAIN_RADIUS_PX: DoubleArray = doubleArrayOf(90.0, 15.0)
        public const val CHAIN_GAP_FRAC: Double = 0.03
        public const val QUAD_DILATE_FRAC: Double = 0.10
        public const val MAX_COARSE_CANDIDATES: Int = 2
        public const val GOOD_COARSE_INLIERS: Int = 20
        public const val PAGE_MARGIN_FRAC: Double = 0.03
    }

    private val doc = docType.uppercase()
    private val sift: SIFT = SIFT.create(nfeatures)
    private val matcher: BFMatcher = BFMatcher(Core.NORM_L2)
    public val pages: List<Pair<String, List<PageTemplate>>>
    public val pageW: Int
    public val pageH: Int
    public val margin: Int
    public val outW: Int
    public val outH: Int

    init {
        val root = net.russiandocs.docproc.config.ModelPaths.root()
        val tdir = templatesDir
            ?: File(File(root, "document_processing"), "pipeline_modules/page_registration/templates").path
        val metaFile = File(tdir, "${doc.lowercase()}.json")
        val meta = Json.parseToJsonElement(metaFile.readText(Charsets.UTF_8)).jsonObject
        val loaded = ArrayList<Pair<String, List<PageTemplate>>>()
        for (pElem in meta["pages"]!!.jsonArray) {
            val p = pElem.jsonObject
            val name = p["name"]!!.jsonPrimitive.content
            val refsJson = p["refs"]?.jsonArray
            val refs = if (refsJson != null && refsJson.isNotEmpty()) {
                refsJson.map { r ->
                    val ro = r.jsonObject
                    File(tdir, ro["image"]!!.jsonPrimitive.content).path to
                        File(tdir, ro["mask"]!!.jsonPrimitive.content).path
                }
            } else {
                listOf(File(tdir, p["image"]!!.jsonPrimitive.content).path to
                    File(tdir, p["mask"]!!.jsonPrimitive.content).path)
            }
            loaded += name to refs.map { (img, mask) -> PageTemplate(name, img, mask, sift) }
        }
        pages = loaded
        val first = pages[0].second[0]
        pageW = first.width
        pageH = first.height
        margin = Math.round(PAGE_MARGIN_FRAC * pageW).toInt()
        outW = pageW + 2 * margin
        outH = pageH + 2 * margin
    }

    public val pageNames: List<String> get() = pages.map { it.first }

    // ------------------------------------------------------------------ matching

    /** Template -> photo matches, MAGSAC homography (photo -> page). `_match`. */
    private fun match(
        tpl: PageTemplate, kp: Array<org.opencv.core.KeyPoint>, desc: Mat?, reproj: Double,
        radius: Double? = null, prior: Array<DoubleArray>? = null,
    ): Triple<Array<DoubleArray>?, Int, List<Pt>?> {
        if (desc == null || desc.rows() < 8) {
            return Triple(null, 0, null)
        }
        val knn = ArrayList<MatOfDMatch>()
        matcher.knnMatch(tpl.descriptors, desc, knn, 2)
        val good = ArrayList<DMatch>()
        for (m in knn) {
            val arr = m.toArray()
            if (arr.size == 2 && arr[0].distance < RATIO_TEST * arr[1].distance) good += arr[0]
        }
        knn.forEach { it.release() }
        if (good.size < 8) return Triple(null, good.size, null)
        var src = Array(good.size) { Pt(kp[good[it].trainIdx].pt.x, kp[good[it].trainIdx].pt.y) }
        var dst = Array(good.size) { tpl.pts[good[it].queryIdx] }
        if (prior != null && radius != null) {
            val pred = Array(src.size) { Homography.apply(prior, src[it]) }
            var disp = Array(src.size) { Pt(pred[it].x - dst[it].x, pred[it].y - dst[it].y) }
            val rough = disp.indices.filter { hypot(disp[it].x, disp[it].y) < 2.5 * radius }
            if (rough.size >= 6) {
                val mx = rough.sumOf { disp[it].x } / rough.size
                val my = rough.sumOf { disp[it].y } / rough.size
                disp = Array(disp.size) { Pt(disp[it].x - mx, disp[it].y - my) }
            }
            val keep = disp.indices.filter { hypot(disp[it].x, disp[it].y) < radius }
            if (keep.size < 8) return Triple(null, keep.size, null)
            src = Array(keep.size) { src[keep[it]] }
            dst = Array(keep.size) { dst[keep[it]] }
        }
        val srcMat = MatOfPoint2f(*src.map { CvPoint(it.x, it.y) }.toTypedArray())
        val dstMat = MatOfPoint2f(*dst.map { CvPoint(it.x, it.y) }.toTypedArray())
        val maskMat = Mat()
        try {
            val hMat = Calib3d.findHomography(srcMat, dstMat, Calib3d.USAC_MAGSAC, reproj, maskMat, 10000, 0.999)
            if (hMat == null || hMat.empty() || maskMat.empty()) {
                return Triple(null, 0, null)
            }
            val maskArr = ByteArray(maskMat.rows())
            maskMat.get(0, 0, maskArr)
            var inliers = 0
            val inlierPts = ArrayList<Pt>()
            for (i in maskArr.indices) if (maskArr[i].toInt() != 0) {
                inliers++
                inlierPts += dst[i]
            }
            val hArr = Homography.fromMat(hMat)
            hMat.release()
            return Triple(hArr, inliers, inlierPts)
        } finally {
            srcMat.release(); dstMat.release(); maskMat.release()
        }
    }

    /** Keypoints/descriptors restricted to inside a (dilated) quad. `_features_in_quad`. */
    private fun featuresInQuad(
        kp: Array<org.opencv.core.KeyPoint>, desc: Mat, quad: Array<Pt>, h: Int, w: Int,
    ): Pair<Array<org.opencv.core.KeyPoint>, Mat?> {
        val cx = quad.sumOf { it.x } / 4; val cy = quad.sumOf { it.y } / 4
        val grown = quad.map { CvPoint(cx + (it.x - cx) * (1 + 2 * QUAD_DILATE_FRAC),
            cy + (it.y - cy) * (1 + 2 * QUAD_DILATE_FRAC)) }
        val poly = org.opencv.core.MatOfPoint(*grown.map {
            CvPoint(Math.round(it.x).toDouble(), Math.round(it.y).toDouble()) }.toTypedArray())
        val m = Mat.zeros(h, w, CvType.CV_8U)
        try {
            Imgproc.fillConvexPoly(m, poly, org.opencv.core.Scalar(1.0))
            val idx = ArrayList<Int>()
            for (i in kp.indices) {
                val px = kp[i].pt.x.toInt().coerceIn(0, w - 1)
                val py = kp[i].pt.y.toInt().coerceIn(0, h - 1)
                if (m.get(py, px)[0] != 0.0) idx += i
            }
            val outKp = Array(idx.size) { kp[idx[it]] }
            val outDesc = if (idx.isEmpty()) null else Mat(idx.size, desc.cols(), desc.type()).also { d ->
                for (r in idx.indices) desc.row(idx[r]).copyTo(d.row(r))
            }
            return outKp to outDesc
        } finally {
            m.release(); poly.release()
        }
    }

    // ---------------------------------------------------------------- refinement

    private fun spreadOk(pts: List<Pt>?, minPx: Double): Boolean {
        if (pts == null || pts.size < 4) return false
        val spanX = pts.maxOf { it.x } - pts.minOf { it.x }
        val spanY = pts.maxOf { it.y } - pts.minOf { it.y }
        return spanX >= minPx && spanY >= minPx
    }

    /** Guided re-match in the canonical frame. `_refine`. Returns (H, inliers, points). */
    private fun refine(
        gray: Mat, tpl: PageTemplate, hIn: Array<DoubleArray>, radii: DoubleArray, minInliers: Int,
    ): Triple<Array<DoubleArray>, Int, List<Pt>?> {
        val page = Mat()
        try {
            Imgproc.warpPerspective(gray, page, Homography.toMat(hIn), Size(tpl.width.toDouble(), tpl.height.toDouble()),
                Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
            val kpMat = MatOfKeyPoint()
            val descMat = Mat()
            sift.detectAndCompute(page, Mat(), kpMat, descMat)
            val kp = kpMat.toArray()
            var prior = Homography.identity()
            var inl = 0
            var pts: List<Pt>? = null
            for (i in radii.indices) {
                val (hd, inlP, ptsP) = match(tpl, kp, descMat, REFINE_REPROJ_PX[i], radii[i], prior)
                if (hd == null || inlP < minInliers) {
                    kpMat.release(); descMat.release()
                    return Triple(hIn, 0, null)
                }
                prior = hd; inl = inlP; pts = ptsP
            }
            kpMat.release(); descMat.release()
            return Triple(Homography.mul(prior, hIn), inl, pts)
        } finally {
            page.release()
        }
    }

    // ---------------------------------------------------------------- validation

    private fun saneQuad(quad: Array<Pt>?, h: Int, w: Int, refArea: Double?): Boolean {
        if (quad == null || quad.any { it.x.isNaN() || it.y.isNaN() || it.x.isInfinite() || it.y.isInfinite() }) return false
        if (!isConvexF(quad)) return false
        val area = polygonArea(quad)
        if (area < 0.02 * h * w || area > 1.5 * h * w) return false
        val q = orderPoints(quad)
        val wt = 0.5 * (dist(q[1], q[0]) + dist(q[2], q[3]))
        val ht = 0.5 * (dist(q[3], q[0]) + dist(q[2], q[1]))
        if (ht < 1e-3) return false
        val ratio = (wt / ht) / (pageW.toDouble() / pageH)
        if (!(ratio > 0.6 && ratio < 1.7)) return false
        if (refArea != null && refArea > 0) {
            val r = area / refArea
            if (!(r > 0.4 && r < 2.5)) return false
        }
        return true
    }

    private fun accept(
        gray: Mat, tpl: PageTemplate, hIn: Array<DoubleArray>, radii: DoubleArray, refArea: Double?, chain: Boolean,
    ): PageRegistration? {
        val h = gray.rows(); val w = gray.cols()
        val quad0 = tpl.quadInImage(hIn)
        if (!saneQuad(quad0, h, w, refArea)) return null
        val minInl = if (chain) MIN_REFINED_INLIERS_CHAIN else MIN_REFINED_INLIERS
        val minSpread = if (chain) MIN_SPREAD_PX_CHAIN else MIN_SPREAD_PX
        val (hOut, inl, pts) = refine(gray, tpl, hIn, radii, minInl)
        if (inl < minInl || !spreadOk(pts, minSpread)) return null
        val quad = tpl.quadInImage(hOut)
        if (!saneQuad(quad, h, w, refArea)) return null
        return PageRegistration(tpl.name, hOut, inl, "x", quad)
    }

    // ------------------------------------------------------------------- driver

    /** Registers every template page in `imgRgb` (upright RGB photo, [Image]). `register`. */
    public fun register(imgRgb: Image, quads: List<Array<Pt>>?): List<PageRegistration> {
        val gray = Mat()
        try {
            Imgproc.cvtColor(imgRgb.mat, gray, Imgproc.COLOR_RGB2GRAY)
            val h = gray.rows(); val w = gray.cols()
            val kpMat = MatOfKeyPoint()
            val descMat = Mat()
            sift.detectAndCompute(gray, Mat(), kpMat, descMat)
            val kp = kpMat.toArray()
            val q = quads ?: emptyList()
            val quadFeats = q.map { featuresInQuad(kp, descMat, it, h, w) }

            val results = MutableList(pages.size) { PageRegistration(pages[it].first) }
            val taken = HashSet<Int>()
            val pending = (pages.indices).toMutableList()

            for (round in 0..1) {
                if (round == 1 && results.none { it.ok }) break
                for (ti in pending.toList()) {
                    val (pname, refs) = pages[ti]
                    var found: PageRegistration? = null
                    val cands = ArrayList<Cand>()
                    val refsToSearch = if (round == 1) emptyList() else refs
                    for (tpl in refsToSearch) {
                        for (qi in q.indices) {
                            if (qi in taken) continue
                            val (qk, qd) = quadFeats[qi]
                            val (hArr, inl, _) = match(tpl, qk, qd, COARSE_REPROJ_PX)
                            if (hArr != null && inl >= MIN_COARSE_INLIERS) {
                                cands += Cand(inl, hArr, "quad$qi", polygonArea(q[qi]), tpl)
                            }
                        }
                        if (cands.isEmpty() || cands.maxOf { it.inl } < GOOD_COARSE_INLIERS) {
                            val (hArr, inl, _) = match(tpl, kp, descMat, COARSE_REPROJ_PX)
                            if (hArr != null && inl >= MIN_COARSE_INLIERS) {
                                cands += Cand(inl, hArr, "global", null, tpl)
                            }
                        }
                        if (cands.isNotEmpty() && cands.maxOf { it.inl } >= GOOD_COARSE_INLIERS) break
                    }
                    cands.sortByDescending { it.inl }
                    for (cand in cands.take(MAX_COARSE_CANDIDATES)) {
                        found = accept(gray, cand.tpl, cand.h, REFINE_RADIUS_PX, cand.refArea, false)
                        if (found != null) {
                            found.method = cand.method; found.ref = refs.indexOf(cand.tpl)
                            break
                        }
                    }
                    if (found == null) {
                        // chain prior from an already registered page of the spread
                        outer@ for (tj in results.indices) {
                            if (tj == ti || !results[tj].ok) continue
                            val other = results[tj]
                            val dy = (ti - tj) * pageH * (1 + CHAIN_GAP_FRAC)
                            val prior = Homography.mul(Homography.translateY(dy), other.H!!)
                            for (tpl in refs) {
                                found = accept(gray, tpl, prior, CHAIN_RADIUS_PX, null, true)
                                if (found != null) {
                                    found.method = "chain:${other.name}"; found.ref = refs.indexOf(tpl)
                                    break
                                }
                            }
                            if (found != null) break@outer
                        }
                    }
                    if (found == null && round == 0) {
                        // Borders quad as a geometric prior
                        outer@ for (qi in q.indices) {
                            if (qi in taken) continue
                            val prior = perspectiveTransform(orderPoints(q[qi]), refs[0].corners)
                            for (tpl in refs) {
                                found = accept(gray, tpl, prior, CHAIN_RADIUS_PX, polygonArea(q[qi]), true)
                                if (found != null) {
                                    found.method = "quadprior$qi"; found.ref = refs.indexOf(tpl)
                                    break
                                }
                            }
                            if (found != null) break@outer
                        }
                    }
                    if (found == null) continue
                    found.name = pname
                    results[ti] = found
                    pending.remove(ti)
                    when {
                        found.method.startsWith("quadprior") -> taken += found.method.substring(9).toInt()
                        found.method.startsWith("quad") -> taken += found.method.substring(4).toInt()
                    }
                }
                if (pending.isEmpty()) break
            }
            kpMat.release(); descMat.release()
            return results
        } finally {
            gray.release()
        }
    }

    private class Cand(val inl: Int, val h: Array<DoubleArray>, val method: String, val refArea: Double?, val tpl: PageTemplate)

    /** Output scale (<=1) at which no registered page is upsampled. `native_scale`. */
    public fun nativeScale(regs: List<PageRegistration>): Double {
        var scale = 1.0
        for (r in regs) {
            if (!r.ok) continue
            val q = orderPoints(r.quad!!)
            val wNative = 0.5 * (dist(q[1], q[0]) + dist(q[2], q[3]))
            scale = min(scale, wNative / pageW)
        }
        return max(scale, 0.25)
    }

    public fun outSize(scale: Double): Pair<Int, Int> =
        max(1, Math.round(outW * scale).toInt()) to max(1, Math.round(outH * scale).toInt())

    /** Warp a photo quad (e.g. from Borders) into the canonical page frame. `warp_quad`. */
    public fun warpQuad(imgRgb: Image, quad: Array<Pt>, scale: Double): Image {
        val m = margin
        val src = orderPoints(quad)
        val dst = arrayOf(
            Pt(m * scale, m * scale), Pt((m + pageW) * scale, m * scale),
            Pt((m + pageW) * scale, (m + pageH) * scale), Pt(m * scale, (m + pageH) * scale))
        val srcMat = MatOfPoint2f(*src.map { CvPoint(it.x, it.y) }.toTypedArray())
        val dstMat = MatOfPoint2f(*dst.map { CvPoint(it.x, it.y) }.toTypedArray())
        val mMat = Imgproc.getPerspectiveTransform(srcMat, dstMat)
        val (w, h) = outSize(scale)
        val out = Mat()
        try {
            Imgproc.warpPerspective(imgRgb.mat, out, mMat, Size(w.toDouble(), h.toDouble()),
                Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
            return Image.wrap(out)
        } finally {
            srcMat.release(); dstMat.release(); mMat.release()
        }
    }

    /** Warp the photo to the canonical page with the margin cushion, at `scale`. `warp_page`. */
    public fun warpPage(imgRgb: Image, reg: PageRegistration, scale: Double): Image {
        val m = margin.toDouble()
        val shift = arrayOf(doubleArrayOf(1.0, 0.0, m), doubleArrayOf(0.0, 1.0, m), doubleArrayOf(0.0, 0.0, 1.0))
        val s = arrayOf(doubleArrayOf(scale, 0.0, 0.0), doubleArrayOf(0.0, scale, 0.0), doubleArrayOf(0.0, 0.0, 1.0))
        val hFull = Homography.mul(s, Homography.mul(shift, reg.H!!))
        val (w, h) = outSize(scale)
        val out = Mat()
        val hMat = Homography.toMat(hFull)
        try {
            Imgproc.warpPerspective(imgRgb.mat, out, hMat, Size(w.toDouble(), h.toDouble()),
                Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE)
            return Image.wrap(out)
        } finally {
            hMat.release()
        }
    }

    /**
     * Page quads (TL,TR,BR,BL) from Borders contours, plus per-quad fit info. `page_quads`.
     * With [lineQuads] the sides are [QuadFit] line fits; otherwise the plain polygon corners.
     */
    public fun pageQuads(segments: List<List<Pt>>?, imageShape: Pair<Int, Int>): Pair<List<Array<Pt>>, List<QuadFit.Info>> {
        val quads = ArrayList<Array<Pt>>()
        val infos = ArrayList<QuadFit.Info>()
        for (s in segments ?: emptyList()) {
            val (q, info) = if (lineQuads) {
                QuadFit.fitQuadLines(s, imageShape, pageH.toDouble() / pageW)
            } else {
                net.russiandocs.docproc.imaging.Geometry.extractQuad(s) to QuadFit.Info().also { it.method = "polygon" }
            }
            if (q != null) {
                quads += orderPoints(q.toTypedArray())
                infos += info
            }
        }
        return quads to infos
    }

    /** Straighten a warped page by its own lines: first the homography ([LineRefine]), then the bend map
     * ([LineDewarp]). `straighten`. */
    public fun straighten(page: Image, scale: Double): Pair<Image, StraightenInfo> {
        val inset = Math.round(margin * scale).toInt()
        var current = page
        var owns = false
        val info = StraightenInfo()
        if (lineRefine) {
            val gray = Io.toGray(current)
            val (hm, refineInfo) = try { LineRefine.refineByLines(gray, inset) } finally { gray.close() }
            info.refine = refineInfo
            if (hm != null) {
                val refined = LineRefine.applyRefinement(current, hm)
                if (owns) current.close()
                current = refined
                owns = true
            }
        }
        if (lineDewarp) {
            val gray = Io.toGray(current)
            val (v, dewarpInfo) = try { LineDewarp.dewarpByLines(gray, inset) } finally { gray.close() }
            info.dewarp = dewarpInfo
            if (v != null) {
                val dewarped = LineDewarp.applyDewarp(current, v)
                v.release()
                if (owns) current.close()
                current = dewarped
                owns = true
            }
        }
        return current to info
    }

    public class StraightenInfo {
        public var refine: LineRefine.Info? = null
        public var dewarp: LineDewarp.Info? = null
    }

    /** IoU of two convex photo quads. `quad_iou`, static in the reference. */
    public fun quadIou(a: Array<Pt>, b: Array<Pt>): Double {
        val ao = orderPoints(a)
        val bo = orderPoints(b)
        val am = MatOfPoint2f(*ao.map { CvPoint(it.x, it.y) }.toTypedArray())
        val bm = MatOfPoint2f(*bo.map { CvPoint(it.x, it.y) }.toTypedArray())
        val inter = Mat()
        return try {
            val interArea = Imgproc.intersectConvexConvex(am, bm, inter)
            val union = polygonArea(ao) + polygonArea(bo) - interArea
            if (union > 0) interArea / union else 0.0
        } finally {
            am.release(); bm.release(); inter.release()
        }
    }

    override fun close() {
        pages.forEach { (_, refs) -> refs.forEach { it.close() } }
        matcher.clear()
    }

    // ---------------------------------------------------------------- small geometry helpers

    private fun dist(a: Pt, b: Pt): Double = hypot(a.x - b.x, a.y - b.y)

    /** `_order_points`: TL (min sum), TR, BR (max sum), BL. Delegates to the already-verified
     * [net.russiandocs.docproc.imaging.Geometry.orderPoints] instead of keeping a second copy here -
     * an earlier local duplicate used `x - y` where Python's `d = np.diff(pts, axis=1)[:, 0]` is
     * `y - x`, which silently swapped TR/BL and corrupted every wt/ht (width/height) check built on
     * top of it (found while diagnosing why every coarse registration candidate failed `saneQuad`). */
    private fun orderPoints(pts: Array<Pt>): Array<Pt> =
        (net.russiandocs.docproc.imaging.Geometry.orderPoints(pts.toList()) ?: pts.toList()).toTypedArray()

    private fun polygonArea(pts: Array<Pt>): Double {
        var area = 0.0
        for (i in pts.indices) {
            val j = (i + 1) % pts.size
            area += pts[i].x * pts[j].y - pts[j].x * pts[i].y
        }
        return abs(area) / 2.0
    }

    private fun isConvexF(pts: Array<Pt>): Boolean {
        // Cross-product sign test at float precision (the Java isContourConvex binding is integer-only —
        // see QuadFit's note on the same platform limitation). A simple quad from a homography transform
        // is not the near-degenerate case that would make sign flips at float precision differ from
        // OpenCV's own float32 test.
        var sign = 0
        for (i in pts.indices) {
            val a = pts[i]; val b = pts[(i + 1) % pts.size]; val c = pts[(i + 2) % pts.size]
            val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            val s = if (cross > 0) 1 else if (cross < 0) -1 else 0
            if (s != 0) {
                if (sign == 0) sign = s else if (sign != s) return false
            }
        }
        return true
    }

    private fun perspectiveTransform(src: Array<Pt>, dst: Array<Pt>): Array<DoubleArray> {
        val srcMat = MatOfPoint2f(*src.map { CvPoint(it.x, it.y) }.toTypedArray())
        val dstMat = MatOfPoint2f(*dst.map { CvPoint(it.x, it.y) }.toTypedArray())
        val m = Imgproc.getPerspectiveTransform(srcMat, dstMat)
        return try {
            Homography.fromMat(m)
        } finally {
            srcMat.release(); dstMat.release(); m.release()
        }
    }
}
