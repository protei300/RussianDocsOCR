package net.russiandocs.docproc.imaging

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point as CvPoint
import org.opencv.core.RotatedRect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs

/** A point in image coordinates. Double, because contours carry sub-pixel positions. */
public data class Pt(val x: Double, val y: Double)

/**
 * Contour extraction and the OpenCV calls around it.
 *
 * **Every OpenCV default argument the reference relies on is passed EXPLICITLY here.** That is CONVENTIONS
 * trap 15 and it is not pedantry: `cv2.convexHull(cnt)` defaults to `clockwise=False`, while the JVM
 * binding, gocv and OpenCvSharp all make the parameter required — so a port CHOOSES a value instead of
 * inheriting one. The Go port chose `true` and lost six pixels of canvas width on an internal-passport
 * spread, because the hull's ORIENTATION decides which vertices Douglas-Peucker keeps. Audit against the
 * Python call site, never against the binding's own defaults.
 */
public object Contours {

    private fun toMatOfPoint2f(points: List<Pt>): MatOfPoint2f =
        MatOfPoint2f(*points.map { CvPoint(it.x, it.y) }.toTypedArray())

    /** Otsu threshold. Returns the binary image and the chosen threshold value. */
    public fun thresholdOtsu(src: Image, invert: Boolean): Pair<Image, Double> {
        val gray = if (src.channels == 1) src.clone() else Io.toGray(src)
        gray.use {
            val binary = Mat()
            val type = (if (invert) Imgproc.THRESH_BINARY_INV else Imgproc.THRESH_BINARY) or
                Imgproc.THRESH_OTSU
            val value = Imgproc.threshold(gray.mat, binary, 0.0, 255.0, type)
            return Image.wrap(binary) to value
        }
    }

    /** External contours only, no hierarchy — `RETR_EXTERNAL` with `CHAIN_APPROX_SIMPLE`. */
    public fun findExternalContours(src: Image): List<List<Pt>> {
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        try {
            Imgproc.findContours(src.mat, contours, hierarchy, Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE)
            return contours.map { contour ->
                contour.toArray().map { Pt(it.x, it.y) }
            }
        } finally {
            hierarchy.release()
            contours.forEach { it.release() }
        }
    }

    /** Contour area, via the shoelace formula OpenCV uses. */
    public fun contourArea(points: List<Pt>): Double {
        if (points.size < 3) {
            return 0.0
        }
        val mat = toMatOfPoint2f(points)
        try {
            return abs(Imgproc.contourArea(mat))
        } finally {
            mat.release()
        }
    }

    /**
     * The convex hull. `clockwise = false` — the reference's default, and NOT the binding's.
     *
     * The JVM binding returns INDICES rather than points, which is a second trap on top of the
     * orientation one: `Imgproc.convexHull` fills a `MatOfInt`, and the caller has to map them back. A
     * port that expected points would find the "hull" is a list of small integers and, worse, might use
     * them as coordinates.
     */
    public fun convexHull(points: List<Pt>): List<Pt> {
        if (points.isEmpty()) {
            return emptyList()
        }
        // Integer points, because the JVM's convexHull takes MatOfPoint (int) rather than MatOfPoint2f.
        // The reference feeds it a contour, which is integral already, so nothing is lost here.
        val input = MatOfPoint(*points.map { CvPoint(it.x, it.y) }.toTypedArray())
        val indices = org.opencv.core.MatOfInt()
        try {
            Imgproc.convexHull(input, indices, false)
            val source = input.toArray()
            return indices.toArray().map { Pt(source[it].x, source[it].y) }
        } finally {
            input.release()
            indices.release()
        }
    }

    /** Perimeter of a CLOSED contour — `cv2.arcLength(pts, True)`. */
    public fun arcLength(points: List<Pt>): Double {
        if (points.size < 2) {
            return 0.0
        }
        val mat = toMatOfPoint2f(points)
        try {
            return Imgproc.arcLength(mat, true)
        } finally {
            mat.release()
        }
    }

    /** Douglas-Peucker simplification of a CLOSED contour. */
    public fun approxPolyDp(points: List<Pt>, epsilon: Double): List<Pt> {
        val input = toMatOfPoint2f(points)
        val output = MatOfPoint2f()
        try {
            Imgproc.approxPolyDP(input, output, epsilon, true)
            return output.toArray().map { Pt(it.x, it.y) }
        } finally {
            input.release()
            output.release()
        }
    }

    /**
     * The four corners of the minimum-area rectangle.
     *
     * Taken from `boxPoints` rather than reconstructed from centre/size/angle: OpenCV changed the
     * `minAreaRect` angle convention around 4.5, and a hand-rolled version silently produces the corners
     * in a different ORDER — which then feeds a perspective transform.
     */
    public fun minAreaRectPoints(points: List<Pt>): List<Pt> {
        val mat = toMatOfPoint2f(points)
        val box = Mat()
        try {
            val rect: RotatedRect = Imgproc.minAreaRect(mat)
            Imgproc.boxPoints(rect, box)
            return (0 until box.rows()).map { r ->
                Pt(box.get(r, 0)[0], box.get(r, 1)[0])
            }
        } finally {
            mat.release()
            box.release()
        }
    }

    /** Warps a quadrilateral onto an axis-aligned rectangle of the given size. */
    public fun warpPerspectiveQuad(src: Image, quad: List<Pt>, width: Int, height: Int): Image {
        require(quad.size == 4) { "imaging: warp needs 4 points, got ${quad.size}" }

        val source = toMatOfPoint2f(quad)
        val destination = MatOfPoint2f(
            CvPoint(0.0, 0.0),
            CvPoint((width - 1).toDouble(), 0.0),
            CvPoint((width - 1).toDouble(), (height - 1).toDouble()),
            CvPoint(0.0, (height - 1).toDouble()),
        )
        val transform = Imgproc.getPerspectiveTransform(source, destination)
        try {
            val dst = Mat()
            Imgproc.warpPerspective(src.mat, dst, transform,
                Size(width.toDouble(), height.toDouble()))
            return Image.wrap(dst)
        } finally {
            source.release()
            destination.release()
            transform.release()
        }
    }

    /** Horizontal concatenation. Heights must match. */
    public fun hStack(a: Image, b: Image): Image {
        require(a.height == b.height) {
            "imaging: hstack heights differ: ${a.height} vs ${b.height}"
        }
        val dst = Mat()
        Core.hconcat(listOf(a.mat, b.mat), dst)
        return Image.wrap(dst)
    }

    /** Vertical concatenation. Widths must match. */
    public fun vStack(a: Image, b: Image): Image {
        require(a.width == b.width) {
            "imaging: vstack widths differ: ${a.width} vs ${b.width}"
        }
        val dst = Mat()
        Core.vconcat(listOf(a.mat, b.mat), dst)
        return Image.wrap(dst)
    }

    /**
     * Rotates around a FRACTIONAL centre.
     *
     * **This is where D-08 does not apply to this port.** `Imgproc.getRotationMatrix2D` takes a
     * double-valued `Point`, so `(sw / 2.0, sh / 2.0)` is expressible and the real OpenCV function can be
     * called. gocv takes an integer `image.Point` and cannot, which is why the Go port builds the matrix
     * by hand — and its naive integer version was measured to shift the deskewer's variance array by
     * 3.8e-3 relative, above the 1e-3 policy.
     */
    public fun rotateAround(src: Image, centreX: Double, centreY: Double, angleDeg: Double): Image {
        val transform = Imgproc.getRotationMatrix2D(CvPoint(centreX, centreY), angleDeg, 1.0)
        try {
            val dst = Mat()
            Imgproc.warpAffine(src.mat, dst, transform,
                Size(src.width.toDouble(), src.height.toDouble()))
            return Image.wrap(dst)
        } finally {
            transform.release()
        }
    }

}
