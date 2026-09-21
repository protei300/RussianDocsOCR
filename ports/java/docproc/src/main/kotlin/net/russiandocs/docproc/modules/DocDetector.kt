package net.russiandocs.docproc.modules

import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Contours
import net.russiandocs.docproc.imaging.Geometry
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Interpolation
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.imaging.Placement
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.imaging.StackDirection
import net.russiandocs.docproc.models.SegmentationModel
import net.russiandocs.docproc.pipeline.Device
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point as CvPoint
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.hypot
import kotlin.math.max

/**
 * Everything `DocDetector.predict_transform` returns in the reference (doc_detector.py:170-186).
 *
 * [pages]/[placements] are empty except for a genuine two-page spread — see [DocDetector.predictTransformFull].
 * [canvas] and [segments] are owned by the caller, and so is every entry of [pages].
 */
public class BordersResult(
    public val canvas: Image,
    public val segments: List<List<Pt>>?,
    public val pages: List<Image> = emptyList(),
    public val placements: List<Placement> = emptyList(),
)

/** Finds the document's borders and returns the perspective-corrected canvas. */
public class DocDetector(
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
) : AutoCloseable {

    private val model = SegmentationModel(
        File(ModelPaths.resolve(root, paths, "DocDetector"), "ONNX").path, device, threads, root)

    /**
     * Returns the corrected canvas and the SELECTED contours.
     *
     * The contours travel out alongside the canvas so the conformance harness can compare them
     * (`borders.segments`) and localise a divergence to the mask rather than to the warp. That distinction
     * earned its keep immediately in the Go port: segments matched while the canvas was six pixels narrow,
     * which placed the bug in the quadrilateral extraction and nowhere else.
     *
     * When no usable segment is found the ORIGINAL image is returned. Not a safety net bolted on — it is
     * what the reference does, and a port that errored instead would fail every document whose borders the
     * model cannot see.
     */
    public fun predictTransform(image: Image, maxPages: Int): Pair<Image, List<List<Pt>>?> =
        predictTransformFull(image, maxPages).let { it.canvas to it.segments }

    /**
     * Same as [predictTransform], but also returns the individual per-page pieces — port of
     * `DocDetector.predict_transform`'s `pages`/`page_placements` keys (doc_detector.py:170-186).
     *
     * [BordersResult.pages]/[BordersResult.placements] are empty unless exactly two pages were stitched:
     * that is the only case the reference's `_fields_from_pages` (pipeline.py:1238, `len(pages) >= 2`)
     * reads them at all, so a single-page document's pieces are not worth carrying past this call.
     */
    public fun predictTransformFull(image: Image, maxPages: Int): BordersResult {
        val (_, detected) = model.predict(image)
        if (detected.isEmpty()) {
            return BordersResult(image.clone(), null)
        }

        // First drop segments without ink (a scanner lid, a blank sheet next to the document), THEN
        // the area rule — the reference's order (doc_detector.py:127).
        val segments = dropBlankSegments(image, detected)
        val kept = selectPages(segments, maxPages)
        if (kept.isEmpty()) {
            return BordersResult(image.clone(), null)
        }

        val chosen = kept.map { segments[it] }
        val result = Geometry.fixPerspective(image, chosen, StackDirection.AUTO,
            Geometry.DOC_MARGIN_FRACTION)

        return if (result.ok && result.canvas != null) {
            BordersResult(result.canvas, chosen, result.pages, result.placements)
        } else {
            // `fixPerspective` never hands pages off on a failed warp — `ok=false` only on its
            // zero-page and exception paths, both of which return `pages=emptyList()`.
            BordersResult(image.clone(), chosen)
        }
    }

    /**
     * Ranks segments by contour area and applies the area-fraction rule.
     *
     * Returns indices in ASCENDING order, matching the reference's `sorted(keep)` — and that order then
     * decides which page [Geometry.fixPerspective] treats as first when stitching a spread.
     *
     * The ranking sort is STABLE and descending: two segments of identical area keep their detection order,
     * so the choice between them is deterministic. `sortedByDescending` is stable on the JVM; a primitive
     * sort would not be.
     */
    private fun selectPages(segments: List<List<Pt>>, maxPages: Int): List<Int> {
        val areas = segments.map { if (it.size >= 3) Contours.contourArea(it) else 0.0 }

        val order = areas.indices.sortedByDescending { areas[it] }
        if (order.isEmpty() || areas[order[0]] <= 0) {
            return emptyList()
        }

        val limit = max(1, maxPages)
        val maxArea = areas[order[0]]
        val keep = mutableListOf(order[0])
        for (index in order.drop(1)) {
            if (keep.size >= limit) {
                break
            }
            if (areas[index] >= SECOND_SEGMENT_AREA_FRACTION * maxArea) {
                keep += index
            }
        }
        return keep.sorted()
    }

    /**
     * The segments to keep once the blank ones are dropped. Port of `drop_blank_segments`.
     *
     * A segment with (almost) no ink inside it is not a document page, however confident the model is:
     * the white lid of a flatbed scanner next to a passport scores 0.90–0.96 as 'Document' and, being 3x
     * larger than a page, used to win the area rule and push both real pages out (~20 of 100 scans in the
     * Damir set came out as an empty canvas). Ink is the mean gradient magnitude inside the eroded mask at
     * ~400 px: measured 1.5–5.4 on lids, 19–101 on passport pages, 24–43 on the mostly bare registration
     * page. A blank segment is dropped only when another segment with real ink exists, so a lone blank
     * sheet still goes through as before.
     */
    internal fun dropBlankSegments(image: Image, segments: List<List<Pt>>): List<List<Pt>> {
        val ink = segmentInks(image, segments)
        return if (ink.any { it >= BLANK_INK }) {
            segments.filterIndexed { i, _ -> ink[i] >= BLANK_INK }
        } else {
            segments
        }
    }

    /**
     * Mean gradient magnitude inside every contour, eroded so the segment's own edge does not count, on
     * the image downscaled to [INK_SCALE_PX]. Port of `segment_ink`, with the shared work — the grey
     * downscale and the two Sobel passes — done once for all segments instead of once per segment; the
     * per-segment numbers are the same.
     */
    private fun segmentInks(image: Image, segments: List<List<Pt>>): DoubleArray {
        Io.toGray(image).use { gray ->
            val h = gray.height
            val w = gray.width
            val s = INK_SCALE_PX / max(h, w).toDouble()
            Io.resize(gray, max(1, (w * s).toInt()), max(1, (h * s).toInt()), Interpolation.AREA)
                .use { small ->
                    val gx = Mat()
                    val gy = Mat()
                    try {
                        Imgproc.Sobel(small.mat, gx, CvType.CV_32F, 1, 0, 3)
                        Imgproc.Sobel(small.mat, gy, CvType.CV_32F, 0, 1, 3)
                        val rows = small.height
                        val cols = small.width
                        val fx = FloatArray(rows * cols)
                        val fy = FloatArray(rows * cols)
                        gx.get(0, 0, fx)
                        gy.get(0, 0, fy)
                        return DoubleArray(segments.size) { i ->
                            segmentInk(segments[i], s, rows, cols, fx, fy)
                        }
                    } finally {
                        gx.release()
                        gy.release()
                    }
                }
        }
    }

    private fun segmentInk(
        contour: List<Pt>,
        scale: Double,
        rows: Int,
        cols: Int,
        gx: FloatArray,
        gy: FloatArray,
    ): Double {
        if (contour.size < 3) {
            return 0.0
        }
        // `np.int32(np.round(pts * s))` on a float32 contour: the product is a float32 multiply and the
        // rounding is half to even. Reproduced in that precision so a corner landing on x.5 rounds the
        // same way.
        val sf = scale.toFloat()
        val polygon = MatOfPoint(*contour.map { p ->
            CvPoint(Math.rint((p.x.toFloat() * sf).toDouble()), Math.rint((p.y.toFloat() * sf).toDouble()))
        }.toTypedArray())
        val mask = Mat.zeros(rows, cols, CvType.CV_8UC1)
        val eroded = Mat()
        val kernel = Mat.ones(5, 5, CvType.CV_8U)
        try {
            Imgproc.fillPoly(mask, listOf(polygon), Scalar(255.0))
            Imgproc.erode(mask, eroded, kernel)
            if (Core.countNonZero(eroded) == 0) {
                return 0.0
            }
            val m = ByteArray(rows * cols)
            eroded.get(0, 0, m)
            var sum = 0.0
            var count = 0
            for (i in m.indices) {
                if (m[i].toInt() != 0) {
                    sum += hypot(gx[i].toDouble(), gy[i].toDouble())
                    count++
                }
            }
            return if (count == 0) 0.0 else sum / count
        } finally {
            polygon.release()
            mask.release()
            eroded.release()
            kernel.release()
        }
    }

    override fun close(): Unit = model.close()

    private companion object {
        /**
         * The share of the largest page's area a second segment must reach to be kept.
         *
         * 0.6, from the reference. It is what stops a background blob being stitched onto a single-page
         * document, and what allows the two halves of a passport spread to both survive.
         */
        const val SECOND_SEGMENT_AREA_FRACTION = 0.6

        /** Mean gradient magnitude below which a segment carries no ink (doc_detector.py:26). */
        const val BLANK_INK = 8.0

        /** The longest side the image is scaled to before the ink is measured. */
        const val INK_SCALE_PX = 400.0
    }
}
