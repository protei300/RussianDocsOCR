package net.russiandocs.docproc.modules

import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Contours
import net.russiandocs.docproc.imaging.Geometry
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.imaging.StackDirection
import net.russiandocs.docproc.models.SegmentationModel
import net.russiandocs.docproc.pipeline.Device
import java.io.File
import kotlin.math.max

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
    public fun predictTransform(image: Image, maxPages: Int): Pair<Image, List<List<Pt>>?> {
        val (_, segments) = model.predict(image)
        if (segments.isEmpty()) {
            return image.clone() to null
        }

        val kept = selectPages(segments, maxPages)
        if (kept.isEmpty()) {
            return image.clone() to null
        }

        val chosen = kept.map { segments[it] }
        val (warped, ok) = Geometry.fixPerspective(image, chosen, StackDirection.AUTO,
            Geometry.DOC_MARGIN_FRACTION)

        return if (ok && warped != null) {
            warped to chosen
        } else {
            image.clone() to chosen
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

    override fun close(): Unit = model.close()

    private companion object {
        /**
         * The share of the largest page's area a second segment must reach to be kept.
         *
         * 0.6, from the reference. It is what stops a background blob being stitched onto a single-page
         * document, and what allows the two halves of a passport spread to both survive.
         */
        const val SECOND_SEGMENT_AREA_FRACTION = 0.6
    }
}
