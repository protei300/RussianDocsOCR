package net.russiandocs.docproc.modules

import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Crop
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.models.DetectionModel
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.postprocess.Box
import java.io.File

/** One detected field: its box and the cropped patch. **The patch is owned by the holder.** */
public class Field(public val box: Box, public val patch: Image) : AutoCloseable {
    override fun close(): Unit = patch.close()
}

/** Closes every field's patch. Safe on a partially built list. */
public fun closeAllFields(fields: Iterable<Field>?) {
    fields?.forEach { it.close() }
}

/** Locates the text fields on a corrected canvas and crops each one. */
public class TextFieldsDetector(
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
) : AutoCloseable {

    private val model = DetectionModel(
        File(ModelPaths.resolve(root, paths, "TextFieldsDetector"), "ONNX").path,
        device, threads, root)

    /**
     * Detects and crops.
     *
     * The crop goes through [Crop.clampedCrop], which is not optional: this detector routinely returns boxes
     * a pixel or two outside the canvas, and a literal translation of the reference's slice would throw on
     * them.
     *
     * [rotateLicence] rotates the `Licence_number` patch a quarter turn. The internal passport prints its
     * series and number sideways, so without this the OCR reads a vertical strip. Only that one field, and
     * only for that document type.
     *
     * On any failure every patch cropped so far is closed before the exception leaves — a partial list of
     * owned Mats that nobody holds is how a leak starts.
     */
    public fun predictTransform(canvas: Image, rotateLicence: Boolean): List<Field> {
        val boxes = model.predict(canvas)
        val fields = ArrayList<Field>(boxes.size)
        try {
            for (box in boxes) {
                var patch = Crop.clampedCrop(canvas, box.x1.toInt(), box.y1.toInt(),
                    box.x2.toInt(), box.y2.toInt())
                if (rotateLicence && box.label == "Licence_number") {
                    val rotated = Io.rot90(patch, 1)
                    patch.close()
                    patch = rotated
                }
                fields += Field(box, patch)
            }
            return fields
        } catch (e: Throwable) {
            closeAllFields(fields)
            throw e
        }
    }

    override fun close(): Unit = model.close()
}

/** Splits a field patch into word crops, in reading order. */
public class WordsDetector(
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
) : AutoCloseable {

    private val model = DetectionModel(
        File(ModelPaths.resolve(root, paths, "WordsDetector"), "ONNX").path, device, threads, root)

    /**
     * Word boxes and their crops, in reading order.
     *
     * The boxes are returned REORDERED, not just the crops: that order is what the conformance
     * `words.<field>.bbox` stage records and what the OCR loop walks, so the two must agree.
     *
     * An empty result is normal — the caller falls back to the whole patch, as the reference does.
     */
    public fun predictTransform(patch: Image): Pair<List<Box>, List<Image>> {
        val boxes = readingOrder(model.predict(patch))

        val words = ArrayList<Image>(boxes.size)
        try {
            for (box in boxes) {
                // Cut ON the box. Python pads small word boxes by 2 px since 1cc8468, and the ports
                // deliberately do NOT follow yet: the words detector is being retrained with the margin
                // inside the labelled box, which may remove the compensation altogether. The ports are
                // synced to the FINAL Python behaviour in one pass before the goldens are regenerated.
                words += Crop.clampedCrop(patch, box.x1.toInt(), box.y1.toInt(),
                    box.x2.toInt(), box.y2.toInt())
            }
            return boxes to words
        } catch (e: Throwable) {
            // Release what was already cropped: a sibling failing must not leave these orphaned.
            words.forEach { it.close() }
            throw e
        }
    }

    override fun close(): Unit = model.close()

    internal companion object {
        /**
         * Sorts word boxes into reading order: cluster into lines by vertical centre proximity (within half a
         * word height), lines top-to-bottom, words left-to-right inside a line. Port of
         * `WordsDetector._reading_order`.
         *
         * A plain x-sort interleaves the lines of a multi-line field — measured on the birth certificates'
         * Birth_place/ZAGS fields as word salad — so this is a correctness rule, not a tidiness one. On a
         * single-line field it reproduces the old x-sorted order exactly.
         *
         * Two things are load-bearing. Every sort is STABLE (`sortedBy` is; a primitive array sort would not
         * be), or two words sharing a centre or an x1 would swap. And the running means are updated per box, in
         * the reference's order: a box joins the FIRST line it fits, and the line's centre and height are the
         * means over the boxes admitted so far — comparing against the first box instead would cluster
         * differently on a field whose line drifts.
         */
        internal fun readingOrder(boxes: List<Box>): List<Box> {
            class Line(var cy: Double, var h: Double, val boxes: MutableList<Box>)

            val lines = ArrayList<Line>()
            for (box in boxes.sortedBy { (it.y1 + it.y2) / 2 }) {
                val cy = (box.y1 + box.y2) / 2
                val h = box.y2 - box.y1
                val line = lines.firstOrNull { kotlin.math.abs(cy - it.cy) < 0.5 * kotlin.math.max(h, it.h) }
                if (line != null) {
                    val n = line.boxes.size.toDouble()
                    line.cy = (line.cy * n + cy) / (n + 1)
                    line.h = (line.h * n + h) / (n + 1)
                    line.boxes += box
                } else {
                    lines += Line(cy, h, mutableListOf(box))
                }
            }

            val ordered = ArrayList<Box>(boxes.size)
            for (line in lines) {   // already top-to-bottom
                ordered += line.boxes.sortedBy { it.x1 }
            }
            return ordered
        }
    }
}
