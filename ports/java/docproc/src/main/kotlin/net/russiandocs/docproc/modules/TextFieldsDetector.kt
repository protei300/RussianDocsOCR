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

/** Splits a field patch into word crops, left to right. */
public class WordsDetector(
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
) : AutoCloseable {

    private val model = DetectionModel(
        File(ModelPaths.resolve(root, paths, "WordsDetector"), "ONNX").path, device, threads, root)

    /**
     * Word boxes and their crops, left to right.
     *
     * **The ordering is the one trap here.** The reference sorts with `bbox.sort(key=lambda x: x[0])`, and
     * Python's sort is STABLE — so words keep the reading-order sort the detector already applied whenever
     * their x1 ties. Kotlin's `sortedBy` is stable; a primitive array sort would not be. Two words sharing an
     * x1 would otherwise swap and reorder two tokens of the joined field string.
     *
     * An empty result is normal — the caller falls back to the whole patch, as the reference does.
     */
    public fun predictTransform(patch: Image): Pair<List<Box>, List<Image>> {
        val boxes = model.predict(patch).sortedBy { it.x1 }

        val words = ArrayList<Image>(boxes.size)
        try {
            for (box in boxes) {
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
}
