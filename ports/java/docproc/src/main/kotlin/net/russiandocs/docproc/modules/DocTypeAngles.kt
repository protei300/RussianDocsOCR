package net.russiandocs.docproc.modules

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.models.Loader
import net.russiandocs.docproc.models.Model
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.postprocess.ClassResult
import net.russiandocs.docproc.postprocess.MetricResult
import net.russiandocs.docproc.tensors.Ops
import java.io.File

/**
 * Document type and 90-degree orientation, from one model with two heads.
 *
 * The field names are the wire contract — this is what the `doctype.label` stage compares, so every one is
 * written out with `@SerialName` rather than inferred.
 */
@Serializable
public data class DocTypeResult(
    @SerialName("doc_type") val docType: String = "NONE",
    @SerialName("doc_type_confidence") val docTypeConfidence: Double = 0.0,
    @SerialName("angle") val angle: Int = 0,
    @SerialName("angle_confidence") val angleConfidence: Double = 0.0,
)

/**
 * Port of `pipeline_modules/doctype_angles_classificator`.
 *
 * One model, two outputs, in the order the config declares them: an embedding that the metric head turns
 * into a document type, and a four-way angle classifier. They are not interchangeable, so both are cast to
 * their expected result type and a mismatch is an error rather than a reinterpretation.
 */
public class DocTypeAngles(
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
) : AutoCloseable {

    private val model: Model
    private val angleLabels: List<Int>

    init {
        val dir = File(ModelPaths.resolve(root, paths, MODULE_NAME), "ONNX").path
        model = Loader.load(dir, device, threads, root)

        if (model.config.outputs.size != 2) {
            model.close()
            throw IllegalArgumentException(
                "modules: $MODULE_NAME expects 2 outputs (embeddings, angle), got " +
                    "${model.config.outputs.size}",
            )
        }
        angleLabels = model.config.outputs[1].labelsAsInts()
    }

    public fun predict(image: Image): DocTypeResult {
        val outputs = model.predict(image)

        // Cast ONCE, here, in the module that knows what it asked for. This is the single place the closed
        // result set is narrowed — see CONVENTIONS §5.
        val metric = outputs[0] as? MetricResult
            ?: throw IllegalStateException(
                "modules: $MODULE_NAME output 0 is ${outputs[0]::class.simpleName}, want MetricResult")
        val angle = outputs[1] as? ClassResult
            ?: throw IllegalStateException(
                "modules: $MODULE_NAME output 1 is ${outputs[1]::class.simpleName}, want ClassResult")

        // The confidence the wire carries is a RATIO against the class threshold, not the raw distance,
        // and it is rounded to two places — matching the reference exactly, because this value is compared
        // as a float with a 1e-3 tolerance that leaves no room for a third digit.
        val confidence = if (metric.threshold > 0) {
            Ops.roundHalfEven(1 - metric.distance / metric.threshold, 2)
        } else {
            0.0
        }

        return DocTypeResult(
            docType = metric.label,
            docTypeConfidence = confidence,
            angle = angleFromLabel(angle.label),
            angleConfidence = angle.confidence,
        )
    }

    /**
     * Predicts, then rotates the image upright.
     *
     * `angle / 90` quarter-turns COUNTER-clockwise, because the angle names how far the document is
     * rotated and the correction undoes it. The result is a new image the caller owns; the input is only
     * borrowed.
     */
    public fun predictTransform(image: Image): Pair<DocTypeResult, Image> {
        val meta = predict(image)

        var current = image.clone()
        repeat(meta.angle / 90) {
            val next = Io.rot90(current, 1)
            current.close()
            current = next
        }
        return meta to current
    }

    /**
     * Maps the classifier's label back to degrees.
     *
     * A lookup rather than `toInt()`: the label comes from the config, and an angle the model does not
     * declare must be an error rather than a number that happens to parse.
     */
    private fun angleFromLabel(label: String): Int {
        for (candidate in angleLabels) {
            if (candidate.toString() == label) {
                return candidate
            }
        }
        throw IllegalArgumentException(
            "modules: $MODULE_NAME angle label \"$label\" is not one of $angleLabels")
    }

    override fun close(): Unit = model.close()

    private companion object {
        const val MODULE_NAME = "DocTypeAngles"
    }
}
