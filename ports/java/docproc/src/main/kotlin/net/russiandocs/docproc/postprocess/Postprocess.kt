package net.russiandocs.docproc.postprocess

import net.russiandocs.docproc.models.ModelOutput
import net.russiandocs.docproc.tensors.NdArray
import net.russiandocs.docproc.tensors.Npy
import net.russiandocs.docproc.tensors.Ops

/** What a postprocessor needs from the preprocessing that fed the model. */
public data class Context(
    val ratio: Double = 1.0,
    val padExtra: IntArray = intArrayOf(0, 0),
    val padLetter: DoubleArray = doubleArrayOf(0.0, 0.0),
    val paddedH: Int = 0,
    val paddedW: Int = 0,
    val origH: Int = 0,
    val origW: Int = 0,
    val resize: Boolean = false,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Context) return false
        return ratio == other.ratio && padExtra.contentEquals(other.padExtra) &&
            padLetter.contentEquals(other.padLetter) && paddedH == other.paddedH &&
            paddedW == other.paddedW && origH == other.origH && origW == other.origW &&
            resize == other.resize
    }

    override fun hashCode(): Int {
        var result = ratio.hashCode()
        result = 31 * result + padExtra.contentHashCode()
        result = 31 * result + padLetter.contentHashCode()
        result = 31 * result + paddedH
        result = 31 * result + paddedW
        result = 31 * result + origH
        result = 31 * result + origW
        result = 31 * result + resize.hashCode()
        return result
    }
}

/**
 * The closed set of postprocessor results.
 *
 * A sealed interface rather than a generic `Model<T>`: the concrete type is not known until `model.json`
 * has been read, so a generic model would still need a runtime cast at the load site. All three preceding
 * ports reached the same conclusion. The cast happens ONCE, in the module that knows what it asked for.
 *
 * Sealed rather than a bare marker, which is the one improvement the JVM allows here: an exhaustive `when`
 * over the results now fails to compile if a variant is added, where the other ports would only fail at
 * runtime.
 */
public sealed interface ModelResult

/** A single label plus its score. Angle heads and the quality classifiers. */
public data class ClassResult(val label: String, val confidence: Double) : ModelResult

/** Nearest-centroid outcome. [label] is `"NONE"` when nothing is close enough. */
public data class MetricResult(
    val label: String,
    val distance: Double,
    val threshold: Double,
) : ModelResult

public interface Postprocessor {
    public fun apply(output: NdArray, context: Context): ModelResult
}

/** Wired rather than omitted, for the same reason as the preprocessor twin (D-06). */
public class NotImplementedPostprocessor(private val tag: String) : Postprocessor {
    override fun apply(output: NdArray, context: Context): ModelResult =
        throw NotImplementedError("postprocess: output type \"$tag\" is not implemented")
}

/**
 * Picks the nearest centroid — the head `DocTypeAngles` uses to name the document type.
 *
 * The reference builds a sklearn `NearestNeighbors(metric='cosine', radius=1)` index for this. With NINE
 * centroids that is a linear scan with extra steps, so this is the scan: no equivalent of sklearn is
 * needed here or anywhere else in the project.
 *
 * Two things about the outcome are easy to get subtly wrong. The `radius` is a HARD FILTER — a centroid
 * further away than the radius is not a neighbour at all, not merely a bad one. And the per-class
 * `max_distance` is applied AFTER the nearest is chosen, so a document can have a clear nearest centroid
 * and still come back `"NONE"`.
 */
public class Metric(npzPath: String, metric: String) : Postprocessor {

    private val radius: Double
    private val cosine: Boolean
    private val labels: Array<String>
    private val centers: Array<FloatArray>
    private val maxDistance: DoubleArray

    init {
        when (metric) {
            "Cosine", "cosine" -> { radius = 1.0; cosine = true }
            "Euclidean", "euclidean" -> { radius = 10.0; cosine = false }
            else -> throw IllegalArgumentException("postprocess: unsupported metric \"$metric\"")
        }

        val blob = Npy.loadNpz(npzPath)
        val labelArray = require(blob, "labels", npzPath)
        val centerArray = require(blob, "centers", npzPath)
        val maxArray = require(blob, "max_distance", npzPath)

        labels = labelArray.asUnicode()
        require(centerArray.shape.size == 2 && centerArray.shape[0] == labels.size) {
            "postprocess: centers ${NdArray.describe(centerArray.shape)} does not align with " +
                "${labels.size} labels"
        }

        val dim = centerArray.shape[1]
        val flat = centerArray.asFloat32()
        centers = Array(labels.size) { i -> flat.copyOfRange(i * dim, (i + 1) * dim) }

        val maxima = maxArray.asFloat32()
        maxDistance = DoubleArray(labels.size) { maxima[it].toDouble() }
    }

    private fun require(blob: Map<String, NdArray>, key: String, path: String): NdArray =
        blob[key] ?: throw IllegalArgumentException("postprocess: $path has no '$key'")

    override fun apply(output: NdArray, context: Context): ModelResult {
        val embedding = output.asFloat32()
        check(centers.isNotEmpty()) { "postprocess: no centroids loaded" }
        require(embedding.size == centers[0].size) {
            "postprocess: embedding has ${embedding.size} dims, centroids have ${centers[0].size}"
        }

        var best = -1
        var bestDistance = Double.POSITIVE_INFINITY
        for (i in centers.indices) {
            val d = if (cosine) {
                Ops.cosineDistance(embedding, centers[i])
            } else {
                Ops.euclideanDistance(embedding, centers[i])
            }
            if (d > radius) {
                continue // outside the radius: not a neighbour at all
            }
            if (d < bestDistance) {
                best = i
                bestDistance = d
            }
        }

        if (best < 0) {
            return MetricResult("NONE", Double.POSITIVE_INFINITY, 0.0)
        }

        val threshold = maxDistance[best]
        return if (bestDistance < threshold) {
            MetricResult(labels[best], bestDistance, threshold)
        } else {
            MetricResult("NONE", bestDistance, threshold)
        }
    }
}

/**
 * A single sigmoid score against a threshold — `BinaryClassification`.
 *
 * **Not an argmax, and getting that wrong is silent.** These outputs have `Shape [1]` and two declared
 * labels: the value is P(second label), compared against `Threshold` (0.5 when the config omits it).
 * Feeding a one-element vector to an argmax returns index 0 every time, so every document came back as
 * the FIRST label — `FAKE` for the spoofing checks — with no error anywhere. Found by conformance in the
 * .NET port, which reported REAL vs FAKE on five of seven cases; nothing in the code would have shown it.
 *
 * The reported confidence is the score for the label that WON, not the raw output, so a below-threshold
 * result reports how strongly it was below.
 */
public class BinaryClass(
    private val labels: List<String>,
    private val threshold: Double,
) : Postprocessor {
    override fun apply(output: NdArray, context: Context): ModelResult {
        val scores = output.asFloat32()
        require(scores.isNotEmpty()) { "postprocess: empty binary score" }
        require(labels.size >= 2) {
            "postprocess: binary output needs 2 labels, got ${labels.size}"
        }
        val score = scores[0].toDouble()
        return if (score > threshold) {
            ClassResult(labels[1], score)
        } else {
            ClassResult(labels[0], 1 - score)
        }
    }
}

/** Argmax over a score vector, with the model's declared labels. */
public class MultiClass(output: ModelOutput) : Postprocessor {

    private val labels: List<String> = output.labelsAsStrings()

    init {
        require(labels.isNotEmpty()) { "postprocess: multiclass output declares no Labels" }
    }

    override fun apply(output: NdArray, context: Context): ModelResult {
        val scores = output.asFloat32()
        require(scores.isNotEmpty()) { "postprocess: empty score vector" }
        val index = Ops.argmax(scores)
        require(index < labels.size) {
            "postprocess: class $index has no label (only ${labels.size} declared)"
        }
        return ClassResult(labels[index], Ops.max(scores))
    }
}
