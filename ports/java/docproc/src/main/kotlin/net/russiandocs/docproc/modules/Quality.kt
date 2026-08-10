package net.russiandocs.docproc.modules

import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Crop
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Interpolation
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.models.Loader
import net.russiandocs.docproc.models.Model
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.postprocess.ClassResult
import java.io.File

/** One tile's classification. */
internal data class TileVerdict(val label: String, val confidence: Double)

/** Shared tiling for the two tile-based quality checks. */
internal object Tiles {

    private const val WINDOW_SIZE = 128

    /**
     * Resizes to a whole number of tiles and classifies each one.
     *
     * **The colour conversion looks like a bug and is not.** The reference calls
     * `cvtColor(COLOR_BGR2RGB)` on an image that is ALREADY RGB, so what actually happens is RGB to BGR —
     * the quality classifiers see BGR. Reproduced exactly: "fixing" it would change every verdict these
     * two models produce.
     *
     * The iteration order is x-outer, y-inner, matching the reference. It does not affect the aggregates
     * below, which are order-independent, but keeping it means a future per-tile stage compares without a
     * reordering step.
     */
    internal fun classify(model: Model, image: Image, canvasX: Int, canvasY: Int): List<TileVerdict> {
        Io.resize(image, canvasX * WINDOW_SIZE, canvasY * WINDOW_SIZE, Interpolation.LINEAR)
            .use { canvas ->
                Io.toBgr(canvas).use { swapped ->
                    val verdicts = ArrayList<TileVerdict>(canvasX * canvasY)
                    for (xStep in 0 until canvasX) {
                        for (yStep in 0 until canvasY) {
                            val x = WINDOW_SIZE * xStep
                            val y = WINDOW_SIZE * yStep
                            Crop.clampedCrop(swapped, x, y, x + WINDOW_SIZE, y + WINDOW_SIZE)
                                .use { tile ->
                                    val result = model.predict(tile)
                                    val cls = result[0] as? ClassResult
                                        ?: throw IllegalStateException(
                                            "modules: quality tile output is " +
                                                "${result[0]::class.simpleName}, want ClassResult")
                                    verdicts += TileVerdict(cls.label, cls.confidence)
                                }
                        }
                    }
                    return verdicts
                }
            }
    }
}

/** Glare detection over a 7x4 tile grid. */
public class Glare(
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
) : AutoCloseable {

    private val model: Model = Loader.load(
        File(ModelPaths.resolve(root, paths, "Glare"), "ONNX").path, device, threads, root)

    public fun predict(image: Image): Pair<String, Double> {
        val tiles = Tiles.classify(model, image, CANVAS_X, CANVAS_Y)
        require(tiles.isNotEmpty()) { "modules: Glare classified no tiles" }

        // Counts CLEAN tiles, so `score` is the fraction that are glared. Written as the reference writes
        // it — adding 0 for a glared tile and 1 otherwise — rather than collapsed into a count of glared
        // tiles, because the two differ if a third label ever appears.
        var sum = 0.0
        for (tile in tiles) {
            sum += if (tile.label == "GLARE" && tile.confidence > CONFIDENCE_GATE) 0.0 else 1.0
        }
        val score = 1 - sum / tiles.size
        return (if (score > 0) "bad" else "good") to score
    }

    override fun close(): Unit = model.close()

    private companion object {
        const val CANVAS_X = 7
        const val CANVAS_Y = 4

        /**
         * A tile counts as glared only ABOVE this confidence. Below it the tile is treated as clean, which
         * is why a low-confidence GLARE verdict does not condemn the document.
         */
        const val CONFIDENCE_GATE = 0.85
    }
}

/** Blur detection over a 7x4 tile grid. */
public class Blur(
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
) : AutoCloseable {

    private val model: Model = Loader.load(
        File(ModelPaths.resolve(root, paths, "Blur"), "ONNX").path, device, threads, root)

    public fun predict(image: Image): Pair<String, Double> {
        val tiles = Tiles.classify(model, image, CANVAS_X, CANVAS_Y)

        // **Only three of the five labels count, and the others are excluded from the DENOMINATOR too.**
        // A tile the model calls something else is not a vote for sharpness, it simply does not vote —
        // which is a different aggregate from treating it as 0.
        var sum = 0.0
        var counted = 0
        for (tile in tiles) {
            when (tile.label) {
                "Blur5" -> { sum += 0.5; counted++ }
                "Blur10" -> { sum += 1.0; counted++ }
                "NonBlur" -> counted++
            }
        }

        // No countable tiles returns "sharp", not a division by zero. Rejecting a document because the
        // classifier had nothing to say about any tile would be a false negative.
        if (counted == 0) {
            return "good" to 1.0
        }

        val score = 1 - sum / counted
        return (if (score > GATE) "good" else "bad") to score
    }

    override fun close(): Unit = model.close()

    private companion object {
        const val CANVAS_X = 7
        const val CANVAS_Y = 4
        const val GATE = 0.9
    }
}

/**
 * Print and LCD spoofing. One type, two instances — they differ only in the model and the gate.
 *
 * The gate is the interesting part: `PrintSpoofing` applies a 0.9 threshold ON TOP of the model's own
 * decision, so a REAL verdict below that confidence becomes FAKE. `LCDSpoofing` passes 0 and takes the
 * model's word. Both are reproduced as-is; the asymmetry is in the reference.
 */
public class Spoofing private constructor(
    public val name: String,
    private val gate: Double,
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
) : AutoCloseable {

    private val model: Model = Loader.load(
        File(ModelPaths.resolve(root, paths, name), "ONNX").path, device, threads, root)

    public fun predict(image: Image): Pair<String, Double> {
        val result = model.predict(image)
        val cls = result[0] as? ClassResult
            ?: throw IllegalStateException(
                "modules: $name output is ${result[0]::class.simpleName}, want ClassResult")
        return if (gate > 0 && cls.confidence < gate) {
            "FAKE" to cls.confidence
        } else {
            cls.label to cls.confidence
        }
    }

    override fun close(): Unit = model.close()

    public companion object {
        public fun print(
            root: String,
            paths: Map<String, String>,
            device: Device,
            threads: Int,
        ): Spoofing = Spoofing("PrintSpoofing", 0.9, root, paths, device, threads)

        public fun lcd(
            root: String,
            paths: Map<String, String>,
            device: Device,
            threads: Int,
        ): Spoofing = Spoofing("LCDSpoofing", 0.0, root, paths, device, threads)
    }
}
