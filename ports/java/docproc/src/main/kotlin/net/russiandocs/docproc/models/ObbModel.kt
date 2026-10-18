package net.russiandocs.docproc.models

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.inference.Session
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.postprocess.ObbBox
import net.russiandocs.docproc.postprocess.Postprocessor
import net.russiandocs.docproc.postprocess.YoloObbDetector
import net.russiandocs.docproc.preprocess.Preprocessor

/**
 * An oriented-box detection model: one YOLOOBB input, one oriented detector output. Port of
 * `processing/models.py::YOLOOBBDetectionModel` (`ModelType: "YOLOOBBDetection"`).
 *
 * Built directly by the modules that use it, as [DetectionModel] is: the `ModelType` switch of [Loader] covers
 * the single-session `UnifiedModel`, and a detector wrapper is chosen by the module that knows what it asked for.
 */
public class ObbDetectionModel(
    dir: String,
    device: Device,
    threads: Int,
    root: String? = null,
) : AutoCloseable {

    public val config: ModelConfig = ModelConfig.load(dir)
    private val pre: Preprocessor = Loader.newPreprocessor(config.inputs[0])
    private val detector: YoloObbDetector
    private val session: Session

    init {
        val post: Postprocessor = Loader.newPostprocessor(config.outputs[0], config.dir, root)
        detector = post as? YoloObbDetector
            ?: throw IllegalArgumentException(
                "models: ${config.name} output 0 is not an oriented detector (${config.outputs[0].type})")
        session = Session(config.modelPath, device, threads)
    }

    /** Oriented detections in the coordinates of [image], reading order (top to bottom, then left to right). */
    public fun predict(image: Image): List<ObbBox> {
        val (tensor, meta) = pre.apply(image)
        val raw = session.run(listOf(tensor))
        return detector.decode(raw[0], contextOf(meta, resize = true))
    }

    override fun close(): Unit = session.close()
}
