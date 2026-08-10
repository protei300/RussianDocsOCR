package net.russiandocs.docproc.models

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.inference.Session
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.postprocess.Box
import net.russiandocs.docproc.postprocess.Context
import net.russiandocs.docproc.postprocess.Postprocessor
import net.russiandocs.docproc.postprocess.YoloDetector
import net.russiandocs.docproc.postprocess.YoloSegmentor
import net.russiandocs.docproc.preprocess.Meta
import net.russiandocs.docproc.preprocess.Preprocessor

/** Turns a preprocessor's [Meta] into the postprocessor's [Context]. */
internal fun contextOf(meta: Meta, resize: Boolean): Context = Context(
    ratio = meta.ratio,
    padExtra = meta.padExtra,
    padLetter = meta.padLetter,
    paddedH = meta.paddedH,
    paddedW = meta.paddedW,
    origH = meta.origH,
    origW = meta.origW,
    resize = resize,
)

/** A detection model: one YOLO input, one detector output. */
public class DetectionModel(
    dir: String,
    device: Device,
    threads: Int,
    root: String? = null,
) : AutoCloseable {

    public val config: ModelConfig = ModelConfig.load(dir)
    private val pre: Preprocessor = Loader.newPreprocessor(config.inputs[0])
    private val detector: YoloDetector
    private val session: Session

    init {
        val post: Postprocessor = Loader.newPostprocessor(config.outputs[0], config.dir, root)
        detector = post as? YoloDetector
            ?: throw IllegalArgumentException(
                "models: ${config.name} output 0 is not a detector (${config.outputs[0].type})")
        session = Session(config.modelPath, device, threads)
    }

    public fun predict(image: Image): List<Box> {
        val (tensor, meta) = pre.apply(image)
        val raw = session.run(listOf(tensor))
        return detector.decode(raw[0], contextOf(meta, resize = true))
    }

    override fun close(): Unit = session.close()
}

/** A segmentation model: one YOLO input, a detector output and a proto-mask output. */
public class SegmentationModel(
    dir: String,
    device: Device,
    threads: Int,
    root: String? = null,
) : AutoCloseable {

    public val config: ModelConfig = ModelConfig.load(dir)
    private val pre: Preprocessor
    private val detector: YoloDetector
    private val segmentor: YoloSegmentor
    private val session: Session

    init {
        require(config.outputs.size == 2) {
            "models: ${config.name} expects 2 outputs (boxes, proto), got ${config.outputs.size}"
        }
        pre = Loader.newPreprocessor(config.inputs[0])

        // numpyOnly: the segmentation path needs RAW float coordinates and the mask coefficients.
        // Truncating and labelling first would discard the sub-pixel information the mask crop depends on.
        // The copy keeps the loader switch as the single construction site.
        val head = Loader.newPostprocessor(config.outputs[0], config.dir, root) as? YoloDetector
            ?: throw IllegalArgumentException(
                "models: ${config.name} output 0 is not a detector (${config.outputs[0].type})")
        detector = head.withNumpyOnly()

        segmentor = Loader.newPostprocessor(config.outputs[1], config.dir, root) as? YoloSegmentor
            ?: throw IllegalArgumentException(
                "models: ${config.name} output 1 is not a segmentor (${config.outputs[1].type})")
        session = Session(config.modelPath, device, threads)
    }

    /**
     * Detects and segments.
     *
     * The segmentor is handed `paddedH`/`paddedW` — the size BEFORE the letterbox but AFTER any extra
     * padding — because that is the space the mask is unpadded into. Passing the original dimensions
     * instead produces a mask that is subtly the wrong shape.
     */
    public fun predict(image: Image): Pair<List<Box>, List<List<Pt>>> {
        val (tensor, meta) = pre.apply(image)
        val raw = session.run(listOf(tensor))

        val boxes = detector.decode(raw[0], contextOf(meta, resize = true))
        if (boxes.isEmpty()) {
            return emptyList<Box>() to emptyList()
        }
        val segments = segmentor.segment(raw[1], boxes, meta.padExtra, meta.paddedH, meta.paddedW)
        return boxes to segments
    }

    override fun close(): Unit = session.close()
}
