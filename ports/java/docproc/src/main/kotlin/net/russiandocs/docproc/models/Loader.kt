package net.russiandocs.docproc.models

import net.russiandocs.docproc.config.Alphabets
import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.inference.Session
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.postprocess.BinaryClass
import net.russiandocs.docproc.postprocess.Context
import net.russiandocs.docproc.postprocess.Metric
import net.russiandocs.docproc.postprocess.ModelResult
import net.russiandocs.docproc.postprocess.MultiClass
import net.russiandocs.docproc.postprocess.OcrProbs
import net.russiandocs.docproc.postprocess.NmsMode
import net.russiandocs.docproc.postprocess.NotImplementedPostprocessor
import net.russiandocs.docproc.postprocess.Postprocessor
import net.russiandocs.docproc.postprocess.YoloDetector
import net.russiandocs.docproc.postprocess.YoloSegmentor
import net.russiandocs.docproc.preprocess.Classification
import net.russiandocs.docproc.preprocess.NotImplementedPreprocessor
import net.russiandocs.docproc.preprocess.Preprocessor
import net.russiandocs.docproc.preprocess.OcrV2
import net.russiandocs.docproc.preprocess.Yolo
import net.russiandocs.docproc.tensors.NdArray
import java.io.File

/**
 * Builds a runnable model from its `model.json`.
 *
 * **The three `when` blocks below are the most portable code in the project, and they must stay that
 * way.** No reflection, no annotations, no DI container, no self-registering initialisers: one `when` per
 * factory, one construction expression per case, cases in the same ORDER as the reference's `match`
 * statements (recorded in MAPPING.md). A dispatch table built by scanning the classpath is more idiomatic
 * on the JVM and would make the correspondence with the other three languages unverifiable.
 *
 * An unknown tag is an ERROR naming the tag (D-06). The reference falls through to `None` and turns a
 * typo into a null dereference three stages later.
 */
public object Loader {

    /** Preprocessor by the input's declared `Type`. Case order follows the reference. */
    public fun newPreprocessor(input: ModelInput): Preprocessor = when (input.type) {
        "Classification" -> Classification(input)
        // Recognised but not implemented yet — wired so a port cannot quietly grow a different
        // behaviour for them, and so the reader sees what exists rather than guessing.
        "YOLO" -> Yolo(input)
        "YOLOOBB" -> NotImplementedPreprocessor("YOLOOBB")
        "OCRv2" -> OcrV2(input)
        // "OCR" is the removed legacy 31x200 grayscale path. No shipped model.json declares it; wired
        // anyway, because an omitted case reads as an oversight.
        "OCR" -> NotImplementedPreprocessor("OCR")
        else -> throw IllegalArgumentException("models: unknown input type \"${input.type}\"")
    }

    /**
     * Postprocessor by the output's declared `Type`.
     *
     * @param root the repository root, needed once OCRProbs arrives: it resolves its ALLOWED charset from
     *   `config/ocr_alphabets.json`, which lives beside the library rather than beside the model.
     */
    public fun newPostprocessor(
        output: ModelOutput,
        dir: String,
        root: String? = null,
    ): Postprocessor = when (output.type) {
        // **The tag is MultiLabelClassification.** The .NET port guessed "MultiClass" and the D-06 error
        // naming the unknown tag is what turned a wrong guess into a one-line fix, instead of the
        // reference's fall-through to None and a null dereference two stages later. Read the artifacts;
        // do not name the tags from memory.
        //
        // BinaryClassification is a SIGMOID against a threshold; MultiLabelClassification is an argmax.
        // Routing the former to the latter silently returns the first label for every binary output —
        // see BinaryClass for what that cost.
        "BinaryClassification" -> BinaryClass(output.labelsAsStrings(), output.threshold ?: 0.5)
        "MultiLabelClassification" -> MultiClass(output)
        "Metric" -> Metric(
            File(dir, ModelPaths.normaliseSeparators(
                output.centers
                    ?: throw IllegalArgumentException("models: Metric output has no Centers"),
            )).path,
            output.metric ?: "cosine",
        )
        // Routed through the switch even though the detection and segmentation models know exactly what
        // they need. The Go port initially built these two by hand and only noticed at M6 that the dispatch
        // design MAPPING.md calls the portable core was being bypassed by precisely the models that use it
        // — and a fourth port would have copied that verbatim.
        "YOLODetector" -> YoloDetector(
            output.labelsAsStrings(), output.iou ?: 0.45, output.cls ?: 0.5,
            NmsMode.CLASS_AGNOSTIC,
        )
        "PerClassYOLODetector" -> YoloDetector(
            output.labelsAsStrings(), output.iou ?: 0.45, output.cls ?: 0.5,
            NmsMode.PER_CLASS,
        )
        "YOLOOBBDetector" -> NotImplementedPostprocessor("YOLOOBBDetector")
        "YOLOSegmentor" -> YoloSegmentor(output.maskFilter ?: 0.5)
        "OCRProbs" -> OcrProbs(
            output.alphabet
                ?: throw IllegalArgumentException(
                    "models: OCRProbs output declares no Alphabet"),
            // The ALLOWED subset comes from config/ocr_alphabets.json, which lives beside the library
            // rather than beside the model — hence the `root` parameter. Passing the model's own full
            // alphabet here instead would disable masking with no error at all.
            root?.let {
                Alphabets.allowedCharset(it, output.script ?: "cyrillic", output.country)
            },
            output.blankIndex ?: 0,
        )
        else -> throw IllegalArgumentException("models: unknown output type \"${output.type}\"")
    }

    /**
     * The model wrapper by `ModelType`.
     *
     * `UnifiedModel` covers everything the reference actually routes through it. The `"UnifedModel"`
     * spelling is deliberate: it is the typo in the shipped `DocTypeAngles/model.json`, which worked in
     * the reference only by falling through to a default. Accepting both is not politeness, it is the
     * difference between loading the shipped artifact and not.
     */
    public fun newModel(config: ModelConfig, device: Device, threads: Int, root: String? = null): Model =
        when (config.modelType) {
            "UnifiedModel", "UnifedModel", "" -> Model(config, device, threads, root)
            else -> throw IllegalArgumentException(
                "models: unknown ModelType \"${config.modelType}\"")
        }

    /** Loads the config from a directory and builds the model. */
    public fun load(dir: String, device: Device, threads: Int, root: String? = null): Model =
        newModel(ModelConfig.load(dir), device, threads, root)
}

/** A model: one session, one preprocessor per input, one postprocessor per output. */
public class Model internal constructor(
    public val config: ModelConfig,
    device: Device,
    threads: Int,
    root: String? = null,
) : AutoCloseable {

    private val pre: List<Preprocessor> = config.inputs.map { Loader.newPreprocessor(it) }
    private val post: List<Postprocessor> =
        config.outputs.map { Loader.newPostprocessor(it, config.dir, root) }
    private val session = Session(config.modelPath, device, threads)

    /**
     * Runs the model over one image and postprocesses every output.
     *
     * Outputs are returned POSITIONALLY, matched to `outputs[i]`. The session collects them by declared
     * name, so the position here is the model's declaration order rather than whatever order ONNX Runtime
     * happened to return — which matters for `DocTypeAngles`, where swapping two heads would silently
     * produce a document type from an angle vector.
     */
    public fun predict(image: Image): List<ModelResult> {
        // Single-input models only, which is every shipped artifact. A second input would need its own
        // preprocessor and a decision about which image feeds it; refusing beats guessing.
        require(pre.size == 1) {
            "models: ${config.name} declares ${pre.size} inputs, only 1 is supported"
        }

        val (tensor, meta) = pre[0].apply(image)
        val raw = session.run(listOf(tensor))

        require(raw.size == post.size) {
            "models: ${config.name} returned ${raw.size} outputs, config declares ${post.size}"
        }

        val context = Context(
            ratio = meta.ratio,
            padExtra = meta.padExtra,
            padLetter = meta.padLetter,
            paddedH = meta.paddedH,
            paddedW = meta.paddedW,
            origH = meta.origH,
            origW = meta.origW,
        )
        return raw.mapIndexed { i, output -> post[i].apply(output, context) }
    }

    /** Runs the model over a tensor the caller prepared. Used by the batched paths. */
    public fun predictTensor(tensor: NdArray, context: Context): List<ModelResult> {
        val raw = session.run(listOf(tensor))
        require(raw.size == post.size) {
            "models: ${config.name} returned ${raw.size} outputs, config declares ${post.size}"
        }
        return raw.mapIndexed { i, output -> post[i].apply(output, context) }
    }

    override fun close(): Unit = session.close()
}
