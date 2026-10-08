package net.russiandocs.docproc.pipeline

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Crop
import net.russiandocs.docproc.imaging.Geometry
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.imaging.Placement
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.imaging.StackDirection
import net.russiandocs.docproc.modules.BordersResult
import net.russiandocs.docproc.modules.DetectedDocument
import net.russiandocs.docproc.modules.DocBox
import net.russiandocs.docproc.modules.DocumentDetector
import net.russiandocs.docproc.modules.DocDetector
import net.russiandocs.docproc.modules.DocDeskewer
import net.russiandocs.docproc.modules.DocTypeAngles
import net.russiandocs.docproc.modules.Blur
import net.russiandocs.docproc.modules.DocTypeResult
import net.russiandocs.docproc.modules.Field
import net.russiandocs.docproc.modules.Glare
import net.russiandocs.docproc.modules.OcrEngine
import net.russiandocs.docproc.modules.PageRegistrar
import net.russiandocs.docproc.modules.PageRegistration
import net.russiandocs.docproc.modules.TextFieldsDetector
import net.russiandocs.docproc.modules.WordsDetector
import net.russiandocs.docproc.modules.closeAllFields
import net.russiandocs.docproc.modules.Spoofing
import net.russiandocs.docproc.geometry.Chain
import net.russiandocs.docproc.geometry.FieldFrame
import net.russiandocs.docproc.geometry.Offset
import net.russiandocs.docproc.geometry.PointMap
import net.russiandocs.docproc.geometry.QuarterTurns
import net.russiandocs.docproc.geometry.Scale
import net.russiandocs.docproc.geometry.Unknown
import net.russiandocs.docproc.modules.Homography as PageHomography
import net.russiandocs.docproc.geometry.Homography
import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point as CvPoint
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import net.russiandocs.docproc.postprocess.Box
import net.russiandocs.docproc.tensors.PyNum
import net.russiandocs.docproc.viewmodel.Builder
import net.russiandocs.docproc.viewmodel.Input
import net.russiandocs.docproc.viewmodel.Payload
import net.russiandocs.docproc.viewmodel.RawBox
import net.russiandocs.docproc.tensors.Ops
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/** Which device the detectors run on. String-valued on the wire, as every port reports it. */
public enum class Device(public val wire: String) {
    CPU("cpu"),
    GPU("gpu"),
    ;

    public companion object {
        public fun parse(value: String): Device = when (value) {
            "cpu" -> CPU
            "gpu" -> GPU
            else -> throw IllegalArgumentException("device must be cpu or gpu, got $value")
        }
    }
}

/** The OCR engine tier. `legacy` was removed in 3.0.0 and raises in the reference. */
public enum class OcrTier(public val wire: String) {
    ACCURATE("accurate"),
    FAST("fast"),
    ;

    public companion object {
        public fun parse(value: String): OcrTier = when (value) {
            "accurate" -> ACCURATE
            "fast" -> FAST
            "legacy" -> throw IllegalArgumentException(
                "ocr='legacy' was removed in 3.0.0: the legacy engines were measurably worse " +
                    "(mean CER 0.123 against 0.062) and are gone from the artifacts",
            )
            else -> throw IllegalArgumentException("ocr must be accurate or fast, got $value")
        }
    }
}

/** Per-run knobs. A type rather than a parameter list, so the call sites match across ports. */
public data class RunOptions(
    val docconf: Double = 0.5,
    val imgSize: Int = 1500,
    val sink: StageSink = NullStageSink,
    /** Stops AFTER the named stage, inclusive. Null runs everything implemented. */
    val upTo: String? = null,
    val includeDebug: Boolean = false,
)

/**
 * Everything one run produced.
 *
 * [AutoCloseable] because it owns images, and it owns EVERY intermediate rather than only the canvas. The
 * Go port's leak was exactly this: a run's images left to the collector, 12.7 MB per document, unbounded,
 * with the conformance suite green throughout — the CLI runs one document per process, so nothing in the
 * harness can see it.
 */
public class Results : AutoCloseable {

    public var docType: String = "NONE"
        internal set

    public var docConfidence: Double = 0.0
        internal set

    public var angle: Int = 0
        internal set

    public var angleConfidence: Double = 0.0
        internal set

    public var device: String = "cpu"
        internal set

    public var timings: MutableMap<String, Double> = LinkedHashMap()
        internal set

    /**
     * The quality verdicts. Heterogeneous by contract: 'good'/'bad' for glare and blur, 'REAL'/'FAKE'
     * for the spoofing checks, and DocConf a number rendered as text until it reaches the wire.
     */
    public var quality: Map<String, String> = emptyMap()
        internal set

    /** The joined OCR values, field name to text. */
    public var ocr: MutableMap<String, String> = LinkedHashMap()
        internal set

    /** Every detected text-field box, in reading order. */
    public var boxes: MutableList<RawBox> = mutableListOf()
        internal set

    /** Per-field word lists, kept for callers that want the pre-join tokens. */
    public var words: List<FieldText> = emptyList()
        internal set

    /** Lines read WHOLE because the word split lost most of them — `PipelineResults.words_fallback`. */
    public var wordsFallback: List<WordsFallback> = emptyList()
        internal set

    /** Lines the gap guard declined to re-read because they carry no ink — `PipelineResults.words_no_ink`. */
    public var wordsNoInk: List<WordsNoInk> = emptyList()
        internal set

    /**
     * Every document the detector found in the frame, largest first, boxes on the INPUT photo —
     * `PipelineResults.documents`. Empty when the detector found nothing (or is switched off or has no
     * weights) and the whole frame was read.
     */
    public var documents: List<DetectedDocument> = emptyList()
        internal set

    /** Which entry of [documents] this run read; null — the whole frame. `document_index`. */
    public var documentIndex: Int? = null
        internal set

    /** The box of the document read, on the input photo; null — the whole frame. `document_box`. */
    public val documentBox: DocBox?
        get() = documentIndex?.let { documents.getOrNull(it)?.box }

    /** Date fields whose word-by-word reading was replaced by a whole-line one — `DatesReadWhole`. */
    public var datesReadWhole: List<DateReread> = emptyList()
        internal set

    /** Canonical `dd.mm.yyyy` per date field that converted — `PipelineResults.ocr_normalized`. */
    public var ocrNormalized: Map<String, String> = emptyMap()
        internal set

    /**
     * The leasing flag read from the STS special marks — `PipelineResults.leasing`: `{"leasing": true}` or null
     * (no leasing in the marks, or not an STS). Only the flag is reported (`Pipeline.LEASING_REPORTED`): the
     * lessor and the contract parse (see [StsMarks.parseLeasing]) but are not reported until the reading of the
     * small print can carry them. A SEPARATE view, like [ocrNormalized]: `ocr["Special_marks"]` keeps the text
     * as read.
     */
    public var leasing: Map<String, Boolean>? = null
        internal set

    /**
     * Index of the other side of this document in [Recognizer.runFrame]'s list — `paired_with`; null: no pair
     * was found, or a single-document call. See [PairSides].
     */
    public var pairedWith: Int? = null
        internal set

    /**
     * Map from the canvas later stages read back to the image passed to `run` — `PipelineResults.geometry`
     * (geometry.py): the crop of the document, the resize, the quarter turns, the border warp, the registrar's
     * straightening and the deskew, in the order they ran. Null before the image is prepared.
     */
    public var geometry: Chain? = null
        internal set

    /** The `quads` of the run: where the read fields and words lie on the input image. Null before the split. */
    public var quads: FieldQuads? = null
        internal set

    /**
     * Points of the canvas, on the image passed to `run` — `PipelineResults.to_input`. Null when a stage of this run
     * changed the image in a way no point map expresses ([Unknown]): the canvas and what was read are fine, the way
     * back is not known.
     */
    public fun toInput(points: List<Pt>): List<Pt>? = geometry?.toInput(points) ?: if (geometry == null) points else null

    /**
     * Where the read text fields lie on the input image — `field_quads`: label -> one quadrilateral per detection,
     * top to bottom. Null when the way back is not known for this run.
     */
    public val fieldQuads: Map<String, List<List<Pt>>>? get() = quads?.fields

    /** label -> the quadrilateral of each word patch, same order as the field's words — `word_quads`. */
    public val wordQuads: Map<String, List<List<Pt>>>? get() = quads?.words

    /** The selected border contours, or null when the model found none. Compared under R-01. */
    public var segments: List<List<Pt>>? = null
        internal set

    /**
     * The corrected canvas, RGB. Null when the document short-circuited as unrecognised.
     *
     * Owned by this instance until [takeCanvas] is called.
     */
    public var canvas: Image? = null
        internal set

    /** Everything else the run allocated, released together in [close]. */
    private val owned = mutableListOf<Image>()

    internal fun own(image: Image): Image {
        owned += image
        return image
    }

    /**
     * Detaches the canvas and hands ownership over, releasing everything else.
     *
     * The one image that must outlive a run is the canvas the service stores; every intermediate must not.
     * Reading the field and returning is what leaked in Go — 663 MB to 4018 MB across 230 documents — so
     * this is the sanctioned way out. The canvas is REMOVED from the owned list before closing, or this
     * would hand back an image it has just released.
     */
    public fun takeCanvas(): Image? {
        val taken = canvas
        canvas = null
        if (taken != null) {
            owned.remove(taken)
        }
        close()
        return taken
    }

    override fun close() {
        canvas?.let { if (!owned.contains(it)) it.close() }
        canvas = null
        for (image in owned) {
            image.close()
        }
        owned.clear()
    }
}

/**
 * The pipeline. Port of `Pipeline` in `document_processing/pipeline/pipeline.py`.
 *
 * Stage coverage grows one milestone at a time, and [STAGES_IMPLEMENTED] must list exactly what this emits
 * and never more: the checker skips what is not claimed, so an over-claiming list turns a missing stage
 * into a confusing failure while an under-claiming one silently stops grading finished work.
 */
public class Recognizer(
    private val device: Device = Device.CPU,
    private val intraOpThreads: Int = 1,
    private val ocrTier: OcrTier = OcrTier.ACCURATE,
    /**
     * `Pipeline(detect_documents=True)`, the reference's default since decision №142: find the documents in
     * the frame first and read the crop of the largest one, cut at full resolution. False — or a weight set
     * without `DocDetect` — reads the whole frame, as before.
     */
    detectDocuments: Boolean = true,
) : AutoCloseable {

    /** `Pipeline.document_detector`: null when switched off or the weight set has no `DocDetect`. */
    private val documentDetector: DocumentDetector?
    private val docTypeAngles: DocTypeAngles
    private val glare: Glare
    private val blur: Blur
    private val printSpoofing: Spoofing
    private val lcdSpoofing: Spoofing
    private val docDetector: DocDetector
    private val deskewer: DocDeskewer
    private val textFields: TextFieldsDetector
    private val words: WordsDetector
    private val cyrillic: OcrEngine
    private val latin: OcrEngine

    /**
     * `Pipeline.page_registrar` — `PageRegistrar()` with the reference's own defaults
     * (`page_registration=True` since 2026-09-17). Pure OpenCV (SIFT + the `templates/` PNGs), no ONNX
     * session, so unlike the fields above it needs no `device`/`threads`.
     */
    private val pageRegistrar: PageRegistrar

    /**
     * `Pipeline._card_registrars`: one template registrar per STS type, built on first use (SIFT features of
     * the templates cost ~0.3 s a type); null for a type with no templates. Guarded, because the pool hands one
     * instance to one thread at a time but the map outlives a run.
     */
    private val cardRegistrars = HashMap<String, PageRegistrar?>()

    init {
        // SLOW — 215 MB of weights and one session each — so construct once and keep the instance. The
        // reference loads them eagerly in its constructor for the same reason, and the service wraps the
        // whole thing in a pool of exactly one.
        val root = ModelPaths.root()
        val paths = ModelPaths.load(root)
        documentDetector = if (detectDocuments) DocumentDetector.openOrNull(root, paths, device, intraOpThreads) else null
        docTypeAngles =DocTypeAngles(root, paths, device, intraOpThreads)
        glare = Glare(root, paths, device, intraOpThreads)
        blur = Blur(root, paths, device, intraOpThreads)
        printSpoofing = Spoofing.print(root, paths, device, intraOpThreads)
        lcdSpoofing = Spoofing.lcd(root, paths, device, intraOpThreads)
        docDetector = DocDetector(root, paths, device, intraOpThreads)
        deskewer = DocDeskewer.forPipeline()
        textFields = TextFieldsDetector(root, paths, device, intraOpThreads)
        words = WordsDetector(root, paths, device, intraOpThreads)
        pageRegistrar = PageRegistrar()

        // **OCR stays on the CPU even when the detectors are on the GPU.** Measured, not assumed:
        // per-word dynamic widths make the CUDA provider recompile the graph on every distinct
        // width, and the Go port measured the whole corpus 13.7x SLOWER on GPU than on CPU. The
        // reference pins ocr_device to cpu for the same reason.
        cyrillic = OcrEngine.cyrillic(root, paths, Device.CPU, intraOpThreads, ocrTier)
        latin = OcrEngine.latin(root, paths, Device.CPU, intraOpThreads, ocrTier)
    }

    /**
     * Runs the pipeline over one file. Port of `Pipeline.process_img`: with documents found in the frame it
     * reads the LARGEST one ([Results.documents] lists all of them, [Results.documentIndex] is the one read).
     */
    public fun run(imagePath: String, options: RunOptions): Results {
        val frame = Io.loadRgb(imagePath)
        try {
            // ---- stage: documents (decision №142) -----------------------------------------------
            //
            // Documents first: the frame is searched at the processing size, the boxes are carried back to
            // the input photo, and the document is CUT THERE, at full resolution plus a margin. Everything
            // after this point reads that crop.
            val head = newResults()
            val documents = findDocuments(frame, options, head)
            head.documents = documents
            if (options.upTo == "documents") {
                return head
            }
            return readDocument(frame, documents, if (documents.isEmpty()) null else 0, options, head)
        } finally {
            frame.close()
        }
    }

    /**
     * Reads EVERY document in the frame, not only the largest. Port of `Pipeline.process_frame` (issue #26).
     *
     * One [Results] per document, largest first; with no document found, a one-element list holding the
     * whole-frame reading — exactly what [run] returns. Two sides of one document lying in the same frame are
     * paired by the number printed on both ([PairSides]): [Results.pairedWith] is the index of the other side in
     * the returned list. Each element is a separate [Results] the caller closes. The `documents` timing belongs
     * to the first result only, as in the reference (every later one starts from fresh results).
     */
    public fun runFrame(imagePath: String, options: RunOptions): List<Results> {
        val frame = Io.loadRgb(imagePath)
        try {
            val head = newResults()
            val documents = findDocuments(frame, options, head)
            head.documents = documents
            if (options.upTo == "documents") {
                return listOf(head)
            }
            if (documents.isEmpty()) {
                return listOf(readDocument(frame, documents, null, options, head))
            }
            val out = ArrayList<Results>()
            try {
                for (index in documents.indices) {
                    val results = if (index == 0) head else newResults().also { it.documents = documents }
                    out += readDocument(frame, documents, index, options, results)
                }
                PairSides.pair(out)
                return out
            } catch (e: Throwable) {
                out.forEach { it.close() }
                throw e
            }
        } finally {
            frame.close()
        }
    }

    private fun newResults(): Results = Results().also { it.device = device.wire }

    /**
     * Prepares the crop of `documents[index]` (the whole frame when [index] is null) and reads it.
     * `Pipeline._read_document` + `_process_document`. Closes [results] when it throws.
     */
    private fun readDocument(
        frame: Image,
        documents: List<DetectedDocument>,
        index: Int?,
        options: RunOptions,
        results: Results,
    ): Results {
        try {
            val crop = if (index != null) documentCrop(frame, documents[index].box) else null

            // Two steps rather than one, because the reference's `_prepare_image` is two and the second
            // only ever SHRINKS. Fusing them would also hide the floor-division trap that makes 2999x1777
            // come out at 1499 rather than 1500.
            //
            // A crop with no area (a degenerate detector box) cannot be read: the reference would hand an
            // empty array to `cv2.resize` and crash, here the whole frame is read instead.
            val cut = crop?.let { (x0, y0, x1, y1) ->
                if (x1 > x0 && y1 > y0) results.own(Crop.clampedCrop(frame, x0, y0, x1, y1)) else null
            }
            results.documentIndex = if (cut != null) index else null
            val prepared = results.own(Io.fitToLongestSide(cut ?: frame, options.imgSize))
            // The first maps of the canvas every later stage reads (geometry.py): the crop of the document, then the
            // resize (`_prepare_image`).
            val source = cut ?: frame
            var geometry = Chain(
                listOfNotNull(
                    if (cut != null) Offset(-crop!![0].toDouble(), -crop[1].toDouble()) else null,
                    Scale(prepared.width.toDouble() / source.width, prepared.height.toDouble() / source.height),
                ),
            )
            results.geometry = geometry

            options.sink.emitImage("prepare", prepared)
            if (options.upTo == "prepare") {
                results.canvas = prepared
                return results
            }

            // ---- stages: doctype.label, rotate ------------------------------------------------
            val started = System.nanoTime()
            val (meta0, upright0) = docTypeAngles.predictTransform(prepared)
            var meta = meta0
            var upright = results.own(upright0)
            results.timings[TIMING_DOCTYPE_ANGLE] = (System.nanoTime() - started) / 1e9

            results.docType = meta.docType
            results.docConfidence = meta.docTypeConfidence
            results.angle = meta.angle
            results.angleConfidence = meta.angleConfidence
            geometry = geometry.then(QuarterTurns(prepared.width, prepared.height, meta.angle / 90))
            results.geometry = geometry

            options.sink.emit("doctype.label", encode(meta))
            options.sink.emitImage("rotate", upright)
            if (options.upTo == "rotate") {
                results.canvas = upright
                return results
            }

            // ---- NONE: borders-first fallback, then the short return -----------------------------
            //
            // A hard but legitimate shot (strong perspective, small document scale, occluded header) can push
            // the raw-frame embedding past the metric NONE-threshold while the same classifier reads the
            // border-cropped document confidently (`_process_document`, measured 2026-08-02: 6/6 synthetic NONE
            // cases recovered at 0.98+, no new false accepts). Costs one extra border detection on NONE frames
            // only. Whatever the fallback finds, `doctype.label` and `rotate` above were already emitted with
            // the first answer, as in the reference.
            var noneCanvas: Image? = null
            if (meta.docType == "NONE") {
                val first = docDetector.predictTransformFull(upright, 1)   // the type is unknown: one page
                noneCanvas = results.own(first.canvas)
                first.pages.forEach { results.own(it) }
                if (!first.segments.isNullOrEmpty()) {
                    val (again, rotatedAgain) = docTypeAngles.predictTransform(first.canvas)
                    results.own(rotatedAgain)
                    meta = again
                    results.docType = again.docType
                    results.docConfidence = again.docTypeConfidence
                    results.angle = again.angle
                    results.angleConfidence = again.angleConfidence
                    if (again.docType.lowercase().contains("intpassport")) {
                        // The crop above ran while the type was still unknown (one page), so a passport spread
                        // lost its second page before OCR could see it. Now that the type is known, redo the
                        // border detection on the same pre-crop frame, which allows the two-page stitch. Type and
                        // angle stay as classified on the single page (measured reliable there); only the canvas
                        // is rebuilt, then turned by the angle that classification asked for.
                        val second = docDetector.predictTransformFull(upright, 2)
                        // Chain order is the order the pixels went: the second border canvas of the photo, then the
                        // turn the classification of the first one asked for (`_canvas` calls of the fallback).
                        geometry = geometry.then(second.geometry)
                            .then(QuarterTurns(second.canvas.width, second.canvas.height, again.angle / 90))
                        results.geometry = geometry
                        var turned: Image = results.own(second.canvas)
                        second.pages.forEach { results.own(it) }
                        repeat(again.angle / 90) {
                            turned = results.own(Io.rot90(turned, 1))
                        }
                        upright = turned
                    } else {
                        geometry = geometry.then(first.geometry)
                            .then(QuarterTurns(first.canvas.width, first.canvas.height, again.angle / 90))
                        results.geometry = geometry
                        upright = rotatedAgain
                    }
                }
            }
            if (meta.docType == "NONE") {
                // The reference prints a notice and returns the results as they stand: no quality, no borders.
                // The canvas it then reports is the border detector's when it ran (`img_with_fixed_perspective`).
                results.canvas = noneCanvas ?: upright
                finaliseTimings(results.timings)
                return results
            }

            // ---- stage: quality ---------------------------------------------------------------
            val quality = runQuality(upright, meta.docTypeConfidence, results.timings)
            results.quality = quality
            options.sink.emit("quality", encodeQuality(quality))
            if (options.upTo == "quality") {
                results.canvas = upright
                return results
            }

            // ---- stages: borders.segments, borders.canvas -------------------------------------
            //
            // max_pages is 2 only for the internal-passport spread; every other type passes 1, so a
            // background blob can never be stitched in.
            val maxPages = if (meta.docType.startsWith("INTPASSPORT") &&
                !meta.docType.contains("ADDR")) 2 else 1

            val bordersStart = System.nanoTime()
            val borders = docDetector.predictTransformFull(upright, maxPages)
            val canvas = results.own(borders.canvas)
            val segments = borders.segments
            results.segments = segments
            // Per-page pieces (empty unless a genuine two-page spread) outlive this stage — the field
            // detector reads them AFTER deskew, on the pre-deskew page pixels, exactly as the reference's
            // `_fields_from_pages` does (see the note at its call site below). Owned by `results` so a
            // thrown exception anywhere in between cannot leak them.
            borders.pages.forEach { results.own(it) }
            results.timings[TIMING_DOC_DETECTOR] = (System.nanoTime() - bordersStart) / 1e9

            // `_register_pages` (pipeline.py:1026) runs right after border detection, whenever
            // page_registration is on — which is the reference's DEFAULT since 2026-09-17, for every
            // document type. It early-returns for anything but a non-address internal passport, and the
            // reference's `_model_call` wrapper times the call regardless — so `timings` carries this key
            // for EVERY document, not only passports.
            val registerStart = System.nanoTime()
            val registered = registerPages(upright, meta.docType, borders)
            results.timings[TIMING_REGISTER_PAGES] = (System.nanoTime() - registerStart) / 1e9
            // `registered` may be a DIFFERENT BordersResult (new canvas, new pages) than `borders` — its
            // pieces are owned here too; the superseded ones stay in `results`' owned list and are simply
            // released a little later than strictly necessary, which is safe (one document per run) and
            // far simpler than unregistering a partial owner set mid-run.
            val effectiveCanvas: Image
            val effectiveSegments = segments
            if (registered !== borders) {
                effectiveCanvas = results.own(registered.canvas)
                registered.pages.forEach { results.own(it) }
            } else {
                effectiveCanvas = canvas
            }

            options.sink.emit("borders.segments", encodeSegments(effectiveSegments))
            options.sink.emitImage("borders.canvas", effectiveCanvas)
            if (options.upTo == "borders.canvas") {
                results.canvas = effectiveCanvas
                return results
            }

            // ---- stage: deskew.canvas ---------------------------------------------------------
            //
            // A canvas the registrar rebuilt is NOT deskewed (`_deskew`): its pages were already straightened by
            // their own lines (line_refine, blob-tolerant), and the projection-profile deskew on top of them took
            // the dark cushion around a small page for text and rotated a straight page by 6 degrees, garbling
            // the MRZ (conformance case 12_CR_INTPASSPORT_2011, measured 2026-09-17). The same holds for a card
            // straightened by its printed blank. The stage is still emitted, with the canvas as it stands.
            val deskewStart = System.nanoTime()
            // The map of the border canvas back to the image the border stage received (`_borders_geometry`): the
            // registrar's own when it rebuilt the canvas, else the border warp's followed by the deskew's rotation.
            val bordersGeometry: PointMap
            val deskewed: Image
            if (registered !== borders) {
                deskewed = effectiveCanvas
                bordersGeometry = registered.geometry
            } else {
                val (turned, _, rotation) = deskewer.deskewWithGeometry(effectiveCanvas)
                deskewed = results.own(turned)
                bordersGeometry = Chain(listOf(borders.geometry)).then(rotation)
            }
            geometry = geometry.then(bordersGeometry)
            results.geometry = geometry
            results.timings[TIMING_DESKEW] = (System.nanoTime() - deskewStart) / 1e9

            options.sink.emitImage("deskew.canvas", deskewed)
            if (options.upTo == "deskew.canvas") {
                results.canvas = deskewed
                return results
            }

            results.canvas = deskewed

            // ---- stage: fields.bbox -----------------------------------------------------------
            val ocrOptions = OcrOptions.forDocType(meta.docType)
            val fieldsStart = System.nanoTime()
            // `_fields_detector` (pipeline.py:1238): a genuine two-page spread (internal passport) is read
            // PAGE BY PAGE — the detector's 640x640 input gives a stitched spread only half of itself per
            // page — using the pages `_doc_detector`/`_register_pages` rectified, with each page's own boxes
            // moved onto the (deskewed) canvas by its `Placement`. Everything else keeps the single
            // whole-canvas call. The `placements.size == pages.size` check mirrors the reference's defensive
            // equal-length check, which is never false here since both come from the same call ([Geometry.
            // fixPerspective] or [registerPages]/[Geometry.stitchPages]).
            val fields: MutableList<Field> = if (registered.pages.size >= 2 && registered.placements.size == registered.pages.size) {
                fieldsFromPages(registered.pages, registered.placements, ocrOptions.needsLicenceRotation).toMutableList()
            } else {
                textFields.predictTransform(deskewed, ocrOptions.needsLicenceRotation).toMutableList().also {
                    // a patch is the canvas cut at its box (`TextFieldsDetector`)
                    for (f in it) {
                        f.cutMap = Offset(-f.box.x1.toInt().toDouble(), -f.box.y1.toInt().toDouble())
                    }
                    // `_read_margins`: the single-canvas path only - a field labelled tight to its letters is
                    // READ from a taller crop; its box, and the `fields.bbox` stage, stay as detected.
                    try {
                        ReadMargins.apply(it, deskewed, ocrOptions.readMargin)
                    } catch (e: Throwable) {
                        closeAllFields(it)
                        throw e
                    }
                }
            }
            results.timings[TIMING_FIELDS_DETECTOR] = (System.nanoTime() - fieldsStart) / 1e9
            results.boxes = fields.map { f ->
                RawBox(f.box.x1, f.box.y1, f.box.x2, f.box.y2, f.box.conf, f.box.cls, f.box.label)
            }.toMutableList()

            // `_note_mrz_zone` (pipeline.py:1245): built from the RAW detector boxes, right after field
            // detection and before splitting — nothing here rewrites `fields.bbox`, it only remembers
            // where each MRZ line was found so a wrong-length reading can be re-cropped later.
            val mrzZone = MrzZone.from(fields.map { it.box }, deskewed)

            try {
                options.sink.emit("fields.bbox", encodeBoxes(fields.map { it.box }))
                if (options.upTo == "fields.bbox") {
                    return results
                }

                // ---- stages: words.<Field>.bbox ---------------------------------------------
                // The address path (INTPASSPORTADDR) is out of scope for this port, so no
                // address.lines stage is emitted and the checker skips it.
                //
                // The BARE type, without the year suffix: the gap guard's SNILS exclusion, the OCR
                // parity rule and the date join all test it, and "SNILS_1996" would match none of them.
                val (bareType, docYear) = OcrOptions.splitDocType(meta.docType)

                val splitStart = System.nanoTime()
                val split = SplitWords.run(fields, ocrOptions, words, bareType)
                val fieldWords = split.fields
                results.wordsFallback = split.fallback
                results.wordsNoInk = split.noInk
                results.timings[TIMING_SPLIT_WORDS] = (System.nanoTime() - splitStart) / 1e9
                try {
                    for (fw in fieldWords) {
                        options.sink.emit("words.${fw.label}.bbox", encodeWordBoxes(fw.wordBoxes))
                    }
                    if (options.upTo == "words") {
                        return results
                    }

                    // ---- stage: quads -------------------------------------------------------
                    //
                    // The way back to the input image (geometry.py): field and word quadrilaterals, or null per key
                    // when this run's way back is not known. Emitted right after the split, before any reading.
                    val quads = FieldQuads.compute(split.kept, ocrOptions.needsLicenceRotation) { results.toInput(it) }
                    results.quads = quads
                    options.sink.emit("quads", quads.toJson())
                    if (options.upTo == "quads") {
                        return results
                    }

                    // ---- stages: ocr.<Field>.words, join ------------------------------------
                    val ocrStart = System.nanoTime()
                    val texts = Ocr.run(fieldWords, bareType, ocrOptions, cyrillic, latin, mrzZone, docYear)
                    results.timings[TIMING_OCR] = (System.nanoTime() - ocrStart) / 1e9
                    Ocr.fixFms(texts, bareType)

                    // Ruler cleanup applies to the FINAL per-field values only: the reference emits
                    // `join` from the raw dict and cleans meta_results['OCR'] afterwards
                    // (pipeline.py:1058), so the conformance payload stays raw here too.
                    val cleanRulers = bareType.lowercase().contains("birthcert")

                    val joined = LinkedHashMap<String, String>()
                    for (text in texts) {
                        options.sink.emit(
                            "ocr.${text.label}.words",
                            JsonArray(text.words.map { JsonPrimitive(it) }),
                        )
                        joined[text.label] = text.value
                        results.ocr[text.label] =
                            if (cleanRulers) Ocr.cleanRulerArtifacts(text.value) else text.value
                    }
                    options.sink.emit(
                        "join",
                        JsonObject(joined.mapValues { JsonPrimitive(it.value) }),
                    )
                    results.words = texts

                    // `_reread_dates_whole` (pipeline.py:1271): AFTER `join` was emitted and the ruler cleanup
                    // above — in the reference both happen inside `_ocr` — and BEFORE `_normalize_dates`, so
                    // the canonical view below is built from the re-read value. The date crops it needs are
                    // the detected fields' own patches, still open here: `fields` is closed by the outer
                    // `finally`, after this method's last use of them.
                    results.datesReadWhole = RereadDates.run(
                        results.ocr, split.dateLines, ocrOptions.ruFields, cyrillic, latin)

                    // `_normalize_dates` (pipeline.py:1977): runs once, on the FINISHED reading, after
                    // ruler cleanup — never touching `results.ocr` itself, since that is what the
                    // accuracy measurement compares against. Fields are recognised BY NAME, the same
                    // convention `_join_field` uses: any key whose name contains "date", case-insensitive.
                    val normalized = LinkedHashMap<String, String>()
                    for ((name, value) in results.ocr) {
                        if (!name.lowercase().contains("date")) {
                            continue
                        }
                        Dates.canonicalDate(name, value)?.let { normalized[name] = it }
                    }
                    results.ocrNormalized = normalized

                    // `_read_leasing`: the flag from the STS special marks, alongside the reading - only for an
                    // STS, and the stage is emitted for the BACK side alone (the side with the marks): null when
                    // there is no leasing, so a port that misses a record differs from the reference instead of
                    // being skipped. Reached only when something was read, as in the reference.
                    if (fieldWords.isNotEmpty()) {
                        results.leasing = StsMarks.readLeasing(results.ocr, bareType, options.sink)
                    }

                    finaliseTimings(results.timings)

                    if (options.upTo == "join") {
                        return results
                    }

                    // ---- stage: viewmodel ---------------------------------------------------
                    options.sink.emit(
                        "viewmodel",
                        json.encodeToJsonElement(
                            Payload.serializer(),
                            buildViewModel(results, options.includeDebug),
                        ),
                    )
                    return results
                } finally {
                    SplitWords.closeAll(fieldWords)
                }
            } finally {
                closeAllFields(fields)
            }
        } catch (e: Throwable) {
            results.close()
            throw e
        }
    }

    /**
     * Documents in the frame, largest first, boxes on the INPUT photo. Port of `Pipeline._find_documents`.
     *
     * The detector reads the frame at the processing size; its boxes are scaled back to the input photo so
     * the crop can be cut at full resolution — a licence on an A4 scan keeps its pixels instead of the ~300
     * left to it once the whole sheet is shrunk to `imgSize`. The shrink is the same `fitToLongestSide` as
     * the later `prepare`, floor-division trap included, so the ratio `sx` is `w / small.width` exactly as
     * the reference computes it from the resized array.
     *
     * Emits the `documents` stage — and only when a detector exists: with none, the reference returns before
     * emitting, and so does this.
     */
    private fun findDocuments(frame: Image, options: RunOptions, results: Results): List<DetectedDocument> {
        val detector = documentDetector ?: return emptyList()
        val started = System.nanoTime()
        val found = Io.fitToLongestSide(frame, options.imgSize).use { small ->
            val sx = frame.width.toDouble() / small.width
            val sy = frame.height.toDouble() / small.height
            detector.predict(small).map { it.scaled(sx, sy) }
        }
        results.timings[TIMING_DOCUMENT_DETECTOR] = (System.nanoTime() - started) / 1e9

        // `round(v, 1)` of the builtin, and `round(conf, 3)`: PyNum.roundDecimal, not the np.round scaling.
        fun r1(v: Double) = JsonPrimitive(PyNum.roundDecimal(v, 1))
        fun corners(b: DocBox) = JsonArray(listOf(r1(b.x1), r1(b.y1), r1(b.x2), r1(b.y2)))
        options.sink.emit(
            "documents",
            JsonArray(found.map { d ->
                JsonObject(
                    mapOf(
                        "box" to corners(d.box),
                        "conf" to JsonPrimitive(PyNum.roundDecimal(d.box.conf, 3)),
                        "pages" to JsonArray(d.pages.map { corners(it) }),
                    ),
                )
            }),
        )
        return found
    }

    /**
     * `(x0, y0, x1, y1)` on the input photo: the document box plus its margin. `Pipeline._document_crop`.
     *
     * [DOCUMENT_CROP_MARGIN] of the box's LONGER side goes out on every side — the border detector still
     * needs a strip of background to find the edges, and the box itself can sit a few pixels inside the
     * paper — floored on the near edge and ceiled on the far one, then clamped to the photo.
     */
    private fun documentCrop(frame: Image, box: DocBox): IntArray = documentCrop(box, frame.width, frame.height)

    /**
     * The four quality checks, run CONCURRENTLY.
     *
     * Launched in the reference's source order and collected positionally — see [Parallel] for why that is
     * not a style choice. Each has its own model and therefore its own session, which is what makes the
     * concurrency worth having: the per-session lock only serialises calls to the SAME session, so four
     * different models genuinely overlap.
     *
     * The verdicts are strings — `"good"`/`"bad"` for glare and blur, `"REAL"`/`"FAKE"` for the two
     * spoofing checks. That inconsistency is in the reference and the wire contract carries it, so the map
     * is deliberately heterogeneous rather than normalised.
     */
    private fun runQuality(
        image: Image,
        docConfidence: Double,
        timings: MutableMap<String, Double>,
    ): Map<String, String> {
        val groupStart = System.nanoTime()

        val labels = Parallel.run(
            listOf(
                { glare.predict(image).first },
                { blur.predict(image).first },
                { printSpoofing.predict(image).first },
                { lcdSpoofing.predict(image).first },
            ),
        )

        // The group's own wall time counts toward the total; its members' do not, or the report would
        // claim more time than actually elapsed. The members are recorded as zero because the reference
        // measures them inside the group and this port does not thread a stopwatch through four lambdas
        // for a value the tolerance spec never compares.
        timings[TIMING_QUALITY_AND_BORDERS] = (System.nanoTime() - groupStart) / 1e9
        timings[TIMING_GLARE] = 0.0
        timings[TIMING_BLUR] = 0.0
        timings[TIMING_PRINT_SPOOFING] = 0.0
        timings[TIMING_LCD_SPOOFING] = 0.0

        // DocConf first, matching the reference's insertion order. The comparison is key-by-key so order
        // does not affect it, but a diff of two dumps is far easier to read when it does.
        val quality = LinkedHashMap<String, String>()
        quality["DocConf"] = docConfidence.toString()
        for ((i, key) in QUALITY_KEYS.withIndex()) {
            quality[key] = labels[i]
        }
        return quality
    }

    /**
     * Rebuilds the internal-passport canvas from template-registered pages, when possible. Port of
     * `Pipeline._register_pages` (pipeline.py:1026).
     *
     * [img] is the UPRIGHT photo (before border detection), matching the reference's call site exactly:
     * `_register_pages(img)` runs with `img` still the rotated frame — the local `img` in `process_img`
     * is reassigned to `img_with_fixed_perspective` only AFTER this call returns, so this function reads
     * and warps from the same pixels [DocDetector] itself started from, not from its output canvas.
     *
     * Returns [borders] UNCHANGED (by reference — callers use `!==` to tell) for every non-passport type,
     * an address page, or whenever nothing could be registered; a NEW [BordersResult] — new canvas, new
     * per-page pieces — when registration replaced the Borders-only warp for at least one page.
     */
    private fun registerPages(img: Image, docType: String, borders: BordersResult): BordersResult {
        val lower = docType.lowercase()
        if (lower.startsWith("sts")) return registerCard(img, docType, borders)
        if (!lower.contains("intpassport") || lower.contains("addr")) return borders

        val (quads, _) = pageRegistrar.pageQuads(borders.segments, img.height to img.width)
        val regs = pageRegistrar.register(img, quads)
        val scale = pageRegistrar.nativeScale(regs)

        // Which Borders quad (if any) agrees with each registered page, and whether to trust that quad's
        // geometry over the template's — pipeline.py:1060-1084.
        val used = HashSet<Int>()
        val quadStep = arrayOfNulls<Int>(regs.size) // Borders quad index to warp from, per page
        val useTemplate = BooleanArray(regs.size)   // true: warp from the registration's own homography
        for (i in regs.indices) {
            val r = regs[i]
            if (!r.ok) continue
            var bestI = -1
            var bestIou = 0.0
            for (qi in quads.indices) {
                if (qi in used) continue
                val iou = pageRegistrar.quadIou(quads[qi], r.quad!!)
                if (iou > bestIou) {
                    bestI = qi
                    bestIou = iou
                }
            }
            if (bestI >= 0 && bestIou >= QUAD_SAME_PAGE_IOU) {
                used += bestI
                val q = quads[bestI]
                val clipped = q.any { it.x <= 1 || it.y <= 1 || it.x >= img.width - 2 || it.y >= img.height - 2 }
                if (clipped && bestIou < QUAD_CLIPPED_IOU) {
                    useTemplate[i] = true
                } else {
                    quadStep[i] = bestI
                }
            } else {
                useTemplate[i] = true
            }
        }
        // Spare Borders quads (not claimed by any registered page), top to bottom — fill a page the
        // registrar could not find at all.
        val spare = quads.indices.filter { it !in used }.sortedBy { quads[it].minOf { p -> p.y } }.toMutableList()

        val pages = ArrayList<Image>()
        val pageGeometries = ArrayList<PointMap>()
        val sources = arrayOfNulls<String>(regs.size)
        val quadUsed = arrayOfNulls<Int>(regs.size)
        var handedOff = false
        try {
            for (i in regs.indices) {
                val r = regs[i]
                // The matrix the page is warped with, photo -> page: the reference keeps it (`M`) because it IS the
                // way back of the page.
                val matrix: Array<DoubleArray> = when {
                    // `!r.ok` implies neither of the two flags below was ever set for this page —
                    // `plan.append(None)` in the reference.
                    !r.ok -> {
                        if (spare.isEmpty()) {
                            continue
                        }
                        val qi = spare.removeAt(0)
                        sources[i] = "borders-spare"
                        quadUsed[i] = qi
                        pageRegistrar.quadMatrix(Geometry.expandQuadF32(quads[qi].toList(), Geometry.DOC_MARGIN_FRACTION).toTypedArray(), scale)
                    }
                    quadStep[i] != null -> {
                        val qi = quadStep[i]!!
                        sources[i] = "borders"
                        quadUsed[i] = qi
                        pageRegistrar.quadMatrix(Geometry.expandQuadF32(quads[qi].toList(), Geometry.DOC_MARGIN_FRACTION).toTypedArray(), scale)
                    }
                    else -> {
                        sources[i] = "template"
                        pageRegistrar.pageMatrix(r, scale)
                    }
                }
                val page = pageRegistrar.warpMatrix(img, matrix, scale)
                val (straightened, sinfo) = try {
                    pageRegistrar.straighten(page, scale)
                } catch (e: Throwable) {
                    page.close()
                    throw e
                }
                if (straightened !== page) page.close()
                pages += straightened
                // page -> the image the registrar received (geometry.py)
                pageGeometries += Chain(listOf(Homography(matrix))).then(sinfo.chain())
            }
            if (pages.isEmpty()) {
                return borders
            }
            val pageQuadsFinal = ArrayList<List<Pt>>()
            for (i in regs.indices) {
                if (sources[i] == null) continue
                val qi = quadUsed[i]
                pageQuadsFinal += if (qi != null) quads[qi].toList() else regs[i].quad!!.toList()
            }
            val (stitched, placements) = Geometry.stitchPages(pages, pageQuadsFinal, StackDirection.VERTICAL)
            if (stitched == null) {
                pages.forEach { it.close() }
                return borders
            }
            // A SNAPSHOT (`toList()`), not the mutable `pages` list itself: `BordersResult.pages` would
            // otherwise alias the same backing list, and `pages.clear()` below — needed so the `finally`
            // does not close what was just handed off — would empty the result's list too. Exactly the
            // bug this comment is replacing: found by a page count that silently became 0 after a
            // successful registration, diagnosed with a one-line stderr trace, not assumed.
            handedOff = true
            val geometry = Geometry.stitchedGeometry(pages.map { it.width to it.height }, placements, pageGeometries)
            return BordersResult(stitched, borders.segments, pages.toList(), placements, geometry, pageGeometries.toList())
        } finally {
            if (!handedOff) {
                pages.forEach { it.close() } // only reached on an early return before hand-off
            }
        }
    }

    /** `Pipeline._card_registrar`: the template registrar of one STS type, built on first use; null without templates. */
    private fun cardRegistrar(docType: String): PageRegistrar? = synchronized(cardRegistrars) {
        if (!cardRegistrars.containsKey(docType)) {
            cardRegistrars[docType] = try {
                PageRegistrar(docType, refitRounds = CARD_REFIT_ROUNDS)
            } catch (e: java.io.FileNotFoundException) {
                null
            }
        }
        cardRegistrars[docType]
    }

    /**
     * How far the Borders canvas bends the card out of a rectangle, as a share of its long side.
     * `Pipeline._card_skew`.
     *
     * The card's corners found by the template are carried into the Borders canvas and fitted by a similarity;
     * the residual is the skew. The same measure the training-data side used on the client's canvases; visible
     * skew starts at about 1.4 % (handoff of 2026-10-02).
     */
    internal fun cardSkew(reg: PageRegistrar, bordersQuad: Array<Pt>, cardQuad: Array<Pt>): Double {
        val m = reg.quadMatrix(bordersQuad)
        // `cv2.perspectiveTransform` on float32 points: computed in double, stored as float32.
        val found = Array(4) {
            val p = PageHomography.apply(m, cardQuad[it])
            Pt(p.x.toFloat().toDouble(), p.y.toFloat().toDouble())
        }
        val mg = reg.margin
        val ideal = arrayOf(
            Pt(mg.toDouble(), mg.toDouble()), Pt((mg + reg.pageW).toDouble(), mg.toDouble()),
            Pt((mg + reg.pageW).toDouble(), (mg + reg.pageH).toDouble()), Pt(mg.toDouble(), (mg + reg.pageH).toDouble()))
        val from = MatOfPoint2f(*ideal.map { CvPoint(it.x, it.y) }.toTypedArray())
        val to = MatOfPoint2f(*found.map { CvPoint(it.x, it.y) }.toTypedArray())
        val inliers = Mat()
        val affine = Calib3d.estimateAffinePartial2D(from, to, inliers, Calib3d.LMEDS, 3.0, 2000L, 0.99, 10L)
        try {
            if (affine == null || affine.empty()) {
                return 1.0
            }
            val a = DoubleArray(6)
            affine.get(0, 0, a)
            var sum = 0.0
            for (i in 0 until 4) {
                val fx = ideal[i].x * a[0] + ideal[i].y * a[1] + a[2]
                val fy = ideal[i].x * a[3] + ideal[i].y * a[4] + a[5]
                val dx = fx - found[i].x
                val dy = fy - found[i].y
                sum += dx * dx + dy * dy
            }
            return kotlin.math.sqrt(sum / 4) / max(reg.pageW, reg.pageH)
        } finally {
            from.release(); to.release(); inliers.release(); affine?.release()
        }
    }

    /**
     * Rebuilds a vehicle registration certificate's canvas from its printed blank. Port of
     * `Pipeline._register_card` (`4d8c2535`).
     *
     * The opposite policy to the passport's. There the Borders quad keeps the geometry whenever it agrees with
     * the template, because the template fit of a sparsely printed page turns by degrees. Here the card often
     * lies in a plastic sleeve or lamination and the Borders quad is the SLEEVE's edge: it overlaps the card well
     * - it "agrees" - and still skews the canvas (on the client's 1217 cards ~12 % came out visibly skewed,
     * 2026-10-02). The card is printed dense, so the template match is strong: when it reaches
     * [CARD_MIN_INLIERS] the template geometry is taken and the page straightened by its own lines; otherwise
     * the Borders canvas stays as it is. Where the card runs past the photo, the canvas is painted the card's own
     * paper colour rather than the smeared edge (`PageRegistrar.fill`).
     *
     * Returns [borders] UNCHANGED (by reference) whenever the Borders canvas is kept.
     */
    private fun registerCard(img: Image, docType: String, borders: BordersResult): BordersResult {
        val reg = cardRegistrar(docType) ?: return borders
        val (quads, _) = reg.pageQuads(borders.segments, img.height to img.width)
        val r = reg.register(img, quads)[0]
        if (!r.ok || r.inliers < CARD_MIN_INLIERS) {
            return borders
        }
        // Where the Borders canvas is not skewed, keep it: re-cutting a canvas that was right only resamples it
        // (measured on the client's test cards: on the 101 not skewed by Borders, 15 fields read better and 17
        // worse - noise; on the 26 skewed by >= 1 %, 5 better, 1 worse). Skew, not offset: a sleeve runs
        // parallel to the card, so its edge sits 2-3 % off the card's even on a straight canvas; what matters
        // is whether the card comes out a rectangle.
        val cardQuad = r.quad!!
        val skews = quads.map { cardSkew(reg, it, cardQuad) }
        if (skews.isNotEmpty() && skews.min() < CARD_SKEW_KEEP) {
            return borders
        }
        val scale = reg.nativeScale(listOf(r))
        val matrix = reg.pageMatrix(r, scale)
        val fill = paperColour(img, cardQuad)
        val page = reg.warpMatrix(img, matrix, scale, fill)
        val (straightened, sinfo) = try {
            reg.straighten(page, scale)
        } catch (e: Throwable) {
            page.close()
            throw e
        }
        if (straightened !== page) page.close()
        val pageGeometry = Chain(listOf(Homography(matrix))).then(sinfo.chain())
        val (stitched, placements) = Geometry.stitchPages(
            listOf(straightened), listOf(cardQuad.toList()), StackDirection.VERTICAL)
        if (stitched == null) {
            straightened.close()
            return borders
        }
        return BordersResult(stitched, borders.segments, listOf(straightened), placements,
            Geometry.stitchedGeometry(listOf(straightened.width to straightened.height), placements, listOf(pageGeometry)),
            listOf(pageGeometry))
    }

    /**
     * The median colour of the card's own pixels (inside the template quad), per channel, truncated to whole
     * levels - `fill = tuple(float(np.median(img[..., c][mask > 0])))` and the `int(v)` of `warp_matrix`. Null
     * when the quad covers nothing. `np.median` of an even count is the mean of the two middle values.
     */
    private fun paperColour(img: Image, quad: Array<Pt>): DoubleArray? {
        val mask = Mat.zeros(img.height, img.width, CvType.CV_8UC1)
        val poly = MatOfPoint(*quad.map { CvPoint(Math.rint(it.x), Math.rint(it.y)) }.toTypedArray())
        try {
            Imgproc.fillPoly(mask, listOf(poly), Scalar(1.0))
            val m = ByteArray(img.width * img.height)
            mask.get(0, 0, m)
            val px = ByteArray(img.width * img.height * 3)
            img.mat.get(0, 0, px)
            val hist = Array(3) { IntArray(256) }
            var n = 0
            for (i in m.indices) {
                if (m[i].toInt() == 0) continue
                n++
                for (c in 0 until 3) hist[c][px[i * 3 + c].toInt() and 0xFF]++
            }
            if (n == 0) return null
            return DoubleArray(3) { c ->
                fun nth(k: Int): Int {
                    var seen = 0
                    for (v in 0 until 256) {
                        seen += hist[c][v]
                        if (seen > k) return v
                    }
                    return 255
                }
                if (n % 2 == 1) nth(n / 2).toDouble() else (nth(n / 2 - 1) + nth(n / 2)) / 2.0
            }
        } finally {
            mask.release(); poly.release()
        }
    }

    /**
     * Detects text fields PAGE BY PAGE and reports every box in CANVAS coordinates. Port of
     * `Pipeline._fields_from_pages` (pipeline.py:1256).
     *
     * Run in [pages] order (the ORIGINAL segment order — not left-to-right/top-to-bottom stitch order),
     * exactly as the reference iterates `zip(pages, placements)`: that order is what `fields.bbox` records,
     * and the harness compares it positionally. Detection runs on each page at its OWN resolution — the
     * whole reason for this path — and only the box coordinates are remapped; the cropped patches already
     * come from the page pixels and need no further transform.
     */
    private fun fieldsFromPages(
        pages: List<Image>,
        placements: List<Placement>,
        rotateLicence: Boolean,
    ): List<Field> {
        val output = ArrayList<Field>()
        try {
            for (i in pages.indices) {
                val (scale, dx, dy) = placements[i]
                val onPage = textFields.predictTransform(pages[i], rotateLicence)
                for (field in onPage) {
                    // `int(round(box[0]*scale+dx))` etc — pipeline.py:1273. The patch is untouched: it was
                    // already cropped from the page at full resolution.
                    val moved = field.box.copy()
                    moved.x1 = PyNum.roundHalfEvenToInt(field.box.x1 * scale + dx).toDouble()
                    moved.y1 = PyNum.roundHalfEvenToInt(field.box.y1 * scale + dy).toDouble()
                    moved.x2 = PyNum.roundHalfEvenToInt(field.box.x2 * scale + dx).toDouble()
                    moved.y2 = PyNum.roundHalfEvenToInt(field.box.y2 * scale + dy).toDouble()
                    // The frame is written as the stages that make the PATCH out of the CANVAS (geometry.py reads a
                    // chain that way round): take the page off its place on the canvas, undo its resize, cut at
                    // the box.
                    val placed = Geometry.placedRect(pages[i].width, pages[i].height, placements[i]).first
                    val frame = Chain(listOf(
                        Offset(-dx, -dy),
                        Scale(pages[i].width / placed[2], pages[i].height / placed[3]),
                        Offset(-field.box.x1.toInt().toDouble(), -field.box.y1.toInt().toDouble()),
                    ))
                    output += Field(moved, field.patch, frame)
                    // The Field wrapper above now owns `field.patch`; `field` itself (the old wrapper) must
                    // not close it too. `Field.close()` only closes `patch`, and both wrappers point at the
                    // SAME Image, so leaving the original list alone (never calling closeAllFields on it) is
                    // what keeps this a move rather than a double-free.
                }
            }
            return output
        } catch (e: Throwable) {
            closeAllFields(output)
            throw e
        }
    }

    /**
     * Rounds every stage time and adds `total`.
     *
     * **`total` sums only the stages that ran SEQUENTIALLY.** The quality group's four members overlap inside
     * `_quality_and_borders`, so adding them as well would claim more time than actually elapsed — the group's
     * own wall time is the honest figure and its members are recorded as zero.
     *
     * Every value is rounded to four places, for the reason every wire float is: unrounded, the text of a
     * double differs between languages in the last digits and the goldens diverge for no semantic reason.
     */
    private fun finaliseTimings(timings: MutableMap<String, Double>) {
        val concurrent = setOf(TIMING_GLARE, TIMING_BLUR, TIMING_PRINT_SPOOFING, TIMING_LCD_SPOOFING)
        var total = 0.0
        for ((key, value) in timings) {
            val rounded = Ops.roundHalfEven(value, 4)
            timings[key] = rounded
            if (key !in concurrent) {
                total += rounded
            }
        }
        timings["total"] = Ops.roundHalfEven(total, 4)
    }

    /**
     * Assembles the view model from a finished run.
     *
     * Built here rather than by the service, so the conformance CLI can emit it without an HTTP layer
     * existing — D-01. Takes the canvas DIMENSIONS out of the result rather than the image, which keeps
     * the builder free of any ownership question.
     */
    public fun buildViewModel(results: Results, includeDebug: Boolean): Payload = Builder.build(
        Input(
            docType = results.docType,
            device = results.device,
            canvasW = results.canvas?.width ?: 0,
            canvasH = results.canvas?.height ?: 0,
            canvasMissing = results.canvas == null,
            boxes = results.boxes,
            ocr = results.ocr,
            quality = results.quality.mapValues { (key, value) ->
                Builder.qualityElement(key, value)
            },
            timings = results.timings,
            segments = results.segments,
            normalized = results.ocrNormalized,
        ),
        includeDebug,
    )

    override fun close() {
        // Every closer runs even if an earlier one throws. Stopping at the first failure would leak the
        // remaining sessions, and on GPU that is retained device memory — which outlives the process's own
        // memory in how long it takes to notice.
        val failures = mutableListOf<Throwable>()
        for (closeable in listOfNotNull(documentDetector, docTypeAngles, glare, blur, printSpoofing, lcdSpoofing, docDetector,
            textFields, words, cyrillic, latin, pageRegistrar) + cardRegistrars.values.filterNotNull()) {
            try {
                closeable.close()
            } catch (e: Throwable) {
                failures += e
            }
        }
        failures.firstOrNull()?.let { first ->
            failures.drop(1).forEach { first.addSuppressed(it) }
            throw first
        }
    }

    /**
     * Boxes in the wire shape: `[x1, y1, x2, y2, conf, cls, label]`.
     *
     * The coordinates are TRUNCATED to int here even though they are already whole after the detector's
     * own truncation — because the reference emits `int(...)` at this point, and the harness compares
     * these rows positionally with a per-column tolerance.
     */
    private fun encodeBoxes(boxes: List<net.russiandocs.docproc.postprocess.Box>): JsonElement =
        JsonArray(boxes.map { b ->
            JsonArray(listOf(
                JsonPrimitive(b.x1.toInt()), JsonPrimitive(b.y1.toInt()),
                JsonPrimitive(b.x2.toInt()), JsonPrimitive(b.y2.toInt()),
                JsonPrimitive(b.conf), JsonPrimitive(b.cls), JsonPrimitive(b.label),
            ))
        })

    /**
     * One field's word boxes, one entry per DETECTION of that field.
     *
     * A null entry stays JSON null and means "this field needs no splitting, so its whole patch is the
     * single word" — a different claim from "the detector found exactly one word". A port that split a
     * field it should not have would otherwise look like agreement.
     */
    private fun encodeWordBoxes(
        wordBoxes: List<List<net.russiandocs.docproc.postprocess.Box>?>,
    ): JsonElement = JsonArray(wordBoxes.map { boxes ->
        if (boxes == null) JsonNull else encodeBoxes(boxes)
    })

    private fun encode(meta: DocTypeResult): JsonElement =
        json.encodeToJsonElement(DocTypeResult.serializer(), meta)

    /**
     * Contours as the harness expects them: a list of point lists, or null when nothing was found.
     *
     * Compared under relaxation R-01 rather than point-for-point, because the number of points
     * `findContours` returns legitimately depends on the OpenCV minor version. Area, an area-weighted
     * centroid and Hausdorff distance are what actually get checked.
     *
     * The coordinates are emitted as INTEGERS: `findContours` returns integral points, and writing them as
     * floats would make the golden's `[[12, 34]]` and this port's `[[12.0, 34.0]]` differ as JSON while
     * being the same contour.
     */
    private fun encodeSegments(segments: List<List<Pt>>?): JsonElement =
        if (segments == null) {
            JsonNull
        } else {
            JsonArray(segments.map { contour ->
                JsonArray(contour.map { p ->
                    JsonArray(listOf(JsonPrimitive(p.x.toInt()), JsonPrimitive(p.y.toInt())))
                })
            })
        }

    /**
     * Encodes the quality map, with `DocConf` as a NUMBER and the verdicts as strings.
     *
     * The map is `Map<String, String>` internally because four of the five values are genuinely strings,
     * but `DocConf` is a float on the wire and the harness compares it with a tolerance. Emitting it as a
     * string would make the comparison exact and fail on the last digit.
     */
    private fun encodeQuality(quality: Map<String, String>): JsonElement = JsonObject(
        quality.mapValues { (key, value) ->
            if (key == "DocConf") {
                JsonPrimitive(value.toDouble())
            } else {
                JsonPrimitive(value)
            }
        },
    )

    public companion object {
        /** `_register_pages`'s IoU gate for trusting a Borders quad over the template match. pipeline.py:25-26. */
        /**
         * Share of the document box.s longer side added around it when it is cut from the frame
         * (`Pipeline.DOCUMENT_CROP_MARGIN`).
         */
        public const val DOCUMENT_CROP_MARGIN: Double = 0.03

        /** [documentCrop] on a photo of [width] x [height] — pure arithmetic, so it can be pinned without a model. */
        internal fun documentCrop(box: DocBox, width: Int, height: Int): IntArray {
            val m = DOCUMENT_CROP_MARGIN * max(box.x2 - box.x1, box.y2 - box.y1)
            return intArrayOf(
                max(0.0, floor(box.x1 - m)).toInt(),
                max(0.0, floor(box.y1 - m)).toInt(),
                min(width.toDouble(), ceil(box.x2 + m)).toInt(),
                min(height.toDouble(), ceil(box.y2 + m)).toInt(),
            )
        }

        /**
         * Template matches a vehicle registration certificate needs before its template geometry replaces the
         * Borders quad (`Pipeline.CARD_MIN_INLIERS`). The card is printed dense (captions, rules, guilloche), so a
         * real match runs to a hundred or more points (median 141 on the client's test cards, 2026-10-02); 40 is
         * the floor the form matching of the training-data side uses, below it the Borders canvas stays.
         */
        public const val CARD_MIN_INLIERS: Int = 40

        /**
         * Skew of the Borders canvas below which it is kept (see [cardSkew]); the residual as a share of the
         * long side. Visible skew starts at about 1.4 % (handoff of 2026-10-02). `Pipeline.CARD_SKEW_KEEP`.
         */
        public const val CARD_SKEW_KEEP: Double = 0.01

        /**
         * Least-squares re-fits after MAGSAC in the card's template match ([PageRegistrar.refitRounds]): the skew
         * decision above must not move with MAGSAC's samples. `Pipeline.CARD_REFIT_ROUNDS`.
         */
        public const val CARD_REFIT_ROUNDS: Int = 5

        public const val QUAD_SAME_PAGE_IOU: Double = 0.40
        public const val QUAD_CLIPPED_IOU: Double = 0.80

        /**
         * The stages this build can emit, in pipeline order. Grows one milestone at a time.
         *
         * The CLI reads this rather than repeating it, so the claim and the behaviour cannot drift.
         */
        public val STAGES_IMPLEMENTED: List<String> =
            listOf(
                "documents", "prepare", "doctype.label", "rotate", "quality",
                "borders.segments", "borders.canvas", "deskew.canvas",
                // **The per-field stages are claimed as PATTERNS, not as names.** The checker expands
                // `words.<Field>.bbox` to cover `words.Last_name_ru.bbox` and so on, because which fields
                // exist depends on the document. Claiming a bare "words" matches nothing, and the symptom is
                // silent: every per-field stage is reported SKIPPED and the run still says PASS. Caught here
                // by the stage count not moving from 7 after the work was done.
                "fields.bbox", "words.<Field>.bbox", "quads", "ocr.<Field>.words", "join", "leasing", "viewmodel",
            )

        /** The quality keys, in the reference's insertion order. */
        private val QUALITY_KEYS =
            listOf("Glare", "Blur", "PrintSpoofing", "LCDSpoofing")

        /**
         * Timing keys, with the reference's leading underscores.
         *
         * They are taken from `func.__name__` on the Python side, so `_doctype_angle` is the name on the
         * wire. The view model's `timings` KEY SET is compared exactly — only the values are ignored — so
         * renaming these to something idiomatic is a breaking change rather than tidying.
         */
        public const val TIMING_DOCUMENT_DETECTOR: String = "_document_detector"
        public const val TIMING_DOCTYPE_ANGLE: String = "_doctype_angle"
        public const val TIMING_QUALITY_AND_BORDERS: String = "_quality_and_borders"
        public const val TIMING_GLARE: String = "_glare"
        public const val TIMING_BLUR: String = "_blur"
        public const val TIMING_PRINT_SPOOFING: String = "_print_spoofing"
        public const val TIMING_LCD_SPOOFING: String = "_lcd_spoofing"
        public const val TIMING_DOC_DETECTOR: String = "_doc_detector"
        public const val TIMING_REGISTER_PAGES: String = "_register_pages"
        public const val TIMING_DESKEW: String = "_deskew"
        public const val TIMING_FIELDS_DETECTOR: String = "_fields_detector"
        public const val TIMING_SPLIT_WORDS: String = "_split_words"
        public const val TIMING_OCR: String = "_ocr"

        private val json = Json { encodeDefaults = true; explicitNulls = true }
    }
}
