package net.russiandocs.docproc.pipeline

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Geometry
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.imaging.Placement
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.imaging.StackDirection
import net.russiandocs.docproc.modules.BordersResult
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
import net.russiandocs.docproc.postprocess.Box
import net.russiandocs.docproc.tensors.PyNum
import net.russiandocs.docproc.viewmodel.Builder
import net.russiandocs.docproc.viewmodel.Input
import net.russiandocs.docproc.viewmodel.Payload
import net.russiandocs.docproc.viewmodel.RawBox
import net.russiandocs.docproc.tensors.Ops

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

    /** Canonical `dd.mm.yyyy` per date field that converted — `PipelineResults.ocr_normalized`. */
    public var ocrNormalized: Map<String, String> = emptyMap()
        internal set

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
) : AutoCloseable {

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

    init {
        // SLOW — 215 MB of weights and one session each — so construct once and keep the instance. The
        // reference loads them eagerly in its constructor for the same reason, and the service wraps the
        // whole thing in a pool of exactly one.
        val root = ModelPaths.root()
        val paths = ModelPaths.load(root)
        docTypeAngles = DocTypeAngles(root, paths, device, intraOpThreads)
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

    /** Runs the pipeline over one file. */
    public fun run(imagePath: String, options: RunOptions): Results {
        val results = Results()
        results.device = device.wire
        try {
            // ---- stage: prepare ---------------------------------------------------------------
            //
            // Two steps rather than one, because the reference's `_prepare_image` is two and the second
            // only ever SHRINKS. Fusing them would also hide the floor-division trap that makes 2999x1777
            // come out at 1499 rather than 1500.
            val source = results.own(Io.loadRgb(imagePath))
            val prepared = results.own(Io.fitToLongestSide(source, options.imgSize))

            options.sink.emitImage("prepare", prepared)
            if (options.upTo == "prepare") {
                results.canvas = prepared
                return results
            }

            // ---- stages: doctype.label, rotate ------------------------------------------------
            val started = System.nanoTime()
            val (meta, upright) = docTypeAngles.predictTransform(prepared)
            results.own(upright)
            results.timings[TIMING_DOCTYPE_ANGLE] = (System.nanoTime() - started) / 1e9

            results.docType = meta.docType
            results.docConfidence = meta.docTypeConfidence
            results.angle = meta.angle
            results.angleConfidence = meta.angleConfidence

            options.sink.emit("doctype.label", encode(meta))
            options.sink.emitImage("rotate", upright)
            if (options.upTo == "rotate") {
                results.canvas = upright
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
            val deskewStart = System.nanoTime()
            val (deskewed, _) = deskewer.deskew(effectiveCanvas)
            results.own(deskewed)
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
            val fields = if (registered.pages.size >= 2 && registered.placements.size == registered.pages.size) {
                fieldsFromPages(registered.pages, registered.placements, ocrOptions.needsLicenceRotation)
            } else {
                textFields.predictTransform(deskewed, ocrOptions.needsLicenceRotation)
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
                val (bareType, _) = OcrOptions.splitDocType(meta.docType)

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

                    // ---- stages: ocr.<Field>.words, join ------------------------------------
                    val ocrStart = System.nanoTime()
                    val texts = Ocr.run(fieldWords, bareType, ocrOptions, cyrillic, latin, mrzZone)
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

                    // `_normalize_dates` (pipeline.py:1977): runs once, on the FINISHED reading, after
                    // ruler cleanup — never touching `results.ocr` itself, since that is what the
                    // accuracy measurement compares against. Fields are recognised BY NAME, the same
                    // convention `_join_field` uses: any key whose name contains "date", case-insensitive.
                    val normalized = LinkedHashMap<String, String>()
                    for ((name, value) in results.ocr) {
                        if (!name.lowercase().contains("date")) {
                            continue
                        }
                        Dates.toDdMmYyyy(value)?.let { normalized[name] = it }
                    }
                    results.ocrNormalized = normalized

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
        val sources = arrayOfNulls<String>(regs.size)
        val quadUsed = arrayOfNulls<Int>(regs.size)
        var handedOff = false
        try {
            for (i in regs.indices) {
                val r = regs[i]
                val page: Image = when {
                    // `!r.ok` implies neither of the two flags below was ever set for this page —
                    // `plan.append(None)` in the reference.
                    !r.ok -> {
                        if (spare.isEmpty()) {
                            continue
                        }
                        val qi = spare.removeAt(0)
                        sources[i] = "borders-spare"
                        quadUsed[i] = qi
                        pageRegistrar.warpQuad(img, Geometry.expandQuad(quads[qi].toList(), Geometry.DOC_MARGIN_FRACTION).toTypedArray(), scale)
                    }
                    quadStep[i] != null -> {
                        val qi = quadStep[i]!!
                        sources[i] = "borders"
                        quadUsed[i] = qi
                        pageRegistrar.warpQuad(img, Geometry.expandQuad(quads[qi].toList(), Geometry.DOC_MARGIN_FRACTION).toTypedArray(), scale)
                    }
                    else -> {
                        sources[i] = "template"
                        pageRegistrar.warpPage(img, r, scale)
                    }
                }
                val (straightened, _) = pageRegistrar.straighten(page, scale)
                if (straightened !== page) page.close()
                pages += straightened
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
            return BordersResult(stitched, borders.segments, pages.toList(), placements)
        } finally {
            if (!handedOff) {
                pages.forEach { it.close() } // only reached on an early return before hand-off
            }
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
                    output += Field(moved, field.patch)
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
        for (closeable in listOf(docTypeAngles, glare, blur, printSpoofing, lcdSpoofing, docDetector,
            textFields, words, cyrillic, latin, pageRegistrar)) {
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
        public const val QUAD_SAME_PAGE_IOU: Double = 0.40
        public const val QUAD_CLIPPED_IOU: Double = 0.80

        /**
         * The stages this build can emit, in pipeline order. Grows one milestone at a time.
         *
         * The CLI reads this rather than repeating it, so the claim and the behaviour cannot drift.
         */
        public val STAGES_IMPLEMENTED: List<String> =
            listOf(
                "prepare", "doctype.label", "rotate", "quality",
                "borders.segments", "borders.canvas", "deskew.canvas",
                // **The per-field stages are claimed as PATTERNS, not as names.** The checker expands
                // `words.<Field>.bbox` to cover `words.Last_name_ru.bbox` and so on, because which fields
                // exist depends on the document. Claiming a bare "words" matches nothing, and the symptom is
                // silent: every per-field stage is reported SKIPPED and the run still says PASS. Caught here
                // by the stage count not moving from 7 after the work was done.
                "fields.bbox", "words.<Field>.bbox", "ocr.<Field>.words", "join", "viewmodel",
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
