package net.russiandocs.docproc.pipeline

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.docproc.geometry.corners
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.modules.AddressLinesDetector
import net.russiandocs.docproc.modules.AddressTextKindClassifier
import net.russiandocs.docproc.modules.OcrEngine
import net.russiandocs.docproc.modules.RegistrationStamps
import net.russiandocs.docproc.modules.WordsDetector
import net.russiandocs.docproc.postprocess.ObbBox
import kotlin.math.hypot

/**
 * One entry of `meta_results['Address_lines']`: what was found on one address line.
 *
 * [stamp] is an index into the stamps (freshest first), or null for a line inside no stamp. It is only REPORTED
 * when the stamp detector ran ([AddressReading.stampsRan]); without it the key does not exist, as in the reference.
 */
public class AddressLine(
    public val kind: String,
    public val pHandwritten: Double,
    public var text: String?,
    public var stamp: Int? = null,
)

/** One registration stamp (`meta_results['RegistrationStamps']` entry). */
public class RegistrationStamp(
    /** `registered`, `deregistered` or `unclear`. */
    public val kind: String,
    public val confidence: Double,
    /** Corners on the image passed to `run`; null when the way back is not known. */
    public val quad: List<Pt>?,
    /** Indices into the address lines. */
    public val lines: MutableList<Int> = mutableListOf(),
    public var address: String = "",
    /** For a `deregistered` stamp: the index of the registration it cancels. */
    public var cancels: Int? = null,
    /** True on the registration nothing cancels. */
    public var current: Boolean = false,
)

/** Everything `_address_lines` leaves in the results. */
public class AddressReading(
    /** `meta_results['AddressLinesDetector']['obbox']`: EVERY detection, including those whose patch was empty. */
    public val detections: List<ObbBox>,
    /** `_address_line_obboxes`: the detection of each entry of [lines], in the same order. */
    public val lineBoxes: List<ObbBox>,
    public val lines: List<AddressLine>,
    /** `Quads['address_lines']`: one quadrilateral per line on the input image; null — the way back is not known. */
    public val lineQuads: List<List<Pt>>?,
    /** `OCR['Address']`; lines joined by a newline, a handwritten one is [AddressPath.HANDWRITTEN]. */
    public val text: String,
    public val hasHandwritten: Boolean,
) {
    /** True once the stamp detector has tied the lines to stamps. */
    public var stampsRan: Boolean = false
        internal set

    /** `meta_results['RegistrationStamps']`; null when the weight set has no stamp detector. */
    public var stamps: List<RegistrationStamp>? = null
        internal set

    /** The `address.lines` stage payload. */
    public fun linesJson(): JsonElement = JsonArray(lines.map { line ->
        val fields = linkedMapOf<String, JsonElement>(
            "kind" to JsonPrimitive(line.kind),
            "p_handwritten" to JsonPrimitive(line.pHandwritten),
            "text" to (line.text?.let { JsonPrimitive(it) } ?: JsonNull),
        )
        if (stampsRan) {
            fields["stamp"] = line.stamp?.let { JsonPrimitive(it) } ?: JsonNull
        }
        JsonObject(fields)
    })

    /** The `address.stamps` stage payload: the list, or null when the weight set has no stamp detector. */
    public fun stampsJson(): JsonElement {
        val list = stamps ?: return JsonNull
        return JsonArray(list.map { s ->
            JsonObject(
                linkedMapOf(
                    "kind" to JsonPrimitive(s.kind),
                    "confidence" to JsonPrimitive(s.confidence),
                    "quad" to (s.quad?.let { quad ->
                        JsonArray(quad.map { JsonArray(listOf(JsonPrimitive(it.x), JsonPrimitive(it.y))) })
                    } ?: JsonNull),
                    "lines" to JsonArray(s.lines.map { JsonPrimitive(it) }),
                    "address" to JsonPrimitive(s.address),
                    "cancels" to (s.cancels?.let { JsonPrimitive(it) } ?: JsonNull),
                    "current" to JsonPrimitive(s.current),
                ),
            )
        })
    }
}

/**
 * The registration address (INTPASSPORTADDR). Port of `pipeline/reading.py::ReadingStage._address_lines`,
 * `_route_word_ocr` and `_is_number_token`.
 */
public object AddressPath {

    /**
     * What a handwritten address line reads as (OCR is printed-only). ASCII brackets on purpose: the report gets
     * printed to consoles that are still cp1251/cp866 on Windows.
     */
    public const val HANDWRITTEN: String = "[рукопись]"

    /**
     * Detects the oriented address lines on the corrected canvas, classifies each as printed or handwritten, splits
     * the printed ones into words and reads them. [toInput] is `PipelineResults.to_input` — null once means null for
     * the whole run.
     *
     * The serial path only: the reference batches the words on a GPU when `ocr_gpu_batch` is on, and this port keeps
     * OCR on the CPU always (see the note on the engines in [Recognizer]).
     */
    public fun read(
        canvas: Image,
        detector: AddressLinesDetector,
        textKind: AddressTextKindClassifier,
        wordsDetector: WordsDetector,
        cyrillic: OcrEngine,
        latin: OcrEngine,
        toInput: (List<Pt>) -> List<Pt>?,
    ): AddressReading {
        val (detections, patches) = detector.predictTransform(canvas)
        try {
            val lineBoxes = ArrayList<ObbBox>()
            val lines = ArrayList<AddressLine>()
            var lineQuads: MutableList<List<Pt>>? = ArrayList()
            var hasHandwritten = false
            // per line: null for a handwritten one, else the word patches (owned, closed below)
            val slots = ArrayList<List<Image>?>()
            try {
                for ((patch, box) in patches.zip(detections)) {
                    if (patch.isEmpty) {
                        continue
                    }
                    lineBoxes += box
                    // The line on the input image, in the order of `lines`
                    val crop = AddressLinesDetector.cropGeometry(box)
                    val onCanvas = crop.toInput(corners(0.0, 0.0, patch.width.toDouble(), patch.height.toDouble()))
                    val quad = onCanvas?.let(toInput)
                    // None once means None for the whole run
                    if (quad == null) {
                        lineQuads = null
                    } else {
                        lineQuads?.add(quad)
                    }
                    val (kind, prob) = textKind.predict(patch)
                    if (kind == "handwritten") {
                        hasHandwritten = true
                        lines += AddressLine("handwritten", prob, null)
                        slots += null
                        continue
                    }
                    // printed line: split into words (left to right); the reading happens below
                    val (_, crops) = wordsDetector.predictTransform(patch)
                    val words = ArrayList<Image>()
                    if (crops.isEmpty()) {
                        words += patch.clone()
                    } else {
                        for (c in crops) {
                            if (c.isEmpty) c.close() else words += c
                        }
                    }
                    slots += words
                    lines += AddressLine("printed", prob, null)
                }

                val texts = ArrayList<String>()
                for ((i, slot) in slots.withIndex()) {
                    if (slot == null) {
                        texts += HANDWRITTEN
                        continue
                    }
                    val read = slot.map { routeWord(it, cyrillic, latin) }
                    // `[t.strip() for t in texts if t]`, joined by a space
                    val text = read.filter { it.isNotEmpty() }.joinToString(" ") { it.trim() }
                    lines[i].text = text
                    texts += text
                }
                return AddressReading(detections, lineBoxes, lines, lineQuads, texts.joinToString("\n"), hasHandwritten)
            } finally {
                slots.forEach { it?.forEach(Image::close) }
            }
        } finally {
            patches.forEach { it.close() }
        }
    }

    /**
     * Both engines read the word; the Latin result is kept only when it is digit-dominated (a house, building or flat
     * number), the Cyrillic one otherwise (`_route_word_ocr`).
     */
    internal fun routeWord(word: Image, cyrillic: OcrEngine, latin: OcrEngine): String {
        val ru = cyrillic.predict(word)
        val en = latin.predict(word)
        return if (isNumberToken(en)) en else ru
    }

    /**
     * True if the Latin-engine output is digit-dominated: at least one digit, and digits not outnumbered by letters
     * among the alphanumeric characters (`_is_number_token`).
     */
    public fun isNumberToken(text: String?): Boolean {
        val alnum = (text ?: "").filter { it.isLetterOrDigit() }
        if (alnum.isEmpty()) {
            return false
        }
        val digits = alnum.count { it.isDigit() }
        val letters = alnum.length - digits
        return digits >= 1 && digits >= letters
    }
}

/**
 * The registration stamps of the residence registration page and the address lines that belong to each. Port of
 * `pipeline/stamps.py::StampsStage`.
 */
public object Stamps {

    /** A canvas wider than this many heights is a spread of two pages, split down the middle. */
    public const val SPREAD_ASPECT: Double = 1.2

    /**
     * Finds the stamps on the address canvas and ties the address lines to them, mutating [reading]
     * (`stamp` on every line, `stamps`). A line belongs to the stamp its centre lies in; a «снят» stamp holds no
     * address and is tied to the registration it cancels; the registration nothing cancels is the current one.
     */
    public fun read(
        canvas: Image,
        detector: RegistrationStamps,
        reading: AddressReading,
        toInput: (List<Pt>) -> List<Pt>?,
    ) {
        val boxes = detector.predict(canvas)
        val quads = boxes.map { obboxQuad(it) }
        val order = orderStamps(quads, canvas.width, canvas.height)
        val position = HashMap<Int, Int>()
        order.forEachIndexed { k, det -> position[det] = k }

        val stamps = order.map { det ->
            RegistrationStamp(
                kind = RegistrationStamps.KINDS[boxes[det].label] ?: "unclear",
                confidence = boxes[det].conf,
                quad = toInput(quads[det]),
            )
        }
        for ((li, pair) in reading.lines.zip(reading.lineBoxes).withIndex()) {
            val (line, box) = pair
            val det = containingStamp(Pt(box.cx, box.cy), quads)
            line.stamp = det?.let { position.getValue(it) }
            if (det != null) {
                stamps[position.getValue(det)].lines += li
            }
        }
        for (stamp in stamps) {
            stamp.address = stamp.lines.joinToString("\n") { li ->
                val line = reading.lines[li]
                if (line.kind == "handwritten") AddressPath.HANDWRITTEN else (line.text ?: "")
            }
        }
        tieCancellations(stamps)
        reading.stamps = stamps
        reading.stampsRan = true
    }

    /**
     * Detection indices, freshest stamp first. The page is filled in order — on a spread the left page before the
     * right one, on a page top to bottom — and the freshest stamp is the last one filled. `_order_stamps`.
     */
    internal fun orderStamps(quads: List<List<Pt>>, width: Int, height: Int): List<Int> {
        val spread = width > SPREAD_ASPECT * height
        class Key(val page: Int, val cy: Double, val cx: Double)

        fun filled(k: Int): Key {
            val cx = quads[k].sumOf { it.x } / quads[k].size
            val cy = quads[k].sumOf { it.y } / quads[k].size
            return Key(if (spread && cx >= width / 2.0) 1 else 0, cy, cx)
        }
        // `sorted(...)[::-1]`: a stable sort ascending, then reversed
        return quads.indices.sortedWith(compareBy<Int>({ filled(it).page }, { filled(it).cy }, { filled(it).cx }))
            .reversed()
    }

    /**
     * Sets `cancels` on every «снят» stamp and `current` on the registration in force. [stamps] is freshest first: a
     * «снят» cancels the nearest registration filled before it, i.e. the next `registered` stamp in the list.
     */
    internal fun tieCancellations(stamps: List<RegistrationStamp>) {
        val cancelled = HashSet<Int>()
        for ((k, stamp) in stamps.withIndex()) {
            if (stamp.kind != "deregistered") {
                continue
            }
            val j = (k + 1 until stamps.size).firstOrNull { stamps[it].kind == "registered" }
            stamp.cancels = j
            if (j != null) {
                cancelled += j
            }
        }
        val current = stamps.indices.firstOrNull { stamps[it].kind == "registered" && it !in cancelled }
        if (current != null) {
            stamps[current].current = true
        }
    }

    /**
     * The stamp's corners on the canvas, clockwise from the top-left corner of its upright patch — the same
     * convention as the address-line quads (`_obbox_quad`).
     */
    internal fun obboxQuad(box: ObbBox): List<Pt> {
        val wi = AddressLinesDetector.side(box.w)
        val hi = AddressLinesDetector.side(box.h)
        return AddressLinesDetector.cropGeometry(box).toInput(corners(0.0, 0.0, wi.toDouble(), hi.toDouble()))!!
    }

    /**
     * Index of the quad that holds [point]; when quads overlap, the one whose centre is nearest. Null when none
     * does (`_containing_stamp`).
     */
    internal fun containingStamp(point: Pt, quads: List<List<Pt>>): Int? {
        var best: Int? = null
        var bestDistance = 0.0
        for ((k, quad) in quads.withIndex()) {
            var allNonNegative = true
            var allNonPositive = true
            for (i in quad.indices) {
                val a = quad[i]
                val b = quad[(i + 1) % quad.size]
                val cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x)
                if (cross < 0) allNonNegative = false
                if (cross > 0) allNonPositive = false
            }
            if (!(allNonNegative || allNonPositive)) {
                continue
            }
            val cx = quad.sumOf { it.x } / quad.size
            val cy = quad.sumOf { it.y } / quad.size
            val d = hypot(point.x - cx, point.y - cy)
            if (best == null || d < bestDistance) {
                best = k
                bestDistance = d
            }
        }
        return best
    }
}
