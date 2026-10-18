package net.russiandocs.docproc.viewmodel

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.tensors.Ops
import net.russiandocs.docproc.tensors.PyNum

/**
 * Everything the builder needs, as plain data.
 *
 * Takes the canvas WIDTH and HEIGHT rather than the image itself. That is deliberate: it keeps the one type
 * whose whole purpose is to be pure and testable from a literal free of any ownership question — the property
 * the reference's own docstring calls intentional.
 */
public data class Input(
    val docType: String = "NONE",
    val device: String = "cpu",
    val canvasW: Int = 0,
    val canvasH: Int = 0,
    val canvasMissing: Boolean = false,
    val boxes: List<RawBox> = emptyList(),
    val ocr: Map<String, String> = emptyMap(),
    val quality: Map<String, JsonElement> = emptyMap(),
    val timings: Map<String, Double> = emptyMap(),
    val segments: List<List<Pt>>? = null,
    /** Canonical `dd.mm.yyyy` per date field that converted — `results.ocr_normalized`. */
    val normalized: Map<String, String> = emptyMap(),
    /**
     * Every detection of the address-line detector, `meta_results['AddressLinesDetector']['obbox']`; null when the
     * address path did not run (every document but a registration page).
     */
    val addressBoxes: List<RawObb>? = null,
    /** `meta_results['Address_lines']`, one per line that was read. */
    val addressLines: List<RawAddressLine> = emptyList(),
)

/** An oriented detection as the pipeline produces it: the row `[cx, cy, w, h, angle_rad, conf, cls, label]`. */
public data class RawObb(
    val cx: Double,
    val cy: Double,
    val w: Double,
    val h: Double,
    val angle: Double,
    val conf: Double,
    val label: String,
)

/** One address line as the pipeline read it. */
public data class RawAddressLine(val kind: String?, val text: String?, val pHandwritten: Double?)

/** A detector box as the pipeline produces it, before the view model's own shape. */
public data class RawBox(
    val x1: Double,
    val y1: Double,
    val x2: Double,
    val y2: Double,
    val conf: Double,
    val cls: Int,
    val label: String,
)

public object Builder {

    public const val FLOAT_PRECISION: Int = 4

    private const val COORD_SPACE_NOTE =
        "Box coordinates are in canvas pixel space and match the canvas image exactly. " +
            "They cannot be mapped onto the original upload: the library does not retain the deskew angle."

    public fun build(input: Input, includeDebug: Boolean): Payload {
        val boxes = buildBoxes(input.boxes, input.ocr)

        val canvas = if (input.canvasMissing) {
            Canvas(isFallback = true)
        } else {
            Canvas(width = input.canvasW, height = input.canvasH)
        }

        val base = Labels.baseDocType(input.docType)
        return Payload(
            docType = input.docType,
            docTypeBase = base.ifEmpty { null },
            docTypeEra = Labels.docTypeEra(input.docType),
            // An unrecognised document is not an error — the SPA renders it as a legitimate state — so
            // `recognised` is a flag rather than an exception.
            recognised = input.docType.isNotEmpty() && input.docType != "NONE",
            device = input.device,
            canvas = canvas,
            coordSpace = "canvas",
            coordSpaceNote = COORD_SPACE_NOTE,
            boxes = boxes,
            fields = buildFields(input.docType, input.ocr, boxes, input.normalized),
            ocr = input.ocr,
            quality = input.quality,
            timings = input.timings,
            address = buildAddress(input.addressBoxes, input.addressLines),
            debug = if (includeDebug) {
                Debug(DocOutline(coordSpace = "prewarp", polygon = polygonOf(input.segments)))
            } else {
                null
            },
        )
    }

    /**
     * Turns detector boxes into view-model boxes, deciding which one owns each field's text.
     *
     * The owner is the HIGHEST-CONFIDENCE detection of a label, chosen with strict `>` so a tie keeps the
     * earliest. Every other detection of an OCR'd label is marked `ambiguous` — the per-detection text
     * genuinely cannot be recovered from the library's output, so saying so is more honest than attaching the
     * same string to both.
     */
    private fun buildBoxes(raw: List<RawBox>, ocr: Map<String, String>): List<Box> {
        if (raw.isEmpty()) {
            return emptyList()
        }

        val bestByLabel = HashMap<String, Int>()
        for (i in raw.indices) {
            val previous = bestByLabel[raw[i].label]
            if (previous == null || raw[i].conf > raw[previous].conf) {
                bestByLabel[raw[i].label] = i
            }
        }

        return raw.mapIndexed { i, b ->
            val ownsText = bestByLabel[b.label] == i
            val inOcr = ocr.containsKey(b.label)
            Box(
                // Positional ids, so the field-to-box links are stable within one response.
                id = "b$i",
                label = b.label,
                display = Labels.fieldDisplay(b.label),
                kind = if (Labels.isNonText(b.label)) "visual" else "text",
                x1 = b.x1.toInt(),
                y1 = b.y1.toInt(),
                x2 = b.x2.toInt(),
                y2 = b.y2.toInt(),
                conf = round(b.conf),
                cls = b.cls,
                text = if (ownsText) ocr[b.label] else null,
                ambiguous = inOcr && !ownsText,
            )
        }
    }

    /**
     * Builds the ordered field list, linking each to its boxes.
     *
     * An ARRAY rather than a map, which removes three problems at once: the link to boxes, the reading order
     * (a map has none, and insertion order is not reading order), and the font choice, which travels as
     * `script`.
     */
    private fun buildFields(
        docType: String,
        ocr: Map<String, String>,
        boxes: List<Box>,
        normalized: Map<String, String>,
    ): List<Field> {
        val byLabel = HashMap<String, MutableList<String>>()
        val confByLabel = HashMap<String, Double?>()
        for (box in boxes) {
            byLabel.getOrPut(box.label) { mutableListOf() }.add(box.id)
            // The confidence reported for a field is the OWNING box's, not the maximum or the mean: it is the
            // confidence of the detection whose text is being shown.
            if (box.text != null) {
                confByLabel[box.label] = box.conf
            }
        }

        // Sorted before ordering, so the unknown tail is deterministic across languages.
        val ordered = Labels.orderFields(docType, ocr.keys.sorted())

        return ordered.map { name ->
            Field(
                name = name,
                display = Labels.fieldDisplay(name),
                value = ocr[name],
                script = Labels.fieldScript(name),
                conf = confByLabel[name],
                boxIds = byLabel[name] ?: emptyList(),
                // Only where a canonical form exists — an empty string counts as none, as the
                // reference's `if canonical:` does.
                normalized = normalized[name]?.takeIf { it.isNotEmpty() },
            )
        }
    }

    /**
     * Oriented address-line boxes plus their printed/handwritten verdicts (`_build_address`). Null unless the
     * address path ran.
     *
     * The two source lists can desynchronise — the pipeline skips an empty patch without recording a slot — and
     * zipping them blindly would caption boxes with the wrong text, so when the lengths disagree `aligned` is false,
     * the geometry is dropped and only the text stays.
     */
    private fun buildAddress(boxes: List<RawObb>?, lines: List<RawAddressLine>): Address? {
        if (boxes == null || (boxes.isEmpty() && lines.isEmpty())) {
            return null
        }
        val aligned = boxes.size == lines.size
        return Address(
            aligned = aligned,
            lines = lines.mapIndexed { i, line ->
                AddressLine(
                    id = "o$i",
                    kind = line.kind,
                    text = line.text,
                    pHandwritten = num(line.pHandwritten),
                    obbox = if (aligned) boxes[i].let { b ->
                        Obbox(
                            cx = num(b.cx), cy = num(b.cy), w = num(b.w), h = num(b.h),
                            // radians: needs the finer precision
                            angleRad = num(b.angle, 6),
                            conf = num(b.conf), label = b.label,
                        )
                    } else {
                        null
                    },
                )
            },
        )
    }

    /** `_num`: Python's builtin `round(float(v), precision)`; NaN and infinity become null. */
    private fun num(value: Double?, precision: Int = FLOAT_PRECISION): Double? =
        if (value == null || value.isNaN() || value.isInfinite()) null else PyNum.roundDecimal(value, precision)

    private fun polygonOf(segments: List<List<Pt>>?): List<List<List<Int>>>? =
        segments?.map { contour -> contour.map { listOf(it.x.toInt(), it.y.toInt()) } }

    /**
     * Rounds a wire float to four places, half to even.
     *
     * **Rounding on the server is what makes the goldens comparable at all.** Left unrounded, the text of a
     * float differs between languages in the seventeenth digit and every port's JSON diverges from the
     * reference's for no semantic reason. NaN and infinity become null rather than unparseable JSON tokens.
     */
    public fun round(value: Double): Double? =
        if (value.isNaN() || value.isInfinite()) null else Ops.roundHalfEven(value, FLOAT_PRECISION)

    /** Wraps a quality verdict for the wire: DocConf is a NUMBER, the rest are strings. */
    public fun qualityElement(key: String, value: String): JsonElement =
        if (key == "DocConf") JsonPrimitive(value.toDouble()) else JsonPrimitive(value)
}
