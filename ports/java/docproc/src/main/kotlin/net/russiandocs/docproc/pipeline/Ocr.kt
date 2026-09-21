package net.russiandocs.docproc.pipeline

import net.russiandocs.docproc.imaging.Crop
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.modules.OcrEngine
import net.russiandocs.docproc.tensors.Ops
import kotlin.math.max
import kotlin.math.min

/** One field's OCR result: the per-word strings and the joined value. */
public class FieldText(
    public val label: String,
    public val words: MutableList<String> = mutableListOf(),
    public var value: String = "",
)

/**
 * The MRZ zone as detected, kept for the re-read ladder — `Pipeline._mrz_zone` / `_note_mrz_zone`.
 *
 * Built once, right after text-field detection, from the RAW detector boxes: nothing here is a rewrite
 * of `fields.bbox`, only a note of where each MRZ line was found so a wrong-length reading can be
 * re-cropped from the same canvas later.
 */
public class MrzZone(
    public val canvas: Image,
    /** Each line's box, TOP TO BOTTOM — the same order the OCR loop walks the field's words in. */
    public val boxes: List<IntArray>,
    /** The union span (left, right) of the LINE-SHAPED boxes, or null when there is at most one. */
    public val span: Pair<Int, Int>?,
) {
    public companion object {
        /**
         * Builds the zone from the detector's raw boxes. Port of `_note_mrz_zone` (pipeline.py:1386).
         * Returns null when the document has no MRZ box at all — the ladder then never engages.
         */
        public fun from(boxes: List<net.russiandocs.docproc.postprocess.Box>, canvas: Image): MrzZone? {
            val idx = boxes.indices.filter { boxes[it].label == "MRZ" }
            if (idx.isEmpty()) {
                return null
            }
            // Top to bottom by box CENTRE — matches `_split_words`' own sort, independently computed
            // here because `_note_mrz_zone` runs before word splitting in the reference too.
            val ordered = idx.sortedBy { (boxes[it].y1 + boxes[it].y2) / 2 }
            val lines = ordered.map { i ->
                val b = boxes[i]
                intArrayOf(b.x1.toInt(), b.y1.toInt(), b.x2.toInt(), b.y2.toInt())
            }
            // A box several line-heights tall is not a single line and its edges say nothing about
            // where a LINE ends — excluded from the span the same way the reference excludes it.
            val lineShaped = lines.filter { (it[2] - it[0]) >= 10 * max(1, it[3] - it[1]) }
            val span = if (lineShaped.size > 1) {
                lineShaped.minOf { it[0] } to lineShaped.maxOf { it[2] }
            } else {
                null
            }
            return MrzZone(canvas, lines, span)
        }
    }
}

public object Ocr {

    /** A machine-readable line is always exactly this many characters. `Pipeline.MRZ_LINE_LEN`. */
    private const val MRZ_LINE_LEN = 44

    /**
     * How far the re-read crop widens per side, as a fraction of the zone span, tried in order until a
     * candidate is exactly [MRZ_LINE_LEN] characters. `Pipeline.MRZ_RETRY_GROWTH` (pipeline.py:1377).
     */
    private val MRZ_RETRY_GROWTH = doubleArrayOf(0.0, 0.05, 0.10, 0.16, 0.24, 0.34)

    /** The zone's closed alphabet: capitals, digits, the filler. `Pipeline.MRZ_ALPHABET`. */
    private val MRZ_ALPHABET: Set<Char> = ('A'..'Z').toSet() + ('0'..'9').toSet() + setOf('<')

    /**
     * Drops edge characters outside the MRZ alphabet — a page border caught by a widened crop reads as
     * '.' or '_'. Only the ends; a wrong character INSIDE the line is left alone. `_trim_to_mrz_alphabet`.
     */
    private fun trimToMrzAlphabet(text: String): String {
        if (text.isEmpty()) {
            return text
        }
        val strip = text.toHashSet().also { it.removeAll(MRZ_ALPHABET) }
        if (strip.isEmpty()) {
            return text
        }
        return text.trim { it in strip }
    }

    /**
     * Re-reads one MRZ line from a widening crop when it came out the wrong length. `_read_mrz`
     * (pipeline.py:1429). Never invents a line — it only re-reads a box the detector already found.
     */
    private fun readMrz(lineIndex: Int, text: String, zone: MrzZone?, latin: OcrEngine): String {
        val trimmed = trimToMrzAlphabet(text)
        if (trimmed.length == MRZ_LINE_LEN) {
            return trimmed
        }
        if (zone == null || lineIndex >= zone.boxes.size) {
            return trimmed
        }
        val canvas = zone.canvas
        val width = canvas.width
        val (x1, y1, x2, y2) = zone.boxes[lineIndex].let { Quad(it[0], it[1], it[2], it[3]) }
        var best = trimmed
        for (growth in MRZ_RETRY_GROWTH) {
            val (left0, right0) = zone.span ?: (x1 to x2)
            // `int(round(...))`: half-to-even, exactly as the reference's `round()` on a float.
            val step = Ops.roundHalfEven((right0 - left0) * growth, 0).toInt()
            val cropLeft = max(0, left0 - step)
            val cropRight = min(width, right0 + step)
            Crop.clampedCrop(canvas, cropLeft, y1, cropRight, y2).use { crop ->
                if (crop.width == 0 || crop.height == 0) {
                    return@use
                }
                var candidate = latin.fixErrors("MRZ", latin.predict(crop))
                candidate = trimToMrzAlphabet(candidate)
                if (candidate.length == MRZ_LINE_LEN) {
                    return candidate
                }
                if (candidate.length > best.length) {
                    best = candidate
                }
            }
        }
        return best
    }

    private data class Quad(val x1: Int, val y1: Int, val x2: Int, val y2: Int)

    /** Routes every word crop to an engine and joins the results per field. */
    public fun run(
        fields: List<FieldWords>,
        docType: String,
        options: OcrOptions,
        cyrillic: OcrEngine,
        latin: OcrEngine,
        mrzZone: MrzZone? = null,
    ): List<FieldText> {
        val output = ArrayList<FieldText>(fields.size)

        for (fw in fields) {
            val words = mutableListOf<String>()
            for (i in fw.patches.indices) {
                val patch = fw.patches[i]

                // **SNILS routes by word-index PARITY, not by field semantics.** Its dates read like
                // "26 СЕНТЯБРЯ 1997 ГОДА", so odd-indexed words go to the CYRILLIC engine even inside a date
                // field. It looks like a bug and it is load-bearing: without it the Russian month name is
                // decoded by the Latin engine and comes out as noise.
                //
                // The order of these branches is the reference's, and it matters — the parity rule is checked
                // BEFORE the date rule, or SNILS months would be routed as dates.
                if ((docType == "SNILS" && i % 2 == 1) || fw.label in options.ruFields) {
                    words += cyrillic.fixErrors(fw.label, cyrillic.predict(patch))
                } else if (fw.label.contains("date", ignoreCase = true)) {
                    words += latin.fixErrors(fw.label, latin.predict(patch))
                } else if (fw.label in options.enFields) {
                    var result = latin.fixErrors(fw.label, latin.predict(patch))
                    if (fw.label == "MRZ") {
                        result = readMrz(i, result, mrzZone, latin)
                    }
                    words += result
                }
                // No else: a field that is neither Russian, a date, nor English contributes no words. The
                // reference has the same gap, and a fallback here would invent text.
            }
            output += FieldText(fw.label, words)
        }

        // Joining happens in a SECOND pass, because a field detected twice appends to what the first
        // detection produced — see joinField.
        val joined = HashMap<String, String>()
        for (field in output) {
            field.value = joinField(joined, field.label, docType, field.words)
        }
        return output
    }

    /**
     * Joins one field's words.
     *
     * All from the reference. The date separator follows the CONTENT: a digit date joins with DOTS —
     * `17.03.1987` — while a date spelled out in words joins with spaces. SNILS is worded by definition and
     * stays hard-coded; birth certificates need both (the 1998 blank has a digit Birth_date next to a worded
     * Issue_date, every date on the 2018 blank is worded). Everything else joins with spaces, and APPENDS to
     * whatever an earlier detection of the same label produced, which is how the internal passport's
     * twice-printed series ends up as one value.
     *
     * The double-space squeeze and the trim are the reference's too. They matter because an empty word — a
     * crop the OCR read as nothing — would otherwise leave a visible gap in the value.
     */
    private fun joinField(
        joined: MutableMap<String, String>,
        label: String,
        docType: String,
        words: List<String>,
    ): String {
        // The MRZ arrives as one detection per line, top to bottom. The line boundary is
        // load-bearing — every check digit lives at a fixed offset in line 2 — so the lines are joined
        // with a newline and nothing else is done to the text: a space would be outside the MRZ
        // alphabet, and the double-space squeeze below must not touch it.
        if (label == "MRZ") {
            val mrz = words.filter { it.isNotEmpty() }.joinToString("\n")
            joined[label] = mrz
            return mrz
        }

        val isDate = label.contains("date", ignoreCase = true)
        // Only multi-word dates are affected: a digit date reaches this point as a single word
        // ("22.06.2010"), where the separator cannot show.
        val worded = docType == "SNILS" || words.any { w -> w.any { c -> c.isLetter() } }

        var value = when {
            isDate && !worded -> words.joinToString(".")
            isDate -> words.joinToString(" ")
            else -> {
                val previous = joined[label] ?: ""
                if (previous.isNotEmpty()) {
                    previous + " " + words.joinToString(" ")
                } else {
                    words.joinToString(" ")
                }
            }
        }

        value = value.replace("  ", " ").trim()
        joined[label] = value
        return value
    }

    /**
     * The FMS code beautifier, ported as the no-op it currently is.
     *
     * `Pipeline._fix_fms` in the reference returns immediately — the dictionary lookup was disabled because a
     * cache miss scans ~16k rows with difflib, costing 3.3-5.1 s per document, and on failure it does not
     * correct the code but REPLACES it with the code of the most similar name. Kept as a named stub so the
     * next port does not have to rediscover why it is absent.
     */
    public fun fixFms(fields: List<FieldText>, docType: String) {
        // Intentionally empty. See the note above.
    }

    /** Runs of the ruler dots the 1998 birth-certificate form prints under every value. */
    private val RULER_RUNS = Regex("""[.,_\-"]{2,}""")

    /** A separator standing alone between spaces, or at either end of the string. */
    private val LONE_SEPARATOR = Regex("""(?:^|(?<=\s))[.,_\-"](?=\s|$)""")

    private val WHITESPACE = Regex("""\s+""")

    /**
     * Collapses the dotted ruler lines out of a joined field value.
     *
     * Port of `Pipeline._clean_ruler_artifacts` (pipeline.py:1061). The rulers land inside the field crops
     * and OCR emits runs of those marks around the real words; they carry no information on this form.
     * Commas and quotes are in the set because that is what the engine emits here («28., ИЮЛЯ 2010»,
     * «"""СЕМ","" ПОННИЛОВИЧ»), not because they were expected. Only runs of two or more and marks standing
     * alone are removed, which is what keeps real punctuation: the comma in «Г. ИРКУТСК, ИРКУТСКАЯ ОБЛАСТЬ»
     * is attached to a word and the hyphen in «II-МЮ» sits between letters, so neither matches. Exactly as
     * in the reference.
     *
     * Both reference patterns are ported verbatim: `java.util.regex` supports the lookbehind the second one
     * needs. (The Go port had to substitute token filtering — RE2 has no lookaround.)
     */
    public fun cleanRulerArtifacts(value: String): String {
        var text = RULER_RUNS.replace(value, " ")
        text = LONE_SEPARATOR.replace(text, " ")
        return WHITESPACE.replace(text, " ").trim()
    }
}
