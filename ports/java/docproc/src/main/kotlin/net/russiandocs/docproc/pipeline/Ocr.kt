package net.russiandocs.docproc.pipeline

import net.russiandocs.docproc.modules.OcrEngine

/** One field's OCR result: the per-word strings and the joined value. */
public class FieldText(
    public val label: String,
    public val words: MutableList<String> = mutableListOf(),
    public var value: String = "",
)

public object Ocr {

    /** Routes every word crop to an engine and joins the results per field. */
    public fun run(
        fields: List<FieldWords>,
        docType: String,
        options: OcrOptions,
        cyrillic: OcrEngine,
        latin: OcrEngine,
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
                    words += latin.fixErrors(fw.label, latin.predict(patch))
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
