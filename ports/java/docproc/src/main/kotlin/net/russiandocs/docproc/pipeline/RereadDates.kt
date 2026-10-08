package net.russiandocs.docproc.pipeline

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.modules.OcrEngine

/**
 * One date field whose word-by-word reading was replaced by a whole-line one —
 * the `{'field', 'split', 'whole'}` records of `meta_results['DatesReadWhole']`.
 */
public data class DateReread(public val field: String, public val split: String, public val whole: String)

/**
 * Re-reads a date field line by line WHOLE when the word-by-word reading is not a date and the whole
 * reading is. Port of `Pipeline._reread_dates_whole` (commit 1af0d270).
 *
 * The word splitter can drop a word without leaving a hole the gap guard sees: on a real 1998 birth
 * certificate the issue date «ДД» МЕСЯЦА ГГГГ г. came out as month and year only — the day, pressed against
 * the left edge of the field crop, was not taken for a word at all, and the empty stretch it left was 2.6
 * typical words wide against the guard's measured 3.0. Read whole, the same crop gives the day glued to the
 * month and year, which converts.
 *
 * **The rule checks itself.** It replaces a reading only when that reading does NOT convert to dd.mm.yyyy
 * and the whole-line one DOES, so a date that already converts is never touched — and no engine is even
 * called for it. The price is the one the gap guard pays too: a line read whole comes back without spaces;
 * the canonical view is unaffected.
 */
public object RereadDates {

    /**
     * The pipeline's call: the field's own engine — Cyrillic for [ruFields], Latin otherwise, exactly the
     * routing of `_ocr_serial` minus the SNILS parity rule (SNILS never reaches here: its date lines are not
     * kept) — reading each whole line and applying that engine's `fix_errors` for the field.
     */
    public fun run(
        ocr: MutableMap<String, String>,
        dateLines: Map<String, List<Image>>,
        ruFields: List<String>,
        cyrillic: OcrEngine,
        latin: OcrEngine,
    ): List<DateReread> = run(ocr, dateLines) { field, line ->
        val engine = if (field in ruFields) cyrillic else latin
        engine.fixErrors(field, engine.predict(line))
    }

    /**
     * Updates [ocr] in place and returns what it replaced (empty when nothing was).
     *
     * [dateLines] maps a date field's name to the whole crops of its lines, in reading order
     * (`SplitResult.dateLines`); [read] turns one line into the corrected text of that field. [ocr] is the
     * FINISHED reading, the dict `results.ocr` holds. Separate from the engine call so the rule can be
     * pinned without a model — the reference's own test does the same with a stand-in engine.
     */
    public fun run(
        ocr: MutableMap<String, String>,
        dateLines: Map<String, List<Image>>,
        read: (field: String, line: Image) -> String,
    ): List<DateReread> {
        if (dateLines.isEmpty() || ocr.isEmpty()) {
            return emptyList()
        }
        val done = ArrayList<DateReread>()
        for ((fieldName, lines) in dateLines) {
            val splitReading = ocr[fieldName] ?: ""
            if (Dates.canonicalDate(fieldName, splitReading) != null) {
                continue
            }
            val reads = lines.map { read(fieldName, it) }
            val whole = reads.filter { it.isNotEmpty() }.joinToString(" ").trim()
            if (Dates.canonicalDate(fieldName, whole) == null) {
                continue
            }
            ocr[fieldName] = whole
            done += DateReread(fieldName, splitReading, whole)
        }
        return done
    }
}
