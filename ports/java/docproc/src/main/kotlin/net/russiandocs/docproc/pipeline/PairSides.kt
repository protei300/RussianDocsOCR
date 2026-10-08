package net.russiandocs.docproc.pipeline

/**
 * Pairs the two sides of one document read from one frame. Port of `PAIRED_SIDES` / `pair_sides`
 * (`91eb0a9`, issue #26).
 *
 * The vehicle registration certificate prints its series and number on both sides (read as `Licence_number` on
 * each), so a front and a back lying in one frame — one sheet of a two-sided scan — pair up by it. A licence
 * back carries no number the pipeline reads, so it is not here.
 */
public object PairSides {

    /** Front type -> back type of a two-sided document whose sides carry the same number. */
    public val PAIRED_SIDES: Map<String, String> = mapOf("STS" to "STSBACK")

    /** Digits of the series and number read on a side; empty when nothing was read. `_document_number`. */
    private fun documentNumber(results: Results): String =
        (results.ocr["Licence_number"] ?: "").filter { it.isDigit() }

    /** The type without its year (`rsplit('_', 1)[0]`); `NONE` for an unread document. */
    private fun family(results: Results): String = OcrOptions.splitDocType(results.docType.ifEmpty { "NONE" }).first

    /**
     * Sets [Results.pairedWith] on the two sides of one document.
     *
     * A front and a back pair when their types belong together ([PAIRED_SIDES]), they read the same number, and
     * that number is unique among the sides of that type in the frame — two certificates scanned on one sheet
     * must not cross-pair, and an ambiguous number pairs nothing rather than guessing. A number of fewer than
     * six digits is no number.
     */
    public fun pair(documents: List<Results>) {
        for ((front, back) in PAIRED_SIDES) {
            val fronts = LinkedHashMap<String, MutableList<Int>>()
            val backs = LinkedHashMap<String, MutableList<Int>>()
            for ((i, r) in documents.withIndex()) {
                val number = documentNumber(r)
                if (number.length < 6) {
                    continue
                }
                val side = when (family(r)) {
                    front -> fronts
                    back -> backs
                    else -> null
                } ?: continue
                side.getOrPut(number) { mutableListOf() } += i
            }
            for ((number, f) in fronts) {
                val b = backs[number] ?: emptyList()
                if (f.size == 1 && b.size == 1) {
                    documents[f[0]].pairedWith = b[0]
                    documents[b[0]].pairedWith = f[0]
                }
            }
        }
    }
}
