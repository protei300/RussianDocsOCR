package net.russiandocs.docproc.pipeline

import java.time.DateTimeException
import java.time.LocalDate

/**
 * ICAO 9303 check digits and the fields a machine-readable zone carries — the part of `validation/mrz.py`
 * that `validation/fields.py` leans on (`check_digit`, `validate_mrz`, `parse_mrz`, `printed_number`,
 * `numbers_agree`). `cross_check` and `validate` are not ported: nothing in the port reads them.
 *
 * Pure functions of a string. The digits here are ASCII only; the reference's `\d` / `isdigit()` accept other
 * scripts' digits too, which the recognition never produces.
 */
public object MrzChecks {

    private const val FILLER = '<'
    private const val LINE_LEN = 44

    /** Document type of the internal passport's zone ("Passport National"). */
    public const val INTERNAL_TYPE: String = "PN"

    /** TD3 line-2 layout: (name, payload start, payload end, check-digit position). */
    private class Line2Field(val name: String, val start: Int, val end: Int, val digitPos: Int)

    private val LINE2_FIELDS = listOf(
        Line2Field("document_number", 0, 9, 9),
        Line2Field("birth_date", 13, 19, 19),
        Line2Field("expiry_date", 21, 27, 27),
        Line2Field("personal_number", 28, 42, 42),
    )

    /** The composite digit covers the number, the birth date and the rest to the personal number, each WITH its digit. */
    private val COMPOSITE_PARTS = listOf(0 to 10, 13 to 20, 21 to 43)
    private const val COMPOSITE_POS = 43

    private val MRZ_CHARS = Regex("[A-Z0-9<]+")

    /** Python's `str.splitlines()` boundaries. */
    private val LINE_BREAK = Regex("\r\n|[\n\r\u000b\u000c\u001c\u001d\u001e\u0085  ]")

    /**
     * ICAO 9303 check digit: weights cycle 7-3-1, `A`-`Z` are 10-35, `<` is 0. Null when the payload holds a
     * character outside the MRZ alphabet — an OCR artifact, for which no meaningful digit exists.
     */
    public fun checkDigit(payload: String): Int? {
        val weights = intArrayOf(7, 3, 1)
        var total = 0
        for ((i, ch) in payload.withIndex()) {
            val value = when {
                ch == FILLER -> 0
                ch in '0'..'9' -> ch - '0'
                ch in 'A'..'Z' -> ch.code - 55
                else -> return null
            }
            total += value * weights[i % 3]
        }
        return total % 10
    }

    /** MRZ string -> lines; the pipeline joins detected lines with `\n`. Empty lines are dropped. */
    public fun splitLines(mrzText: String?): List<String> {
        if (mrzText.isNullOrEmpty()) {
            return emptyList()
        }
        return mrzText.split(LINE_BREAK).map { it.trim() }.filter { it.isNotEmpty() }
    }

    /** A field the document does not carry: all filler, filler in place of the digit (the internal passport's expiry date). */
    private fun isEmptyField(payload: String, digit: String): Boolean =
        payload == FILLER.toString().repeat(payload.length) && digit == FILLER.toString()

    /**
     * Verdict of [validateMrz]. [checks] holds `ok` / `failed` / `not_applicable` per part (and `composite`);
     * it is EMPTY when the zone is malformed (then [ok] is false and [error] says why).
     */
    public class Validation(
        public val ok: Boolean,
        public val checks: Map<String, String>,
        public val lines: Int,
        public val error: String? = null,
    )

    /** EXT-4: verify every check digit of the zone. `ok` is false for a malformed zone: lost characters are a finding. */
    public fun validateMrz(mrzText: String?): Validation {
        val lines = splitLines(mrzText)
        if (lines.size < 2) {
            return Validation(false, emptyMap(), lines.size, "expected 2 lines")
        }
        val line2 = lines[1]
        if (line2.length != LINE_LEN || !MRZ_CHARS.matches(line2)) {
            return Validation(false, emptyMap(), lines.size, "line 2 malformed (len=${line2.length})")
        }

        val checks = LinkedHashMap<String, String>()
        for (f in LINE2_FIELDS) {
            val payload = line2.substring(f.start, f.end)
            val digit = line2.substring(f.digitPos, f.digitPos + 1)
            if (isEmptyField(payload, digit)) {
                checks[f.name] = "not_applicable"
                continue
            }
            val expected = checkDigit(payload)
            checks[f.name] = if (expected != null && expected.toString() == digit) "ok" else "failed"
        }

        val composite = COMPOSITE_PARTS.joinToString("") { (a, b) -> line2.substring(a, b) }
        val expected = checkDigit(composite)
        checks["composite"] =
            if (expected != null && expected.toString() == line2.substring(COMPOSITE_POS, COMPOSITE_POS + 1)) "ok" else "failed"

        return Validation(checks.values.all { it != "failed" }, checks, lines.size)
    }

    /**
     * YYMMDD -> `dd.mm.yyyy`, or null if it is not a real calendar date. `_mrz_date`.
     *
     * The zone carries no century, so it comes from what the field means: a birth (or issue) date is in the
     * past — a two-digit year later than the current one belongs to the previous century; an expiry date is
     * always 20xx. The current year is read from the clock, as the reference's `date.today()`.
     */
    internal fun mrzDate(raw: String, future: Boolean): String? {
        if (raw.length != 6 || !raw.all { it in '0'..'9' }) {
            return null
        }
        val yy = raw.substring(0, 2).toInt()
        val mm = raw.substring(2, 4).toInt()
        val dd = raw.substring(4, 6).toInt()
        val century = if (future) 2000 else if (yy <= LocalDate.now().year % 100) 2000 else 1900
        return try {
            val d = LocalDate.of(century + yy, mm, dd)
            "%02d.%02d.%04d".format(java.util.Locale.ROOT, d.dayOfMonth, d.monthValue, d.year)
        } catch (e: DateTimeException) {
            null
        }
    }

    /** What a zone carries; a missing or malformed part is null. Keys of `parse_mrz`. */
    public class Parsed(
        public val documentType: String? = null,
        public val issuingState: String? = null,
        public val surname: String? = null,
        public val givenNames: String? = null,
        public val documentNumber: String? = null,
        public val nationality: String? = null,
        public val birthDate: String? = null,
        public val sex: String? = null,
        public val expiryDate: String? = null,
        public val personalNumber: String? = null,
        public val seriesLastDigit: String? = null,
        public val issueDate: String? = null,
        public val subdivisionCode: String? = null,
    )

    /** Python-style `s[a:b]`: never throws, clamps. */
    private fun slice(s: String, a: Int, b: Int): String {
        val from = a.coerceAtMost(s.length)
        val to = b.coerceAtMost(s.length)
        return if (from >= to) "" else s.substring(from, to)
    }

    /**
     * Field values carried by the zone. [internal] says whether this is an internal passport's zone, whose
     * optional data holds three printed fields; null takes it from the zone's own type characters — a caller
     * that knows the type from the classifier should pass it, because those two characters can be misread.
     */
    public fun parseMrz(mrzText: String?, internal: Boolean? = null): Parsed {
        val lines = splitLines(mrzText)
        var documentType: String? = null
        var issuingState: String? = null
        var surname: String? = null
        var givenNames: String? = null
        if (lines.isNotEmpty()) {
            val line1 = lines[0]
            documentType = slice(line1, 0, 2).replace(FILLER.toString(), "").ifEmpty { null }
            issuingState = slice(line1, 2, 5).ifEmpty { null }
            val rest = slice(line1, 5, line1.length)
            val cut = rest.indexOf("<<")
            val first = if (cut >= 0) rest.substring(0, cut) else rest
            val given = if (cut >= 0) rest.substring(cut + 2) else ""
            surname = first.replace(FILLER, ' ').trim().ifEmpty { null }
            givenNames = given.replace(FILLER, ' ').trim().ifEmpty { null }
        }
        var documentNumber: String? = null
        var nationality: String? = null
        var birthDate: String? = null
        var sex: String? = null
        var expiryDate: String? = null
        var personalNumber: String? = null
        var seriesLastDigit: String? = null
        var issueDate: String? = null
        var subdivisionCode: String? = null
        if (lines.size > 1 && lines[1].length == LINE_LEN) {
            val line2 = lines[1]
            documentNumber = line2.substring(0, 9).replace(FILLER.toString(), "").ifEmpty { null }
            nationality = line2.substring(10, 13).ifEmpty { null }
            birthDate = mrzDate(line2.substring(13, 19), future = false)
            sex = line2[20].toString().takeIf { it == "M" || it == "F" }
            expiryDate = mrzDate(line2.substring(21, 27), future = true)
            personalNumber = line2.substring(28, 42).replace(FILLER.toString(), "").ifEmpty { null }
            val isInternal = internal ?: (documentType == INTERNAL_TYPE)
            if (isInternal) {
                // The internal passport's optional data: the LAST digit of the series, the issue date as YYMMDD, the
                // subdivision code as six digits without the hyphen, then filler (FMS order No. 279, table 2).
                val optional = line2.substring(28, 42)
                val digit = slice(optional, 0, 1)
                val code = slice(optional, 7, 13)
                seriesLastDigit = digit.takeIf { it.isNotEmpty() && it[0] in '0'..'9' }
                issueDate = mrzDate(slice(optional, 1, 7), future = false)
                subdivisionCode = code.takeIf { it.length == 6 && it.all { c -> c in '0'..'9' } }
            }
        }
        return Parsed(
            documentType, issuingState, surname, givenNames, documentNumber, nationality, birthDate, sex,
            expiryDate, personalNumber, seriesLastDigit, issueDate, subdivisionCode,
        )
    }

    /**
     * The zone's document number in the form the page prints it, digits only. External passport: the number
     * field as is. Internal passport: the first three digits of the series + the fourth one from the optional
     * data + the six digits of the number — the zone splits the ten printed digits in two places.
     */
    public fun printedNumber(fields: Parsed, internal: Boolean? = null): String? {
        val number = fields.documentNumber
        if (number.isNullOrEmpty()) {
            return null
        }
        val isInternal = internal ?: (fields.documentType == INTERNAL_TYPE)
        if (!isInternal) {
            return number
        }
        val last = fields.seriesLastDigit
        if (last.isNullOrEmpty() || number.length != 9) {
            return null
        }
        return number.substring(0, 3) + last + number.substring(3)
    }

    /**
     * Zone number against the page's digits. Five of the 19 external passports of the 2003 form write the
     * number WITHOUT its two-digit series (seven digits): a seven-digit zone number is compared with the END of
     * the printed number.
     */
    public fun numbersAgree(zoneNumber: String?, pageDigits: String?): Boolean {
        if (zoneNumber.isNullOrEmpty() || pageDigits.isNullOrEmpty()) {
            return false
        }
        if (zoneNumber.length == 7 && pageDigits.length == 9) {
            return pageDigits.endsWith(zoneNumber)
        }
        return zoneNumber == pageDigits
    }
}
