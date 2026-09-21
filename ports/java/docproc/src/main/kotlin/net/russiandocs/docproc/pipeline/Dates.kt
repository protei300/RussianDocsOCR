package net.russiandocs.docproc.pipeline

/**
 * Canonical `dd.mm.yyyy` view of a recognised date. Port of `pipeline/dates.py`.
 *
 * The pipeline returns dates AS PRINTED — «15 ОКТЯБРЯ 2020 Г.» on a 2018 birth certificate, «10 ДЕКАБРЯ
 * 1999 ГОДА» on a SNILS, «03.АВГУСТ.1989» on a 1997 internal passport — because that is what the ground
 * truth describes and what the accuracy measurement compares against. The canonical view is built
 * ALONGSIDE the reading, never in place of it: `Results.ocr` keeps the reading, `Results.ocrNormalized`
 * holds this.
 *
 * Two rules shape everything here, both from the reference:
 *
 * - **Never guess.** No year in the text, a month word that does not match, a date outside the calendar
 *   (31.02) — no canonical value. The caller falls back to the reading instead of receiving a plausible
 *   invention.
 * - **Never touch the reading.** Trailing «Г.» / «ГОДА» stay in the reading; they are printed on the
 *   document. They simply have no place in `dd.mm.yyyy`.
 *
 * Pure functions of a string: no image, no model, no configuration.
 */
public object Dates {

    /**
     * Month names as the documents print them, nominative and genitive: a birth certificate writes
     * «15 ОКТЯБРЯ», a SNILS «10 ДЕКАБРЯ», the 1997 internal passport the nominative «03.АВГУСТ.1989».
     */
    private val MONTHS: Map<String, Int> = mapOf(
        "ЯНВАРЬ" to 1, "ЯНВАРЯ" to 1,
        "ФЕВРАЛЬ" to 2, "ФЕВРАЛЯ" to 2,
        "МАРТ" to 3, "МАРТА" to 3,
        "АПРЕЛЬ" to 4, "АПРЕЛЯ" to 4,
        "МАЙ" to 5, "МАЯ" to 5,
        "ИЮНЬ" to 6, "ИЮНЯ" to 6,
        "ИЮЛЬ" to 7, "ИЮЛЯ" to 7,
        "АВГУСТ" to 8, "АВГУСТА" to 8,
        "СЕНТЯБРЬ" to 9, "СЕНТЯБРЯ" to 9,
        "ОКТЯБРЬ" to 10, "ОКТЯБРЯ" to 10,
        "НОЯБРЬ" to 11, "НОЯБРЯ" to 11,
        "ДЕКАБРЬ" to 12, "ДЕКАБРЯ" to 12,
    )

    /** Words a document prints next to a date that carry no date information. */
    private val NOISE: Set<String> = setOf("Г", "Г.", "ГОД", "ГОДА", "ГОДУ")

    /**
     * `[^\W\d_]+|\d+` with re.UNICODE: a run of letters, or a run of decimal digits. Punctuation is
     * never a token, which is what makes «03.АВГУСТ.1989» and «15 ОКТЯБРЯ 2020 Г.» tokenise alike.
     */
    private val TOKEN = Regex("""\p{L}+|\p{Nd}+""")

    /** dd.mm.yyyy for a real calendar date, else null (31.02 is not a date). `_as_date`. */
    private fun asDate(day: Int, month: Int, year: Int): String? {
        if (month !in 1..12 || year < 1900 || year > 2100) {
            return null
        }
        if (day < 1 || day > daysInMonth(month, year)) {
            return null
        }
        return "%02d.%02d.%04d".format(java.util.Locale.ROOT, day, month, year)
    }

    private fun daysInMonth(month: Int, year: Int): Int {
        val days = intArrayOf(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        val leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
        return if (month == 2 && leap) 29 else days[month - 1]
    }

    /**
     * Canonical `dd.mm.yyyy`, or null when the text does not yield one. `to_ddmmyyyy`.
     *
     * Handles what the documents actually print:
     *
     * - `"22.06.2010"`            -> `"22.06.2010"` (already canonical)
     * - `"15 ОКТЯБРЯ 2020 Г."`    -> `"15.10.2020"`
     * - `"10 ДЕКАБРЯ 1999 ГОДА"`  -> `"10.12.1999"`
     * - `"03.АВГУСТ.1989"`        -> `"03.08.1989"`
     * - `"5 МАЯ"`                 -> null (no year: guessing one would invent data)
     * - `"31.02.2020"`            -> null (not a calendar date)
     */
    public fun toDdMmYyyy(text: String?): String? {
        if (text.isNullOrEmpty()) {
            return null
        }
        val tokens = TOKEN.findAll(text).map { it.value.uppercase() }
            .filter { it !in NOISE && it != "Г" }
            .toList()
        if (tokens.isEmpty()) {
            return null
        }

        var day: Int? = null
        var month: Int? = null
        var year: Int? = null
        for (token in tokens) {
            if (token[0].isDigit()) {
                val value = token.toIntOrNull() ?: return null
                if (token.length == 4 && year == null) {
                    year = value
                } else if (day == null && value in 1..31) {
                    day = value
                } else if (month == null && value in 1..12) {
                    month = value
                } else if (year == null && token.length <= 2) {
                    // A two-digit year is ambiguous (26 -> 1926 or 2026?) and this module does not
                    // guess, so it is left unresolved.
                    return null
                }
                // Any other number (a third digit group, a stray five-digit run) is ignored, as the
                // reference's if/elif chain falls through without acting.
            } else {
                val resolved = MONTHS[token]
                if (resolved == null || month != null) {
                    return null
                }
                month = resolved
            }
        }

        if (day == null || month == null || year == null) {
            return null
        }
        return asDate(day, month, year)
    }

    /**
     * Canonical view of every date field that yields one. `canonical_dates`.
     *
     * A NEW map holding only the fields that converted: a field that did not convert is simply absent,
     * so the consumer can tell "no canonical form" from "canonical form equals the reading". Never
     * touches [ocr] — the reading is what the accuracy measurement compares against.
     */
    public fun canonicalDates(ocr: Map<String, String>, fields: Iterable<String>): Map<String, String> {
        val out = LinkedHashMap<String, String>()
        for (name in fields) {
            val value = ocr[name] ?: continue
            val canonical = toDdMmYyyy(value)
            if (!canonical.isNullOrEmpty()) {
                out[name] = canonical
            }
        }
        return out
    }
}
