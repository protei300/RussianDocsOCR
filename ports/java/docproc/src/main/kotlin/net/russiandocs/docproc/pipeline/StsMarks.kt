package net.russiandocs.docproc.pipeline

import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive

/**
 * The leasing record in the special marks of a vehicle registration certificate, as far as the parser found it.
 * `parse_leasing`'s dict; a part that is not printed or not found is null.
 *
 * [role] is `lessor_named` when the marks name the lessor («ЛИЗИНГОДАТЕЛЬ ...») and `lessee` when they only say
 * the owner is the lessee («ЛИЗИНГОПОЛУЧАТЕЛЬ»). [until] is the end of the leasing term («ЛИЗИНГ ДО 31.12.2027»).
 */
public data class Leasing(
    val role: String?,
    val lessor: String?,
    val contractNumber: String?,
    val contractDate: String?,
    val contractDateNormalized: String?,
    val until: String?,
    val untilNormalized: String?,
) {
    /** `'leasing': True` in the reference's dict: a record exists exactly when one of these is built. */
    val leasing: Boolean get() = true
}

/**
 * Special marks of the vehicle registration certificate (STS): words torn by a line break, and the leasing
 * record. Port of `pipeline/sts_marks.py` (`dd37ea31`).
 *
 * The special marks are printed by the registry's printer into a narrow area and wrapped at its edge WITHOUT a
 * hyphen, so a word can be torn in two: «ЛИЗИ» at the end of one line, «НГА» at the start of the next. Measured
 * on 140 real STS backs (external set, 2026-10-07): 4 such tears on known words — «ЛИЗИ|НГА», «ЛИЗИН|ГОДАТЕЛЬ»,
 * «КОМ|ПАНИЯ», «АВТОЛ|ИЗИНГ». The position of the line end does NOT tell a tear from a break between words
 * (lines that tear end at 0.90-0.92 of the document's widest line, lines that do not at 0.87-1.00), so the tear
 * is recognised by vocabulary: the two pieces are glued only when together they make a known word and neither
 * piece is a known word on its own. Unknown words (a company name outside the list, a contract number) stay as
 * read — a wrong glue would merge two real words, which is worse than leaving a torn one.
 *
 * Pure functions of strings, like [Dates].
 *
 * **Regexes carry `(?U)` (UNICODE_CHARACTER_CLASS).** In Python 3 `\b`, `\w`, `\s` and `\d` on a `str` are
 * Unicode-aware; in `java.util.regex` they are ASCII unless asked, and `\bДОГ\w*` would then never match a
 * Cyrillic word at all — silently, on every document.
 */
public object StsMarks {

    /**
     * Words that stand as they are (no ending needed). Special-marks terms, the words seen on real backs
     * (2026-10-07) and lessor names (public leasing companies). A word is known when it is a WORD, or a word or
     * stem plus one of [ENDINGS]: «ЛИЗИНГ» + «А», «РЕГИСТРАЦИ» + «И». A bare stem is not a word —
     * «РЕГИСТРАЦИ» at a line end is a torn piece.
     */
    private val WORDS = listOf(
        "ЛИЗИНГОДАТЕЛЬ", "ЛИЗИНГОПОЛУЧАТЕЛЬ", "ЛИЗИНГ", "ДОГОВОР", "СОБСТВЕННИК",
        "ВЛАДЕЛЕЦ", "ДУБЛИКАТ", "ВЗАМЕН", "УВЭОС", "ВЫДАН", "УЧЕТ", "ТЕНТ",
        "ФУРГОН", "РЕФРИЖЕРАТОР", "ОБТЕКАТЕЛЬ", "МОЩНОСТЬ", "ДЕЙСТВИТЕЛЬНО",
        "ОБОРУДОВАНО", "УСТАНОВЛЕНО", "АДРЕС",
        "АВТОЛИЗИНГ", "ИНТЕРЛИЗИНГ", "РОСЛИЗИНГ", "ЕВРОПЛАН", "ГАЗПРОМБАНК",
        "СБЕРБАНК", "СОВКОМБАНК", "КАРКАДЕ", "АЛЬФАМОБИЛЬ", "МЭЙДЖОР", "ЭЛЕМЕНТ",
    )

    /** Stems that only occur with an ending. */
    private val STEMS = listOf(
        "ЛИЗИНГОВ", "СОБСТВЕННОСТ", "ВЛАДЕЛЬЦ", "ОБЩЕСТВ", "ОГРАНИЧЕНН",
        "ОТВЕТСТВЕННОСТ", "АКЦИОНЕРН", "ПУБЛИЧН", "КОМПАНИ", "УТРАЧЕНН",
        "РЕГИСТРАЦИ", "ВРЕМЕНН", "ЗАМЕН", "СМЕН", "ИЗМЕНЕНИ", "ПЛАТФОРМ", "ВОРОТ",
        "ПОДРАЗДЕЛЕНИ", "КОНСТРУКЦИ", "БАЛТИЙСК",
    )

    /** Noun and adjective endings of the case forms the marks use. */
    private val ENDINGS = setOf(
        "А", "Я", "У", "Ю", "Е", "И", "Ы", "О", "Ь", "ОМ", "ЕМ", "ЁМ", "ОВ", "ЕВ", "АМ",
        "ЯМ", "АХ", "ЯХ", "ОЙ", "ЕЙ", "ИЙ", "ЫЙ", "АЯ", "ЯЯ", "ОЕ", "ЕЕ", "ЫЕ", "ИЕ",
        "ИЯ", "ИИ", "ИЮ", "ЬЮ", "ОЮ", "ЕЮ", "ОГО", "ЕГО", "ОМУ", "ЕМУ", "ЫМ", "ИМ",
        "ЫХ", "ИХ", "АМИ", "ЯМИ", "ЫМИ", "ИМИ",
    )

    private val BASES = WORDS + STEMS

    private val NOT_LETTERS = Regex("[^А-ЯЁA-Z0-9]")

    private fun clean(word: String?): String = NOT_LETTERS.replace((word ?: "").uppercase(), "")

    /** A vocabulary word as is, or a word or stem with a case ending. `_known`. */
    private fun known(word: String): Boolean {
        if (word in WORDS) {
            return true
        }
        return BASES.any { base -> word.startsWith(base) && word.substring(base.length) in ENDINGS }
    }

    /**
     * Words of a multi-line field, line by line -> words with tears glued. `glue_torn_words`.
     *
     * [lines] is a list of lines, each a list of the words read on it, top to bottom. The last word of a line
     * and the first of the next are glued when together they are a known word and neither alone is: «ЛИЗИ» +
     * «НГА» -> «ЛИЗИНГА», while «ЛИЗИНГОДАТЕЛЬ» + «АО» stay two words.
     *
     * Deliberately NOT glued by the length of the line, although the registry does wrap by character count: on
     * 140 real backs (2026-10-07) the rule «a full line goes on into the next one» glued about half of its
     * cases wrongly («КРОНШ.» + «БАЗЫ», «КВТ» + «Л.С», a date line onto the line above) — the width differs
     * between documents and the reading of real marks is often noisy.
     */
    public fun glueTornWords(lines: List<List<String>>): List<String> {
        val out = ArrayList<String>()
        for (line in lines) {
            var words = line.filter { it.isNotEmpty() }
            if (words.isEmpty()) {
                continue
            }
            if (out.isNotEmpty()) {
                val tail = clean(out.last())
                val head = clean(words[0])
                if (tail.isNotEmpty() && head.isNotEmpty() && known(tail + head) &&
                    !known(tail) && !known(head)) {
                    out[out.size - 1] = out.last() + words[0]
                    words = words.drop(1)
                }
            }
            out.addAll(words)
        }
        return out
    }

    /**
     * The parts of [parseLeasing] that reach `results.leasing`: only the flag (`Pipeline.LEASING_REPORTED`),
     * by measurement (STS word-break synthetic, 240 shots, detector v10, 2026-10-07): found 54/64, false 0/176 -
     * while the lessor was right in 7/58, the contract number in 3/56, its date in 2/46. The parser finds them
     * where the reading is clean; the reading of the small special-marks print is not, and a wrong value is
     * worse than none for an integrator. Widen this when the reading improves - the parser needs no change.
     */
    private val LEASING_REPORTED = mapOf("leasing" to true)

    /**
     * The leasing flag from the STS special marks, alongside the reading (`Pipeline._read_leasing`): `{"leasing":
     * true}` or null. Only an STS carries special marks. The `leasing` stage is emitted for the BACK side alone -
     * the side with the marks - and is null when there is no leasing: a port that misses a record must differ
     * from the reference, not be skipped. [ocr] is the finished reading, [bareType] the type without its year.
     */
    internal fun readLeasing(ocr: Map<String, String>, bareType: String, sink: StageSink): Map<String, Boolean>? {
        if (!bareType.uppercase().startsWith("STS")) {
            return null
        }
        val reported = if (parseLeasing(ocr["Special_marks"]) != null) LEASING_REPORTED else null
        if (bareType.uppercase().startsWith("STSBACK")) {
            sink.emit("leasing", if (reported == null) JsonNull else JsonObject(reported.mapValues { JsonPrimitive(it.value) }))
        }
        return reported
    }

    private val NUMBER = Regex("(?U)№\\s*([0-9A-ZА-ЯЁ][0-9A-ZА-ЯЁ/\\-]*)")

    /**
     * «ДОГ ЛИЗ АХ ЭЛ/УЛН-123/ДЛ», «ПО ДОГ ЛИЗИНГА 12/34-СКТ»: the number is the first token with a digit after
     * the abbreviated «договор лизинга», within three tokens.
     */
    private val ABBR_NUMBER = Regex("(?U)\\bДОГ\\w*\\.?\\s+(?:ЛИЗ\\w*\\.?\\s+)?((?:\\S+\\s+){0,2}?\\S*\\d\\S*)")
    private const val DATE_PATTERN = "(\\d{1,2}[.,]\\d{1,2}[.,]\\d{4})"
    private val DATE = Regex("(?U)$DATE_PATTERN")

    /** The 2010 edition: «<nn> <nnnn> Л.Д <дата>». */
    private val SHORT_LEASE = Regex("(?U)\\bЛ\\s*\\.\\s*Д\\b")

    /** «по договору лизинга». */
    private val LEASE_CONTRACT = Regex("(?U)\\bПО\\s+ДЛ\\b")
    private val UNTIL = Regex("(?U)\\bЛИЗИНГ\\w*\\s+(?:ДЕЙСТВ\\w*\\s+)?ДО\\W{0,2}$DATE_PATTERN")
    private val FROM_DATE = Regex("(?U)\\bОТ\\s+$DATE_PATTERN")

    /**
     * What follows the lessor's name on real marks: the contract, the lessee, the registration term, the
     * validity, the next mark (engine power ...).
     */
    private val LESSOR_END = Regex("(?U)[.,;(]|№|\\bДОГ\\w*|\\bПО\\s+ДЛ\\b|ЛИЗИНГОПОЛУЧАТЕЛЬ|\\bВРЕМ\\w*" +
        "|\\bДЕЙСТВ\\w*|\\bМОЩНОСТ\\w*|\\bСРОК\\w*|\\bДО\\b")

    private val WHITESPACE = Regex("(?U)\\s+")

    private fun normal(dateText: String?): String? =
        if (dateText == null) null else Dates.toDdMmYyyy(dateText.replace(',', '.'))

    /**
     * The leasing record in the special marks, or null when there is none. `parse_leasing`.
     *
     * The text is the reading AFTER [glueTornWords]: a torn «ЛИЗИ НГ» is not found. Never guesses: no leasing
     * word -> null, and every part is taken only where the marks print it. Real marks are abbreviated freely and
     * read noisily, so expect the flag far more often than the parts (140 real backs, 2026-10-07). The lessee's
     * own name is never taken out: it is the owner, already read.
     */
    public fun parseLeasing(text: String?): Leasing? {
        if (text.isNullOrEmpty()) {
            return null
        }
        val t = text.uppercase().replace('Ё', 'Е').trim().split(WHITESPACE).joinToString(" ")
        if (!("ЛИЗИНГ" in t || SHORT_LEASE.containsMatchIn(t) || LEASE_CONTRACT.containsMatchIn(t))) {
            return null
        }

        var lessor: String? = null
        var role: String? = null
        if ("ЛИЗИНГОДАТЕЛЬ" in t) {
            role = "lessor_named"
            val after = t.split("ЛИЗИНГОДАТЕЛЬ", limit = 2)[1]
            // `re.split(..., maxsplit=1)[0]`: the text up to the first match (the whole text when none).
            val end = LESSOR_END.find(after)
            val head = if (end == null) after else after.substring(0, end.range.first)
            lessor = head.trim(' ', ':', '-').ifEmpty { null }   // quotes stay
        } else if ("ЛИЗИНГОПОЛУЧАТЕЛЬ" in t) {
            role = "lessee"
        }

        val number = NUMBER.find(t) ?: ABBR_NUMBER.find(t)
        var dateText: String? = null
        val from = FROM_DATE.find(t)
        if (from != null) {
            dateText = from.groupValues[1]
        } else {
            val short = SHORT_LEASE.find(t)
            if (short != null) {
                // `_DATE.search(t, short.end())`
                dateText = DATE.find(t, short.range.last + 1)?.groupValues?.get(1)
            }
        }
        val until = UNTIL.find(t)
        val untilText = until?.groupValues?.get(1)
        return Leasing(
            role = role,
            lessor = lessor,
            contractNumber = number?.groupValues?.get(1)?.trim(),
            contractDate = dateText,
            contractDateNormalized = normal(dateText),
            until = untilText,
            untilNormalized = normal(untilText),
        )
    }
}
