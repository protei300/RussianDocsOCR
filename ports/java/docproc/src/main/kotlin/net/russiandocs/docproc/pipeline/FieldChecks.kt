package net.russiandocs.docproc.pipeline

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import java.time.DateTimeException
import java.time.LocalDate

/** One check that ran over a field: `{id, kind, result, reference?}`. [reference] is the value the field was compared with. */
public data class FieldCheck(
    val id: String,
    val kind: String,
    val result: String,
    val reference: String? = null,
)

/** A soft remark that never changes a verdict: `{id, text}`. */
public data class FieldNote(val id: String, val text: String)

/** The verdict for one field: `{status, checks, notes}`. [status] is `verified`, `mismatch` or `unchecked`. */
public data class FieldVerdict(
    val status: String,
    val checks: List<FieldCheck>,
    val notes: List<FieldNote>,
)

/**
 * A verdict for every recognised field: verified, mismatch or unchecked. Port of `validation/fields.py`
 * (decision No. 152; check ids are the rows of
 * `docs/validation-checks.md`).
 *
 * - `verified` — at least one CONFIRMING check agreed and no hard check disagreed;
 * - `mismatch` — a hard check disagreed: almost certainly a reading error;
 * - `unchecked` — nothing applicable, or nothing to lean on.
 *
 * A CONFIRMING check is one where two independent readings agree or a check digit holds; a REFUTING one is a
 * plausibility rule that only counts when it fails; a note is a usual-but-not-guaranteed rule that never
 * changes a verdict. A disagreement between two fields marks BOTH. The reading itself is never changed.
 *
 * Pure functions of the OCR dict and the document type: no image, no model. The order of the entries in
 * `checks` is part of the contract (the conformance stage compares it): dates, number forms, the zone, the
 * SNILS check number, then the soft note — and inside each exactly as in the reference.
 *
 * Digits are ASCII only here; the reference's `\d` / `isdigit()` would also accept other scripts' digits, which
 * the recognition never produces.
 */
public object FieldChecks {

    public const val VERIFIED: String = "verified"
    public const val MISMATCH: String = "mismatch"
    public const val UNCHECKED: String = "unchecked"

    public const val CONFIRMING: String = "confirming"
    public const val REFUTING: String = "refuting"

    private const val OK = "ok"

    /**
     * Cyrillic -> zone characters of the internal passport's top line, FMS order No. 279 (2011-06-30), the
     * "modernised clear" table: letters with no Latin counterpart become digits.
     */
    private val ZONE_LETTERS: Map<Char, Char> = mapOf(
        'А' to 'A', 'Б' to 'B', 'В' to 'V', 'Г' to 'G', 'Д' to 'D', 'Е' to 'E', 'Ё' to '2',
        'Ж' to 'J', 'З' to 'Z', 'И' to 'I', 'Й' to 'Q', 'К' to 'K', 'Л' to 'L', 'М' to 'M',
        'Н' to 'N', 'О' to 'O', 'П' to 'P', 'Р' to 'R', 'С' to 'S', 'Т' to 'T', 'У' to 'U',
        'Ф' to 'F', 'Х' to 'H', 'Ц' to 'C', 'Ч' to '3', 'Ш' to '4', 'Щ' to 'W', 'Ъ' to 'X',
        'Ы' to 'Y', 'Ь' to '9', 'Э' to '6', 'Ю' to '7', 'Я' to '8',
    )

    /** Sex as printed -> the zone's letter. The page prints «МУЖ.»/«ЖЕН.», the external passport also «M»/«F». */
    private val SEX: Map<Char, String> = mapOf('М' to "M", 'Ж' to "F", 'M' to "M", 'F' to "F")

    /** Years between issue and expiry: licences 10; external passports 10 (biometric) or 5 (2003 form) — either accepted for both. */
    private val VALIDITY_YEARS: Map<String, List<Int>> = mapOf("DL" to listOf(10), "EXTPASSPORT" to listOf(5, 10))

    /** Minimum age at issue: 14 for the internal passport, 16 for a licence. SNILS and the external passport have none. */
    private val MIN_AGE: Map<String, Int> = mapOf("INTPASSPORT" to 14, "DL" to 16)

    /** Soft: the second pair of the internal passport's series is the year the blank was printed, within this many years of the issue. */
    private const val SERIES_YEAR_WINDOW = 3

    private val DATE_FIELDS = listOf("Birth_date", "Issue_date", "Expiration_date")

    /** The printed form of numbers, whitespace collapsed (full match): field, check id, pattern. */
    private class NumberForm(val field: String, val id: String, val pattern: Regex)

    private val NUMBER_FORMS: Map<String, List<NumberForm>> = mapOf(
        "INTPASSPORT" to listOf(
            NumberForm("Licence_number", "PASS-1", Regex("""\d{2} \d{2} \d{6}""")),
            NumberForm("Issue_organisation_code", "PASS-4", Regex("""\d{3}-\d{3}""")),
        ),
        "EXTPASSPORT" to listOf(NumberForm("Licence_number", "EXT-1", Regex("""\d{2} \d{7}"""))),
        "SNILS" to listOf(NumberForm("Licence_number", "SNILS-1", Regex("""\d{3}-\d{3}-\d{3} \d{2}"""))),
    )

    private val LINE1_PREFIX = mapOf("INTPASSPORT" to "PNRUS", "EXTPASSPORT" to "P<RUS")
    private val LATIN: Set<Char> = ('A'..'Z').toSet() + '<'
    private val NAME_CHARS: Map<String, Set<Char>> = mapOf(
        "INTPASSPORT" to LATIN + "2346789".toSet(),
        "EXTPASSPORT" to LATIN,
    )

    /**
     * The document family the checks are defined for, or null. Order matters as in `OcrOptionsClass.make_options`:
     * the registration page and the licence back side contain the names of the families they are not. Everything
     * outside the four families is `unchecked` throughout — their rules were not measured.
     */
    public fun family(docType: String?): String? {
        val label = (docType ?: "").uppercase()
        if ("INTPASSPORTADDR" in label || "DLBACK" in label) {
            return null
        }
        for (name in listOf("INTPASSPORT", "EXTPASSPORT", "SNILS", "DL")) {
            if (label.startsWith(name)) {
                return name
            }
        }
        return null
    }

    /** Collects check results per field and turns them into verdicts. */
    private class Verdicts(val ocr: Map<String, String>) {
        private class Entry {
            val checks = ArrayList<FieldCheck>()
            val notes = ArrayList<FieldNote>()
        }

        private val entries = LinkedHashMap<String, Entry>().also { m -> ocr.keys.forEach { m[it] = Entry() } }

        /** The field's reading, stripped, or null when absent or empty. */
        fun value(name: String): String? = ocr[name]?.trim()?.takeIf { it.isNotEmpty() }

        /** Records one check over one or several fields (only those present in `ocr`), one [reference] for all. */
        fun add(fields: List<String>, id: String, kind: String, agrees: Boolean, reference: String? = null) {
            for (name in fields) {
                entries[name]?.checks?.add(FieldCheck(id, kind, if (agrees) OK else MISMATCH, reference))
            }
        }

        fun add(name: String, id: String, kind: String, agrees: Boolean, reference: String? = null) =
            add(listOf(name), id, kind, agrees, reference)

        /** As [add], with a reference per field (a field without one gets no `reference`). */
        fun add(fields: List<String>, id: String, kind: String, agrees: Boolean, references: Map<String, String?>) {
            for (name in fields) {
                entries[name]?.checks?.add(FieldCheck(id, kind, if (agrees) OK else MISMATCH, references[name]))
            }
        }

        fun note(fields: List<String>, id: String, text: String) {
            for (name in fields) {
                entries[name]?.notes?.add(FieldNote(id, text))
            }
        }

        fun result(): Map<String, FieldVerdict> {
            val out = LinkedHashMap<String, FieldVerdict>()
            for ((name, entry) in entries) {
                val status = when {
                    entry.checks.any { it.result == MISMATCH } -> MISMATCH
                    entry.checks.any { it.kind == CONFIRMING } -> VERIFIED
                    else -> UNCHECKED
                }
                out[name] = FieldVerdict(status, entry.checks.toList(), entry.notes.toList())
            }
            return out
        }
    }

    private fun asDate(name: String, text: String?): LocalDate? {
        val canonical = if (text.isNullOrEmpty()) null else Dates.canonicalDate(name, text)
        if (canonical.isNullOrEmpty()) {
            return null
        }
        val (day, month, year) = canonical.split('.').map { it.toInt() }
        return LocalDate.of(year, month, day)
    }

    /** The same day [years] later; 29.02 maps to both 28.02 and 01.03. */
    private fun addYears(day: LocalDate, years: Int): List<LocalDate> {
        val year = day.year + years
        return try {
            listOf(LocalDate.of(year, day.monthValue, day.dayOfMonth))
        } catch (e: DateTimeException) {
            listOf(LocalDate.of(year, 2, 28), LocalDate.of(year, 3, 1))
        }
    }

    private fun age(birth: LocalDate, at: LocalDate): Int {
        val before = (at.monthValue to at.dayOfMonth).let { (m, d) ->
            m < birth.monthValue || (m == birth.monthValue && d < birth.dayOfMonth)
        }
        return at.year - birth.year - if (before) 1 else 0
    }

    private fun fmt(day: LocalDate): String =
        "%02d.%02d.%04d".format(java.util.Locale.ROOT, day.dayOfMonth, day.monthValue, day.year)

    private fun digits(text: String?): String = (text ?: "").filter { it in '0'..'9' }

    /** `' '.join(text.split())`: whitespace runs collapsed to one space, the ends dropped. */
    private fun collapseSpaces(text: String): String {
        val words = ArrayList<String>()
        val current = StringBuilder()
        for (ch in text) {
            if (ch.isWhitespace()) {
                if (current.isNotEmpty()) {
                    words.add(current.toString())
                    current.setLength(0)
                }
            } else {
                current.append(ch)
            }
        }
        if (current.isNotEmpty()) {
            words.add(current.toString())
        }
        return words.joinToString(" ")
    }

    /** GEN-1, GEN-2, PASS-7 / DL-4, DL-2 / EXT-3. */
    private fun checkDates(v: Verdicts, fam: String, today: LocalDate) {
        val dates = HashMap<String, LocalDate>()
        for (name in DATE_FIELDS) {
            val text = v.value(name) ?: continue
            val parsed = asDate(name, text)
            // GEN-1: every reference date in samples/ converts, so a reading that does not is not a date.
            v.add(name, "GEN-1", REFUTING, parsed != null)
            if (parsed != null) {
                dates[name] = parsed
            }
        }

        val birth = dates["Birth_date"]
        val issue = dates["Issue_date"]
        val expiry = dates["Expiration_date"]
        for ((name, day) in listOf("Birth_date" to birth, "Issue_date" to issue)) {
            if (day != null) {
                v.add(name, "GEN-2", REFUTING, !day.isAfter(today))
            }
        }
        if (birth != null && issue != null) {
            v.add(listOf("Birth_date", "Issue_date"), "GEN-2", REFUTING, birth.isBefore(issue))
            val minAge = MIN_AGE[fam]
            if (minAge != null) {
                val id = if (fam == "INTPASSPORT") "PASS-7" else "DL-4"
                v.add(listOf("Birth_date", "Issue_date"), id, REFUTING, age(birth, issue) >= minAge)
            }
        }
        val years = VALIDITY_YEARS[fam]
        if (years != null && issue != null && expiry != null) {
            val expected = years.flatMap { addYears(issue, it) }
            val id = if (fam == "DL") "DL-2" else "EXT-3"
            if (years.size == 1 && expected.size == 1) {
                val back = addYears(expiry, -years[0])
                val references = mapOf(
                    "Expiration_date" to fmt(expected[0]),
                    "Issue_date" to if (back.size == 1) fmt(back[0]) else null,
                )
                v.add(listOf("Issue_date", "Expiration_date"), id, CONFIRMING, expiry in expected, references)
            } else {
                v.add(listOf("Issue_date", "Expiration_date"), id, CONFIRMING, expiry in expected)
            }
        }
    }

    private fun isNameFiller(ch: Char): Boolean = ch.isWhitespace() || ch == '-' || ch == '<'

    /** A name for comparison with the zone: no spaces, hyphens or fillers. */
    private fun zoneNames(text: String): String = text.uppercase().filterNot { isNameFiller(it) }

    /**
     * A Cyrillic name as the internal passport's zone writes it. Ё and Е are made equal on both sides (Ё is
     * '2' in the zone): whether a name was typed with Ё is the clerk's choice, not shared by page and zone.
     */
    private fun cyrillicToZone(text: String): String =
        zoneNames(text).replace('Ё', 'Е').map { ZONE_LETTERS[it] ?: it }.joinToString("")

    /** Equal, or — when the zone's name area is full — the page starts with it. */
    private fun namesAgree(zone: String, page: String, truncated: Boolean): Boolean {
        if (zone.isEmpty() || page.isEmpty()) {
            return false
        }
        if (zone == page) {
            return true
        }
        return truncated && page.startsWith(zone)
    }

    /** MRZ-1 .. MRZ-9 (docs/validation-checks.md, section MRZ). */
    private fun checkMrz(v: Verdicts, fam: String) {
        val mrzText = v.value("MRZ") ?: return
        val checked = MrzChecks.validateMrz(mrzText)
        // The check digits cover line 2 only, so for the zone as a whole they can refute but not confirm; the
        // confirmation is the names of line 1 agreeing with the page (MRZ-8 below).
        v.add("MRZ", "MRZ-1", REFUTING, checked.ok)
        val digitsOk = checked.checks
        val internal = fam == "INTPASSPORT"
        // The layout is chosen by the classifier's type, not by the zone's own first characters, which may be misread.
        val zone = MrzChecks.parseMrz(mrzText, internal)

        fun trusted(vararg parts: String): Boolean = parts.all { digitsOk[it] == OK }

        // MRZ-2: the number, in the form the page prints it.
        val pageNumber = digits(v.value("Licence_number"))
        val numberParts = if (internal) arrayOf("document_number", "personal_number") else arrayOf("document_number")
        val zoneNumber = MrzChecks.printedNumber(zone, internal)
        if (pageNumber.isNotEmpty() && !zoneNumber.isNullOrEmpty() && trusted(*numberParts)) {
            v.add("Licence_number", "MRZ-2", CONFIRMING, MrzChecks.numbersAgree(zoneNumber, pageNumber), zoneNumber)
        }

        // MRZ-3..MRZ-5: dates, each behind its own check digit.
        class DatePart(val page: String, val zoneValue: String?, val part: String, val id: String)

        val dateParts = ArrayList<DatePart>()
        dateParts.add(DatePart("Birth_date", zone.birthDate, "birth_date", "MRZ-3"))
        if (internal) {
            dateParts.add(DatePart("Issue_date", zone.issueDate, "personal_number", "MRZ-5"))
        } else {
            dateParts.add(DatePart("Expiration_date", zone.expiryDate, "expiry_date", "MRZ-4"))
        }
        for (p in dateParts) {
            val text = v.value(p.page)
            if (text == null || p.zoneValue.isNullOrEmpty() || !trusted(p.part)) {
                continue
            }
            val pageValue = asDate(p.page, text)
            v.add(p.page, p.id, CONFIRMING, pageValue != null && fmt(pageValue) == p.zoneValue, p.zoneValue)
        }

        // MRZ-6: the subdivision code (internal passport only).
        val codePage = digits(v.value("Issue_organisation_code"))
        val codeZone = zone.subdivisionCode
        if (internal && codePage.isNotEmpty() && !codeZone.isNullOrEmpty() && trusted("personal_number")) {
            v.add(
                "Issue_organisation_code", "MRZ-6", CONFIRMING, codePage == codeZone,
                "${codeZone.substring(0, 3)}-${codeZone.substring(3)}",
            )
        }

        // MRZ-7: sex. No check digit covers it, but two independent readings agreeing is a confirmation all the same.
        val sexZone = zone.sex
        if (!sexZone.isNullOrEmpty()) {
            for (name in listOf("Sex_ru", "Sex_en")) {
                val text = v.value(name) ?: continue
                v.add(name, "MRZ-7", CONFIRMING, SEX[text[0].uppercaseChar()] == sexZone, sexZone)
            }
        }

        // MRZ-8: names from the top line. No check digit covers that line, so a disagreement says "one of the two
        // readings is wrong", not which - and a line of the wrong length may have lost part of a name: not compared.
        val lines = mrzText.trim().let { MrzChecks.splitLines(it) }
        val line1 = lines.firstOrNull() ?: ""
        val tail = if (line1.length > 5) line1.substring(5) else ""
        val wellFormed = line1.length == 44 && line1.startsWith(LINE1_PREFIX.getValue(fam)) &&
            "<<" in tail && NAME_CHARS.getValue(fam).containsAll(tail.toSet())
        v.add("MRZ", "MRZ-9", REFUTING, wellFormed)
        if (!wellFormed) {
            return
        }
        val truncated = !lines[0].endsWith("<")
        var surname = (zone.surname ?: "").replace(" ", "")
        val givenText = zone.givenNames ?: ""
        val space = givenText.indexOf(' ')
        var zoneFirst = if (space >= 0) givenText.substring(0, space) else givenText
        var zoneRest = if (space >= 0) givenText.substring(space + 1).replace(" ", "") else ""
        val convert: (String) -> String
        val names: List<String>
        if (internal) {
            // Ё is '2' in the zone; made equal to Е on both sides (cyrillicToZone).
            surname = surname.replace('2', 'E')
            zoneFirst = zoneFirst.replace('2', 'E')
            zoneRest = zoneRest.replace('2', 'E')
            convert = ::cyrillicToZone
            names = listOf("Last_name_ru", "First_name_ru", "Middle_name_ru")
        } else {
            convert = ::zoneNames
            names = listOf("Last_name_en", "First_name_en", "Middle_name_en")
        }
        val agreed = ArrayList<Boolean>()
        for ((name, zoneValue) in names.zip(listOf(surname, zoneFirst, zoneRest))) {
            val text = v.value(name)
            if (text == null || zoneValue.isEmpty()) {
                continue
            }
            agreed.add(namesAgree(zoneValue, convert(text), truncated))
            v.add(name, "MRZ-8", CONFIRMING, agreed.last(), zoneValue)
        }
        // The zone itself is confirmed only when every compared name agrees. A disagreement does not mark it: the
        // zone's own misreadings are caught for certain by MRZ-1 and MRZ-9, and which of the two readings of a
        // name is wrong is unknown - the page field already carries that mismatch.
        if (agreed.isNotEmpty() && agreed.all { it }) {
            v.add("MRZ", "MRZ-8", CONFIRMING, true)
        }
    }

    /** SNILS check number over nine digits (weights 9..1, mod 101). */
    private fun snilsControl(body: String): Int {
        var total = 0
        for ((i, ch) in body.withIndex()) {
            total += (ch - '0') * (9 - i)
        }
        while (total > 101) {
            total %= 101
        }
        return if (total == 100 || total == 101) 0 else total
    }

    /** PASS-1, PASS-4, EXT-1, SNILS-1: the printed form of numbers (refuting). */
    private fun checkForms(v: Verdicts, fam: String) {
        for (form in NUMBER_FORMS[fam] ?: emptyList()) {
            val text = v.value(form.field) ?: continue
            v.add(form.field, form.id, REFUTING, form.pattern.matches(collapseSpaces(text)))
        }
    }

    /** SNILS-2 (the check number); its form is SNILS-1 in [checkForms]. */
    private fun checkSnils(v: Verdicts) {
        val text = v.value("Licence_number") ?: return
        val digits = digits(text)
        if (digits.length != 11) {
            return
        }
        val body = digits.substring(0, 9)
        // Numbers up to 001-001-998 are issued without a valid check number.
        if (body.toInt() <= 1001998) {
            return
        }
        val expected = snilsControl(body)
        v.add(
            "Licence_number", "SNILS-2", CONFIRMING, digits.substring(9).toInt() == expected,
            "%02d".format(java.util.Locale.ROOT, expected),
        )
    }

    /** PASS-3 (soft): the blank's year in the series against the issue year. */
    private fun noteSeriesYear(v: Verdicts) {
        val digits = digits(v.value("Licence_number"))
        val issue = asDate("Issue_date", v.value("Issue_date"))
        if (digits.length != 10 || issue == null) {
            return
        }
        val yy = digits.substring(2, 4).toInt()
        val blankYear = if (2000 + yy <= issue.year + SERIES_YEAR_WINDOW) 2000 + yy else 1900 + yy
        if (kotlin.math.abs(issue.year - blankYear) > SERIES_YEAR_WINDOW) {
            v.note(
                listOf("Licence_number", "Issue_date"), "PASS-3",
                "series printed in $blankYear, issued in ${issue.year}",
            )
        }
    }

    /**
     * Verdict per field of [ocr]: `{name: {status, checks, notes}}`. Every key of [ocr] gets an entry, so "no
     * entry" never has to be told apart from "unchecked". [today] is read from the clock, as the reference's
     * `date.today()` (the "not in the future" rule); tests pin it.
     */
    public fun fieldChecks(
        ocr: Map<String, String>?,
        docType: String?,
        today: LocalDate = LocalDate.now(),
    ): Map<String, FieldVerdict> {
        val v = Verdicts(ocr ?: emptyMap())
        val fam = family(docType) ?: return v.result()
        checkDates(v, fam, today)
        checkForms(v, fam)
        if (fam == "INTPASSPORT" || fam == "EXTPASSPORT") {
            checkMrz(v, fam)
        }
        if (fam == "SNILS") {
            checkSnils(v)
        }
        if (fam == "INTPASSPORT") {
            noteSeriesYear(v)
        }
        return v.result()
    }

    /** The JSON of the `checks` stage: key order id, kind, result, reference (when there is one). */
    public fun toJson(checks: Map<String, FieldVerdict>): JsonObject =
        JsonObject(
            checks.mapValues { (_, verdict) ->
                JsonObject(
                    mapOf(
                        "status" to JsonPrimitive(verdict.status),
                        "checks" to JsonArray(
                            verdict.checks.map { c ->
                                val entry = LinkedHashMap<String, JsonElement>()
                                entry["id"] = JsonPrimitive(c.id)
                                entry["kind"] = JsonPrimitive(c.kind)
                                entry["result"] = JsonPrimitive(c.result)
                                if (c.reference != null) {
                                    entry["reference"] = JsonPrimitive(c.reference)
                                }
                                JsonObject(entry)
                            },
                        ),
                        "notes" to JsonArray(
                            verdict.notes.map { n ->
                                JsonObject(mapOf("id" to JsonPrimitive(n.id), "text" to JsonPrimitive(n.text)))
                            },
                        ),
                    ),
                )
            },
        )
}
