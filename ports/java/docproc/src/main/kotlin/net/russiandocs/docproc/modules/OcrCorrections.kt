package net.russiandocs.docproc.modules

/**
 * The text-only fixes the reference applies after decoding.
 *
 * Every one is a pure string function, which is why they are here rather than in the engine: they are the
 * part of OCR post-processing that can be unit-tested without a model.
 */
public object OcrCorrections {

    /**
     * Normalises a date to `dd.MM.yyyy`.
     *
     * **This function has THREE outcomes, not two, and the third is invisible from inside it.** On eight
     * digits forming a real date it returns the formatted date. On anything that is not eight digits it
     * returns the SUBSTITUTED string — `O` to `0` and `-` to `.` applied. But on eight digits that are NOT a
     * valid date, the reference's `strptime` raises and its `except` returns the argument THAT HANDLER
     * received, which never saw the substitutions — so the original comes back untouched.
     *
     * Concretely: `"123"` comes back substituted, `"O6-13-1985"` comes back completely unchanged. Confirmed
     * in the .NET port by CALLING the reference, not by reading it — a first implementation that returned the
     * substituted string in both cases would differ only on malformed input, where nobody looks.
     */
    public fun checkDdmmyyyy(date: String): String {
        val substituted = date.replace("O", "0").replace("-", ".")

        val digits = substituted.filter { it in '0'..'9' }
        if (digits.length != 8) {
            return substituted
        }

        val day = digits.substring(0, 2).toInt()
        val month = digits.substring(2, 4).toInt()
        val year = digits.substring(4, 8).toInt()

        // The ORIGINAL, not the substituted string — see the note above.
        return if (isValidDate(day, month, year)) {
            "%02d.%02d.%04d".format(day, month, year)
        } else {
            date
        }
    }

    private fun isValidDate(day: Int, month: Int, year: Int): Boolean {
        if (month < 1 || month > 12 || day < 1 || year < 1 || year > 9999) {
            return false
        }
        val days = intArrayOf(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        var limit = days[month - 1]
        if (month == 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) {
            limit = 29
        }
        return day <= limit
    }

    /**
     * Latin sex: anything containing M is male, everything else is female.
     *
     * Note the asymmetry — it is not "F means female". An unreadable crop therefore becomes F rather than
     * empty, which is the reference's behaviour and not obviously the right one; reproduced because the
     * goldens encode it.
     */
    public fun checkEnSex(sex: String): String =
        if (sex.trimStart('.').uppercase().replace(".", "").contains('M')) "M" else "F"

    /** Cyrillic sex, with the same asymmetry. The М here is CYRILLIC U+041C. */
    public fun checkRusSex(sex: String): String =
        if (sex.trimStart('.').uppercase().replace(".", "").contains('М')) "М" else "Ж"

    /** Keeps only the characters a licence class can contain. */
    public fun checkDriverClass(driverClass: String): String {
        val allowed = "ABCDEM1"
        return driverClass.replace(" ", "").filter { it in allowed }
    }

    /**
     * Strips LEADING dots only.
     *
     * `lstrip('.')`, not `strip('.')`. Names pick up a spurious leading dot from the crop edge; a trailing one
     * is rare and the reference leaves it, so trimming both ends would diverge on exactly the cases where it
     * matters.
     */
    public fun stripEdgeDots(name: String): String = name.trimStart('.')
}
