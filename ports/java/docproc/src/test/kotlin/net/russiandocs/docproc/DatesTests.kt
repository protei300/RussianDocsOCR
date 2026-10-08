package net.russiandocs.docproc

import net.russiandocs.docproc.pipeline.Dates
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

/**
 * The record date of a civil-registry entry (`Act_date` of a birth certificate) — the cases of
 * `tests/test_date_canon.py` that belong to it. Pure functions of a string, so pinned here rather than
 * only through a conformance run that needs models.
 */
class DatesTests {

    /** The general converter learned the printed words «месяца» and «числа». */
    @Test
    fun generalConverterSkipsRecordDateWords() {
        assertEquals("15.06.2010", Dates.toDdMmYyyy("2010 ГОДА ИЮНЯ МЕСЯЦА 15 ЧИСЛА"))
        assertEquals("15.06.2010", Dates.toDdMmYyyy("2010 года июня месяца 15"))
        assertEquals("02.03.2025", Dates.toDdMmYyyy("2 МАРТА 2025 Г."))
        assertNull(Dates.toDdMmYyyy("2010 ГОДА ИЮНЯ МЕСЯЦА 31 ЧИСЛА")) // 31 июня не бывает
        assertNull(Dates.toDdMmYyyy("2010 ГОДА ИЮНЯ МЕСЯЦА ЧИСЛА"))    // нет дня
    }

    @Test
    fun recordDateReadsThroughGlueAndMisreadWords() {
        val cases = listOf(
            "2015ГОДАИЮНЯИЕСЯЦА16" to "16.06.2015",   // склеено, «месяца» прочитано с ошибкой
            "2026ТОДАМАЯНСЯЦА3" to "03.05.2026",
            "2002ТОДАИЮНЯ,МЕСЯЦА18" to "18.06.2002",
            "2003 ДЕКАБРЯ МЕСЯЦА 27" to "27.12.2003",
            "2 МАРТА 2025 Г." to "02.03.2025",        // обычный порядок берёт общий разбор
        )
        for ((printed, canonical) in cases) {
            assertEquals(canonical, Dates.recordDateToDdMmYyyy(printed), printed)
        }
    }

    @Test
    fun recordDateStillRefusesRatherThanGuesses() {
        val refused = listOf(
            "2010 ГОДА ЦЮЛЯ МЕСЯЦА 17",   // сам месяц прочитан неверно
            "2020 ГОДА ИЮЛЯ МЕСЯЦА",      // нет дня
            "2020 ЯНВАРЯ 110201114335",   // в рамку попал номер записи
            ".110266032",
            "2010 ИЮНЯ МАЯ 15",           // два месяца
            "2010ГОДАИЮНЯ 15 16",         // два дня
        )
        for (text in refused) {
            assertNull(Dates.recordDateToDdMmYyyy(text), text)
        }
    }

    @Test
    fun onlyTheRecordDateGetsTheLenientReading() {
        val ocr = mapOf("Act_date" to "2015ГОДАИЮНЯИЕСЯЦА16", "Issue_date" to "2015ГОДАИЮНЯИЕСЯЦА16")
        assertEquals(
            mapOf("Act_date" to "16.06.2015"),
            Dates.canonicalDates(ocr, listOf("Act_date", "Issue_date")),
        )
    }

    /**
     * BIRTHCERT_1998 prints «10» ЯНВАРЯ 2013 г., the field box starts on the opening quote, and «» are not in
     * the Cyrillic engine's alphabet, so it reads the quote as the nearest letter (issue #23). A single letter
     * right next to the day is a read quote, not a word. Cases of `tests/test_date_canon.py`.
     */
    @Test
    fun aQuoteReadAsALetterNextToTheDayIsDropped() {
        val cases = listOf(
            "И 10 ЯНВАРЯ 2013" to "10.01.2013",      // opening quote
            "И10 ЯНВАРЯ 2013" to "10.01.2013",       // no space
            "И 10 Н ЯНВАРЯ 2013" to "10.01.2013",    // both quotes
            "10 П ЯНВАРЯ 2013 Г." to "10.01.2013",   // closing quote
        )
        for ((printed, canonical) in cases) {
            assertEquals(canonical, Dates.toDdMmYyyy(printed), printed)
        }
    }

    /** The rule is narrow: a longer word, or a letter not touching the day, still refuses. */
    @Test
    fun onlyALoneLetterAtTheDayIsDropped() {
        assertNull(Dates.toDdMmYyyy("ИЗ 10 ЯНВАРЯ 2013"))   // a word, not a single letter
        assertNull(Dates.toDdMmYyyy("10 ЯНВАРЯ И 2013"))   // the letter is at the year, not the day
        assertNull(Dates.toDdMmYyyy("И 10 ЯНВАРЯ"))        // the quote is dropped, but there is no year
    }
}
