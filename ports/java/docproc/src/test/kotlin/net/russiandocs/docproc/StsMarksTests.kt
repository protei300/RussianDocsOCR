package net.russiandocs.docproc

import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.pipeline.Ocr
import net.russiandocs.docproc.pipeline.OcrOptions
import net.russiandocs.docproc.pipeline.StageSink
import net.russiandocs.docproc.pipeline.StsMarks
import net.russiandocs.docproc.tensors.NdArray
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * STS special marks: words torn by a line break, and the leasing record — the cases of
 * `tests/test_sts_marks.py`. The tears are the four found on real STS backs (2026-10-07); the leasing texts follow
 * the two editions the generator and the real samples print.
 */
class StsMarksTests {

    @Test
    fun aKnownWordTornByTheLineBreakIsGlued() {
        assertEquals(listOf("ПО", "ДЛ", "ЛИЗИНГА", "ОТ"),
            StsMarks.glueTornWords(listOf(listOf("ПО", "ДЛ", "ЛИЗИ"), listOf("НГА", "ОТ"))))
        assertEquals(listOf("ЛИЗИНГОДАТЕЛЬ", "АО"),
            StsMarks.glueTornWords(listOf(listOf("ЛИЗИН"), listOf("ГОДАТЕЛЬ", "АО"))))
        assertEquals(listOf("ЛИЗИНГОВАЯ", "КОМПАНИЯ"),
            StsMarks.glueTornWords(listOf(listOf("ЛИЗИНГОВАЯ", "КОМ"), listOf("ПАНИЯ"))))
        assertEquals(listOf("ГАЗПРОМБАНК", "АВТОЛИЗИНГ"),
            StsMarks.glueTornWords(listOf(listOf("ГАЗПРОМБАНК", "АВТОЛ"), listOf("ИЗИНГ"))))
    }

    @Test
    fun wordsThatAreNotATornKnownWordStayApart() {
        for (lines in listOf(
            listOf(listOf("ЛИЗИНГ"), listOf("ДОГОВОР")),            // two whole words
            listOf(listOf("ЛИЗИНГОДАТЕЛЬ"), listOf("АО", "ВТБ")),   // a whole word, then a name
            listOf(listOf("ООО", "КАРКА"), listOf("ПЕТРОВ")),      // joined is not a known word
            listOf(listOf("№АЛ2682"), listOf("26/01-26")),         // numbers are never glued
        )) {
            assertEquals(lines.flatten(), StsMarks.glueTornWords(lines))
        }
    }

    @Test
    fun emptyLinesAreSkipped() {
        assertEquals(listOf("ЛИЗИНГ"),
            StsMarks.glueTornWords(listOf(emptyList(), listOf("ЛИЗИ"), listOf(""), listOf("НГ"))))
    }

    @Test
    fun theFullRecordWithTheLessorOnTheNextLine() {
        val got = StsMarks.parseLeasing("ПО ДЛ №АЛ268226/01-26 ОТ 12.03.2024 ЛИЗИНГОДАТЕЛЬ АО ВТБ ЛИЗИНГ")!!
        assertEquals("lessor_named", got.role)
        assertEquals("АО ВТБ ЛИЗИНГ", got.lessor)
        assertEquals("АЛ268226/01-26", got.contractNumber)
        assertEquals("12.03.2024", got.contractDate)
        assertEquals("12.03.2024", got.contractDateNormalized)
        assertNull(got.until)
        assertNull(got.untilNormalized)
    }

    @Test
    fun realAbbreviations() {
        var got = StsMarks.parseLeasing("ЛИЗИНГОПОЛУЧАТЕЛЬ ИВАНОВ ИВАН ЛИЗИНГОДАТЕЛЬ АО ЛИЗИНГОВАЯ КОМПАНИЯ " +
            "КАМАЗ ЛИЗИНГ ВРЕМ. УЧЕТ ДО 01.01.2027")!!
        assertEquals("lessor_named", got.role)
        assertEquals("АО ЛИЗИНГОВАЯ КОМПАНИЯ КАМАЗ ЛИЗИНГ", got.lessor)
        got = StsMarks.parseLeasing("ЛИЗИНГ ДО 31.12.2027. ДОГ ЛИЗ АХ ЭЛ/УЛН-123456/ДЛ ООО ЭЛЕМЕНТ")!!
        assertEquals("31.12.2027", got.untilNormalized)
        assertEquals("АХ ЭЛ/УЛН-123456/ДЛ", got.contractNumber)
        assertNull(got.contractDate)                      // «ДО» is the term, not the contract
        got = StsMarks.parseLeasing("ЛИЗИНГОДАТЕЛЬ ООО АВТОЛИЗИНГ, ДЕЙСТВИТЕЛЬНО ДО 01.02.2026")!!
        assertEquals("ООО АВТОЛИЗИНГ", got.lessor)
    }

    @Test
    fun theLessorEndsWhereTheContractBegins() {
        val got = StsMarks.parseLeasing("В ЛИЗИНГЕ ЛИЗИНГОДАТЕЛЬ ООО \"КАРКАДЕ\" ДОГОВОР ЛИЗИНГА №45120 ОТ 05.11.2016")!!
        assertEquals("ООО \"КАРКАДЕ\"", got.lessor)
        assertEquals("45120", got.contractNumber)
        assertEquals("05.11.2016", got.contractDateNormalized)
    }

    @Test
    fun theOwnerAsLesseeNamesNoLessor() {
        val got = StsMarks.parseLeasing("ЛИЗИНГОПОЛУЧАТЕЛЬ ДОГОВОР №1234-Л ОТ 03.04.2015")!!
        assertEquals("lessee", got.role)
        assertNull(got.lessor)
        assertEquals("1234-Л", got.contractNumber)
    }

    @Test
    fun the2010ShortForm() {
        val got = StsMarks.parseLeasing("77 1234 Л.Д 14.02.2013")!!
        assertEquals("14.02.2013", got.contractDateNormalized)
        assertNull(got.role)
        assertNull(got.lessor)
    }

    @Test
    fun noLeasingRecord() {
        // a tear left unglued is not found either
        for (text in listOf("ДУБЛИКАТ", "СМЕНА СОБСТВЕННИКА", "", null, "ЛИЗИ НГ")) {
            assertNull(StsMarks.parseLeasing(text), text)
        }
    }

    @Test
    fun thePipelineGluesOnlyTheFieldsTheOptionsName() {
        val sts = OcrOptions.forDocType("STS_1996")
        assertEquals(listOf("ПО", "ЛИЗИНГА", "ОТ"),
            Ocr.glueTorn("Special_marks", mutableListOf("ПО", "ЛИЗИ", "НГА", "ОТ"), listOf(2, 2), sts))
        assertEquals(listOf("ЛИЗИ", "НГА"),
            Ocr.glueTorn("Last_name_ru", mutableListOf("ЛИЗИ", "НГА"), listOf(1, 1), sts))
        // line lengths that do not add up to the words: left alone
        assertEquals(listOf("ЛИЗИ", "НГА"),
            Ocr.glueTorn("Special_marks", mutableListOf("ЛИЗИ", "НГА"), listOf(3), sts))
        assertTrue(OcrOptions().glueTorn.isEmpty())
    }

    private class Recorder : StageSink {
        val stages = LinkedHashMap<String, JsonElement>()
        override fun emit(stage: String, payload: JsonElement) {
            stages[stage] = payload
        }
        override fun emitArray(stage: String, array: NdArray) {}
        override fun emitImage(stage: String, image: Image) {}
    }

    @Test
    fun onlyTheFlagReachesTheResultsAndTheBackAlwaysEmitsTheStage() {
        val sink = Recorder()
        val marks = mapOf("Special_marks" to "ПО ДЛ №АЛ1/01-26 ОТ 12.03.2024 ЛИЗИНГОДАТЕЛЬ АО ВТБ ЛИЗИНГ")
        assertEquals(mapOf("leasing" to true), StsMarks.readLeasing(marks, "STSBACK", sink))
        assertEquals(JsonObject(mapOf("leasing" to JsonPrimitive(true))), sink.stages["leasing"],
            "the conformance stage carries it")

        sink.stages.clear()
        assertNull(StsMarks.readLeasing(mapOf("Special_marks" to "ДУБЛИКАТ"), "STSBACK", sink))
        assertEquals(JsonNull, sink.stages["leasing"], "an STS back without leasing still emits the stage, as null")

        sink.stages.clear()
        assertNull(StsMarks.readLeasing(mapOf("Special_marks" to "ЛИЗИНГ"), "DL", sink))
        assertTrue(sink.stages.isEmpty(), "only an STS carries special marks")

        // the front has no marks to read: no stage
        assertEquals(mapOf("leasing" to true), StsMarks.readLeasing(mapOf("Special_marks" to "ЛИЗИНГ"), "STS", sink))
        assertTrue(sink.stages.isEmpty(), "the front side does not emit the stage")
    }
}
