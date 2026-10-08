package net.russiandocs.docproc

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.pipeline.DateReread
import net.russiandocs.docproc.pipeline.RereadDates
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A date field re-read whole when the word-by-word reading is not a date — the cases of
 * `tests/test_dates_read_whole.py`, with a stand-in reader in place of the OCR engine. The values are made
 * up («14» МАРТА 2019). The rule replaces a reading only when it does NOT convert to dd.mm.yyyy and the
 * whole-line reading DOES.
 */
class RereadDatesTests {

    @BeforeTest
    fun loadNatives() {
        NativeLibraries.load()
    }

    private fun line(): Image = Image.wrap(Mat.zeros(22, 355, CvType.CV_8UC3))

    private class Reader(private val text: String) : (String, Image) -> String {
        var calls = 0
        override fun invoke(field: String, line: Image): String {
            calls++
            return text
        }
    }

    @Test
    fun aDateThatLostItsDayIsReadWhole() {
        val ocr = mutableMapOf("Issue_date" to "МАРТА 2019")
        line().use { l ->
            val done = RereadDates.run(ocr, mapOf("Issue_date" to listOf(l)), Reader("14МАРТА2019"))
            assertEquals("14МАРТА2019", ocr["Issue_date"])
            assertEquals(listOf(DateReread("Issue_date", "МАРТА 2019", "14МАРТА2019")), done)
        }
    }

    @Test
    fun aDateThatAlreadyConvertsIsNeverReRead() {
        val ocr = mutableMapOf("Issue_date" to "28 ИЮЛЯ 2010")
        val reader = Reader("14МАРТА2019")
        line().use { l ->
            val done = RereadDates.run(ocr, mapOf("Issue_date" to listOf(l)), reader)
            assertEquals("28 ИЮЛЯ 2010", ocr["Issue_date"])
            assertEquals(0, reader.calls)
            assertTrue(done.isEmpty())
        }
    }

    @Test
    fun aWholeReadingThatIsNoDateEitherChangesNothing() {
        val ocr = mutableMapOf("Issue_date" to "МАРТА 2019")
        line().use { l ->
            val done = RereadDates.run(ocr, mapOf("Issue_date" to listOf(l)), Reader("МАРТА2019"))
            assertEquals("МАРТА 2019", ocr["Issue_date"])
            assertTrue(done.isEmpty())
        }
    }

    @Test
    fun nothingRememberedMeansNothingDone() {
        val ocr = mutableMapOf("Issue_date" to "МАРТА 2019")
        val reader = Reader("14МАРТА2019")
        assertTrue(RereadDates.run(ocr, emptyMap(), reader).isEmpty())
        assertEquals("МАРТА 2019", ocr["Issue_date"])
        assertEquals(0, reader.calls)
    }

    @Test
    fun theLinesOfOneFieldAreJoinedWithASpace() {
        val ocr = mutableMapOf("Issue_date" to "")
        val texts = ArrayDeque(listOf("14МАРТА", "2019"))
        line().use { a -> line().use { b ->
            RereadDates.run(ocr, mapOf("Issue_date" to listOf(a, b))) { _, _ -> texts.removeFirst() }
        } }
        assertEquals("14МАРТА 2019", ocr["Issue_date"])
    }
}
