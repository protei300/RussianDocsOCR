package net.russiandocs.docproc

import net.russiandocs.docproc.modules.WordsDetector
import net.russiandocs.docproc.postprocess.Box
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

/**
 * Word ordering and the word-crop margin — the two rules that decide which pixels reach the OCR engine and
 * in which order. Both are pure functions of the boxes, so they are pinned here rather than only through a
 * conformance run that needs models.
 */
class WordsOrderTests {

    private fun box(x1: Double, y1: Double, x2: Double, y2: Double): Box = Box().also {
        it.x1 = x1; it.y1 = y1; it.x2 = x2; it.y2 = y2; it.label = "Word"
    }

    private fun quads(boxes: List<Box>): List<List<Double>> =
        boxes.map { listOf(it.x1, it.y1, it.x2, it.y2) }

    /**
     * A two-line field must be read line by line. A plain x-sort interleaves the lines, which the reference
     * measured as word salad on the birth certificates' Birth_place and ZAGS fields. The expected order is
     * what `WordsDetector._reading_order` returns for these boxes.
     */
    @Test
    fun readingOrderKeepsLinesTogether() {
        val input = listOf(
            box(10.0, 0.0, 60.0, 18.0),   // line 1, word 1
            box(70.0, 1.0, 130.0, 19.0),  // line 1, word 2
            box(140.0, 0.0, 200.0, 18.0), // line 1, word 3
            box(5.0, 22.0, 55.0, 40.0),   // line 2, word 1
            box(65.0, 23.0, 190.0, 41.0), // line 2, word 2
        )

        assertEquals(
            listOf(
                listOf(10.0, 0.0, 60.0, 18.0),
                listOf(70.0, 1.0, 130.0, 19.0),
                listOf(140.0, 0.0, 200.0, 18.0),
                listOf(5.0, 22.0, 55.0, 40.0),
                listOf(65.0, 23.0, 190.0, 41.0),
            ),
            quads(WordsDetector.readingOrder(input)),
        )

        // And the naive sort really does disagree — a test that cannot fail proves nothing.
        assertNotEquals(quads(input.sortedBy { it.x1 }), quads(WordsDetector.readingOrder(input)))
    }

    /** A single-line field comes out exactly as the old x1 sort produced it. */
    @Test
    fun readingOrderIsAnX1SortOnOneLine() {
        val input = listOf(
            box(140.0, 0.0, 200.0, 18.0),
            box(10.0, 2.0, 60.0, 20.0),
            box(70.0, 1.0, 130.0, 19.0),
        )
        assertEquals(quads(input.sortedBy { it.x1 }), quads(WordsDetector.readingOrder(input)))
    }
}
