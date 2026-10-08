package net.russiandocs.docproc

import net.russiandocs.docproc.modules.DocBox
import net.russiandocs.docproc.modules.DocumentDetector
import net.russiandocs.docproc.pipeline.Recognizer
import net.russiandocs.docproc.postprocess.Box
import net.russiandocs.docproc.tensors.PyNum
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * What the document detector does AFTER the network: grouping boxes into documents, and cutting the crop.
 * Pure arithmetic over boxes, so pinned here without weights — `tests/test_document_detector.py`.
 */
class DocumentDetectorTests {

    private fun box(label: String, x1: Double, y1: Double, x2: Double, y2: Double, conf: Double = 0.9) =
        Box().also {
            it.x1 = x1; it.y1 = y1; it.x2 = x2; it.y2 = y2; it.conf = conf; it.label = label
        }

    @Test
    fun documentsComeOutLargestFirstAndPagesBelongToTheDocumentTheyLieIn() {
        val spread = box("document", 0.0, 0.0, 500.0, 700.0, 0.95)
        val card = box("document", 600.0, 0.0, 700.0, 60.0, 0.80)
        val lower = box("page", 5.0, 360.0, 495.0, 700.0)
        val upper = box("page", 5.0, 0.0, 495.0, 350.0)
        val stray = box("page", 900.0, 900.0, 950.0, 950.0)   // inside no document: belongs to none

        val docs = DocumentDetector.groupDocuments(listOf(card, lower, spread, upper, stray))

        assertEquals(2, docs.size)
        assertEquals(DocBox(0.0, 0.0, 500.0, 700.0, 0.95), docs[0].box)
        assertEquals(DocBox(600.0, 0.0, 700.0, 60.0, 0.80), docs[1].box)
        // pages by (y1, x1): the upper one first, whatever order the detector listed them in
        assertEquals(listOf(0.0, 360.0), docs[0].pages.map { it.y1 })
        assertEquals(0, docs[1].pages.size)
    }

    @Test
    fun aPageNeedsSeventyPercentInsideToBelong() {
        val doc = box("document", 0.0, 0.0, 100.0, 100.0)
        val mostlyIn = box("page", 40.0, 0.0, 140.0, 100.0)    // 60 % inside: not owned
        val enough = box("page", 20.0, 0.0, 120.0, 100.0)      // 80 % inside: owned
        val docs = DocumentDetector.groupDocuments(listOf(doc, mostlyIn, enough))
        assertEquals(1, docs[0].pages.size)
        assertEquals(20.0, docs[0].pages[0].x1)
    }

    @Test
    fun noDocumentMeansNoDocuments() {
        assertEquals(0, DocumentDetector.groupDocuments(emptyList()).size)
        assertEquals(0, DocumentDetector.groupDocuments(listOf(box("page", 0.0, 0.0, 10.0, 10.0))).size)
    }

    /**
     * The crop is the box plus 3 % of its LONGER side on every side, floored at the near edge and ceiled at
     * the far one, clamped to the photo. The expected numbers are what `Pipeline._document_crop` gives.
     */
    @Test
    fun theCropAddsAMarginOfTheLongerSideAndClampsToThePhoto() {
        // longer side 1759.1, margin 52.773: 6 - 52.773 < 0 -> 0 ; 1765.1 + 52.773 -> ceil 1818 ; 1077 + 52.8 -> 1130
        assertContentEquals(
            intArrayOf(0, 0, 1818, 1130),
            Recognizer.documentCrop(DocBox(6.0, 6.0, 1765.1, 1077.0, 0.9), 2000, 1200),
        )
        // clamped on the far side
        assertContentEquals(
            intArrayOf(0, 0, 1770, 1080),
            Recognizer.documentCrop(DocBox(6.0, 6.0, 1765.1, 1077.0, 0.9), 1770, 1080),
        )
        // an interior box: 100 x 50, margin 3 -> (97, 47, 203, 103)
        assertContentEquals(
            intArrayOf(97, 47, 203, 103),
            Recognizer.documentCrop(DocBox(100.0, 50.0, 200.0, 100.0, 0.9), 1000, 1000),
        )
    }

    /** `round(v, 1)` of the builtin rounds the EXACT binary value: 2.675 is 2.67499..., so 2.67 — not 2.68. */
    @Test
    fun theStagePayloadIsRoundedLikeThePythonBuiltin() {
        assertEquals(2.67, PyNum.roundDecimal(2.675, 2))
        assertEquals(1765.1, PyNum.roundDecimal(1765.1, 1))
        assertEquals(0.2, PyNum.roundDecimal(0.25, 1))     // exact tie: half to even
        assertEquals(0.3, PyNum.roundDecimal(0.35, 1))     // 0.35 is 0.34999...98 in binary
    }
}
