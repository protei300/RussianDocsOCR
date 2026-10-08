package net.russiandocs.docproc

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.modules.Field
import net.russiandocs.docproc.pipeline.PairSides
import net.russiandocs.docproc.pipeline.ReadMargins
import net.russiandocs.docproc.pipeline.Results
import net.russiandocs.docproc.pipeline.SplitWords
import net.russiandocs.docproc.postprocess.Box
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The reading rules for the vehicle registration certificate that need no models — the cases of
 * `tests/test_sts_reading.py` (read margins), `tests/test_paired_duplicates.py` and the pairing cases of
 * `tests/test_document_detector.py`.
 */
class StsReadingTests {

    @BeforeTest
    fun loadNatives() {
        NativeLibraries.load()
    }

    /** A 300x200 canvas whose red channel holds the row number. */
    private fun canvas(): Image {
        val mat = Mat(200, 300, CvType.CV_8UC3)
        val row = ByteArray(300 * 3)
        for (y in 0 until 200) {
            for (x in 0 until 300) {
                row[x * 3] = y.toByte()
            }
            mat.put(y, 0, row)
        }
        return Image.wrap(mat)
    }

    private fun box(x1: Int, y1: Int, x2: Int, y2: Int, conf: Double, label: String): Box = Box().also {
        it.x1 = x1.toDouble(); it.y1 = y1.toDouble(); it.x2 = x2.toDouble(); it.y2 = y2.toDouble()
        it.conf = conf; it.label = label
    }

    private fun cutAll(img: Image, boxes: List<Box>): MutableList<Field> =
        boxes.map { b ->
            Field(b, net.russiandocs.docproc.imaging.Crop.clampedCrop(img, b.x1.toInt(), b.y1.toInt(), b.x2.toInt(), b.y2.toInt()))
        }.toMutableList()

    /** Row numbers (red channel of the first column) of a patch, top to bottom. */
    private fun rows(f: Field): List<Int> = (0 until f.patch.height).map { (f.patch.mat.get(it, 0)[0]).toInt() }

    @Test
    fun theReadCropGrowsAndTheBoxDoesNot() {
        canvas().use { img ->
            val b = box(20, 100, 280, 120, 0.9, "Special_marks")
            val fields = cutAll(img, listOf(b))
            ReadMargins.apply(fields, img, mapOf("Special_marks" to 0.25))
            assertEquals(30, fields[0].patch.height)            // 20 px + 5 above + 5 below
            assertEquals(95, rows(fields[0]).first())
            assertEquals(124, rows(fields[0]).last())
            assertEquals(100.0, fields[0].box.y1)               // the box is as the detector gave it
            assertEquals(120.0, fields[0].box.y2)
            fields.forEach { it.close() }
        }
    }

    @Test
    fun theMarginStopsHalfwayToTheNextLine() {
        canvas().use { img ->
            val upper = box(20, 80, 280, 96, 0.9, "Special_marks")
            val lower = box(20, 100, 280, 116, 0.9, "Special_marks")
            val fields = cutAll(img, listOf(upper, lower))
            ReadMargins.apply(fields, img, mapOf("Special_marks" to 0.5))
            val u = rows(fields[0])
            val l = rows(fields[1])
            assertTrue(u.max() < l.min(), "the two crops overlap")
            assertEquals(97, u.max())
            assertEquals(98, l.min())
            fields.forEach { it.close() }
        }
    }

    @Test
    fun aBoxBesideItDoesNotLimitTheMargin() {
        canvas().use { img ->
            val marks = box(20, 100, 140, 120, 0.9, "Special_marks")
            val beside = box(160, 80, 280, 98, 0.9, "Reg_number")         // no shared width
            val fields = cutAll(img, listOf(marks, beside))
            ReadMargins.apply(fields, img, mapOf("Special_marks" to 0.25))
            assertEquals(30, fields[0].patch.height)
            fields.forEach { it.close() }
        }
    }

    @Test
    fun otherFieldsAreUntouched() {
        canvas().use { img ->
            val fields = cutAll(img, listOf(box(20, 100, 280, 120, 0.9, "Vehicle_color")))
            ReadMargins.apply(fields, img, mapOf("Special_marks" to 0.25))
            assertEquals(20, fields[0].patch.height)
            fields.forEach { it.close() }
        }
    }

    // ---- a line labelled as both languages (test_paired_duplicates.py) ----

    @Test
    fun theSameLineLabeledBothWaysKeepsTheConfidentLabel() {
        val boxes = listOf(box(513, 384, 801, 423, 0.922, "Birth_place_ru"), box(513, 384, 801, 422, 0.619, "Birth_place_en"))
        assertEquals(setOf(1), SplitWords.pairedDuplicateIndices(boxes))
    }

    @Test
    fun theEnglishLabelWinsWhenItIsTheConfidentOne() {
        val boxes = listOf(box(100, 10, 300, 40, 0.55, "Birth_place_ru"), box(100, 10, 300, 40, 0.90, "Birth_place_en"))
        assertEquals(setOf(0), SplitWords.pairedDuplicateIndices(boxes))
    }

    @Test
    fun aRealRuEnPairSideBySideIsKept() {
        // «Г. ЧЕЛЯБИНСК / USSR»: two boxes on one line, not overlapping
        val boxes = listOf(box(407, 291, 571, 317, 0.95, "Birth_place_ru"), box(582, 289, 652, 316, 0.93, "Birth_place_en"))
        assertTrue(SplitWords.pairedDuplicateIndices(boxes).isEmpty())
    }

    @Test
    fun stackedLinesOverlappingLikeALicenceAreKept() {
        val boxes = listOf(box(100, 100, 400, 140, 0.9, "Last_name_ru"), box(100, 116, 400, 156, 0.9, "Last_name_en"))
        assertTrue(SplitWords.pairedDuplicateIndices(boxes).isEmpty())
    }

    @Test
    fun differentFieldsOnTheSameBoxAreNotThisRule() {
        val boxes = listOf(box(100, 10, 300, 40, 0.9, "Birth_place_ru"), box(100, 10, 300, 40, 0.6, "Last_name_en"))
        assertTrue(SplitWords.pairedDuplicateIndices(boxes).isEmpty())
    }

    // ---- the two sides of one document (test_document_detector.py) ----

    private fun side(docType: String, number: String?): Results = Results().also {
        it.docType = docType
        if (number != null) it.ocr = linkedMapOf("Licence_number" to number)
    }

    @Test
    fun frontAndBackWithOneNumberPairUp() {
        val docs = listOf(side("STS_2019", "99 87 786940"), side("DL_2020", "99 12 345678"),
            side("STSBACK_2019", "9987 786940"))
        PairSides.pair(docs)
        assertEquals(listOf(2, null, 0), docs.map { it.pairedWith })
    }

    @Test
    fun twoCertificatesOnOneSheetDoNotCrossPair() {
        val docs = listOf(side("STS_2019", "99 87 786940"), side("STSBACK_2019", "99 80 895276"),
            side("STS_2019", "99 80 895276"), side("STSBACK_2019", "99 87 786940"))
        PairSides.pair(docs)
        assertEquals(listOf(3, 2, 1, 0), docs.map { it.pairedWith })
    }

    @Test
    fun anAmbiguousOrUnreadNumberPairsNothing() {
        val docs = listOf(side("STS_2019", "99 87 786940"), side("STS_2019", "99 87 786940"),
            side("STSBACK_2019", "99 87 786940"), side("STSBACK_1996", null))
        PairSides.pair(docs)
        assertTrue(docs.all { it.pairedWith == null })
        assertNull(docs[3].pairedWith)
    }
}
