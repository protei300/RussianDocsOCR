package net.russiandocs.docproc

import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.pipeline.OcrTier
import net.russiandocs.docproc.pipeline.Recognizer
import net.russiandocs.docproc.pipeline.RunOptions
import net.russiandocs.docproc.imaging.Crop
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.imaging.Pt
import org.opencv.core.Core
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point as CvPoint
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgcodecs.Imgcodecs
import java.io.File
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * `Pipeline.process_frame` and the NONE short return, end to end with the real weights (12 sessions, so one
 * recognizer for the class). The frames are the conformance STS samples; a test skips itself, loudly, when the
 * sample or the weights are not in the tree.
 */
class FrameTests {

    @BeforeTest
    fun loadNatives() {
        NativeLibraries.load()
    }

    private fun sample(name: String): File? =
        File(ModelPaths.root(), "conformance/material/$name/synth_01_$name.jpg").takeIf { it.isFile }

    private fun withRecognizer(block: (Recognizer) -> Unit) {
        if (!File(ModelPaths.root(), "document_processing/models/DocDetect").isDirectory) {
            println("SKIPPED: no weights in the tree")
            return
        }
        Recognizer(Device.CPU, intraOpThreads = 0, ocrTier = OcrTier.ACCURATE).use(block)
    }

    @Test
    fun anUnrecognisedFrameReturnsAtNoneWithoutReadingFields() {
        withRecognizer { recognizer ->
            val blank = File.createTempFile("blank", ".jpg")
            try {
                val mat = Mat(900, 1400, CvType.CV_8UC3, Scalar(200.0, 200.0, 200.0))
                Imgcodecs.imwrite(blank.path, mat)
                mat.release()
                recognizer.run(blank.path, RunOptions()).use { results ->
                    assertEquals("NONE", results.docType)
                    assertTrue(results.ocr.isEmpty() && results.boxes.isEmpty(), "a NONE frame reads nothing")
                    assertTrue(results.quality.isEmpty(), "and does not even check its quality")
                    assertNotNull(results.canvas, "the canvas of a NONE frame is the one the pipeline stopped at")
                }
            } finally {
                blank.delete()
            }
        }
    }

    @Test
    fun oneDocumentInTheFrameIsOneResultAndPairsNothing() {
        val back = sample("STSBACK_1996") ?: return println("SKIPPED: no sample")
        withRecognizer { recognizer ->
            val results = recognizer.runFrame(back.path, RunOptions())
            try {
                assertEquals(1, results.size)
                assertEquals("STSBACK_1996", results[0].docType)
                assertEquals(null, results[0].pairedWith)
                assertEquals(mapOf("leasing" to true), results[0].leasing)
                // the same reading as process_img: the first result IS the largest document
                recognizer.run(back.path, RunOptions()).use { single ->
                    assertEquals(single.ocr, results[0].ocr)
                    assertEquals(single.documentIndex, results[0].documentIndex)
                }
            } finally {
                results.forEach { it.close() }
            }
        }
    }

    @Test
    fun everyDocumentOfASheetIsReadAndOnlyEqualNumbersPair() {
        // The synthetic front and back carry DIFFERENT numbers (the generator draws one per side), so on a sheet
        // they are two documents of one family that must NOT pair: pairing by the number is what stops two
        // certificates scanned on one sheet from crossing. The pairing rule itself is pinned by StsReadingTests.
        val front = sample("STS_1996") ?: return println("SKIPPED: no sample")
        val back = sample("STSBACK_1996") ?: return println("SKIPPED: no sample")
        withRecognizer { recognizer ->
            val sheet = File.createTempFile("sheet", ".jpg")
            try {
                val a = Imgcodecs.imread(front.path)
                val b = Imgcodecs.imread(back.path)
                val joined = Mat()
                Core.hconcat(listOf(a, b), joined)
                Imgcodecs.imwrite(sheet.path, joined)
                listOf(a, b, joined).forEach { it.release() }

                val results = recognizer.runFrame(sheet.path, RunOptions())
                try {
                    assertEquals(2, results.size, "both documents are read")
                    assertEquals(setOf("STS_1996", "STSBACK_1996"), results.map { it.docType }.toSet())
                    assertEquals(listOf<Int?>(0, 1), results.map { it.documentIndex }, "each result knows its own document")
                    val numbers = results.map { it.ocr["Licence_number"]?.filter { c -> c.isDigit() } }
                    assertTrue(numbers[0] != numbers[1], "the two sides carry different numbers: $numbers")
                    assertTrue(results.all { it.pairedWith == null })
                } finally {
                    results.forEach { it.close() }
                }
            } finally {
                sheet.delete()
            }
        }
    }

    /** Normalised correlation of two same-size patches, blurred so the 2.5x downscale does not alias. */
    private fun correlation(a: Mat, b: Mat): Double {
        val ga = Mat(); val gb = Mat()
        Imgproc.cvtColor(a, ga, Imgproc.COLOR_RGB2GRAY)
        Imgproc.cvtColor(b, gb, Imgproc.COLOR_RGB2GRAY)
        Imgproc.GaussianBlur(ga, ga, Size(0.0, 0.0), 2.0)
        Imgproc.GaussianBlur(gb, gb, Size(0.0, 0.0), 2.0)
        val r = Mat()
        Imgproc.matchTemplate(ga, gb, r, Imgproc.TM_CCOEFF_NORMED)
        val v = r.get(0, 0)[0]
        listOf(ga, gb, r).forEach { it.release() }
        return v
    }

    /** The photo cut through a quadrilateral onto a [w] x [h] patch; [shift] moves the quadrilateral (a control). */
    private fun cutFromPhoto(photo: Image, quad: List<Pt>, w: Int, h: Int, shift: Double = 0.0): Mat {
        // continuous coordinates -> OpenCV's pixel-centre coordinates, on both sides
        val src = MatOfPoint2f(*quad.map { CvPoint(it.x - 0.5 + shift, it.y - 0.5 + shift) }.toTypedArray())
        val dst = MatOfPoint2f(CvPoint(-0.5, -0.5), CvPoint(w - 0.5, -0.5), CvPoint(w - 0.5, h - 0.5), CvPoint(-0.5, h - 0.5))
        val m = Imgproc.getPerspectiveTransform(src, dst)
        val out = Mat()
        Imgproc.warpPerspective(photo.mat, out, m, Size(w.toDouble(), h.toDouble()), Imgproc.INTER_AREA)
        listOf(src, dst, m).forEach { it.release() }
        return out
    }

    @Test
    fun aFieldOfARealSampleIsCutFromThePhotoByItsQuadrilateral() {
        // The check a synthetic mark cannot make: the same pixels. The pipeline cut the field out of its canvas
        // (straightened by the printed blank, bent back, resized); cutting the same field out of the PHOTO through
        // the quadrilateral must give the same picture, and moving the quadrilateral by a few pixels must not.
        for (name in listOf("STS_1996", "STSBACK_1996")) {
            val path = sample(name) ?: return println("SKIPPED: no sample")
            withRecognizer { recognizer ->
                Io.loadRgb(path.path).use { photo ->
                    recognizer.run(path.path, RunOptions()).use { results ->
                        val canvas = results.canvas!!
                        var checked = 0
                        for (box in results.boxes) {
                            val quads = results.fieldQuads!![box.label] ?: continue
                            if (quads.size != 1 || results.boxes.count { it.label == box.label } != 1) continue
                            val w = (box.x2 - box.x1).toInt()
                            val h = (box.y2 - box.y1).toInt()
                            if (w < 80 || h < 16) continue
                            Crop.clampedCrop(canvas, box.x1.toInt(), box.y1.toInt(), box.x2.toInt(), box.y2.toInt()).use { own ->
                                val through = cutFromPhoto(photo, quads[0], own.width, own.height)
                                val moved = cutFromPhoto(photo, quads[0], own.width, own.height, shift = 25.0)
                                val good = correlation(own.mat, through)
                                val bad = correlation(own.mat, moved)
                                println("  $name ${box.label}: through the quadrilateral $good, moved by 25 px $bad")
                                assertTrue(good > 0.95, "$name ${box.label}: correlation $good")
                                assertTrue(good > bad + 0.15, "$name ${box.label}: $good is not better than the moved one $bad")
                                through.release(); moved.release()
                                checked++
                            }
                        }
                        assertTrue(checked >= 3, "$name: only $checked fields were checked")
                    }
                }
            }
        }
    }
}
