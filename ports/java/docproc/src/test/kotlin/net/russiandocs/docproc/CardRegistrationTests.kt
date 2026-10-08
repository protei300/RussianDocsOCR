package net.russiandocs.docproc

import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.modules.PageRegistrar
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point as CvPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.hypot
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * A vehicle registration certificate straightened by its printed blank — the cases of
 * `tests/test_card_registration.py`.
 *
 * A card in a plastic sleeve gets the sleeve's edge from the border detector, and the canvas comes out skewed. The
 * pipeline therefore registers the card against the cleaned blank of its type
 * (`page_registration/templates/sts*.json`) and takes the template geometry when the match is strong
 * (`Recognizer.registerCard`). No models here: the blank itself, put into a photo in perspective, is the card.
 */
class CardRegistrationTests {

    private val types = listOf("STS_1996", "STSBACK_1996", "STS_2019", "STSBACK_2019")

    @BeforeTest
    fun loadNatives() {
        NativeLibraries.load()
    }

    /** The card warped onto a plain background at [corners] (TL, TR, BR, BL); RGB. */
    private fun photoOf(cardGray: Mat, corners: Array<Pt>, width: Int = 1600, height: Int = 1900): Image {
        val w = cardGray.cols().toDouble()
        val h = cardGray.rows().toDouble()
        val src = MatOfPoint2f(CvPoint(0.0, 0.0), CvPoint(w, 0.0), CvPoint(w, h), CvPoint(0.0, h))
        val dst = MatOfPoint2f(*corners.map { CvPoint(it.x, it.y) }.toTypedArray())
        val m = Imgproc.getPerspectiveTransform(src, dst)
        val card = Mat()
        Imgproc.cvtColor(cardGray, card, Imgproc.COLOR_GRAY2RGB)
        val warped = Mat()
        Imgproc.warpPerspective(card, warped, m, Size(width.toDouble(), height.toDouble()),
            Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))
        val ones = Mat(cardGray.rows(), cardGray.cols(), CvType.CV_8UC1, Scalar(255.0))
        val inside = Mat()
        Imgproc.warpPerspective(ones, inside, m, Size(width.toDouble(), height.toDouble()))
        val bg = Mat(height, width, CvType.CV_8UC3, Scalar(90.0, 110.0, 130.0))
        warped.copyTo(bg, inside)
        listOf(src, dst, m, card, warped, ones, inside).forEach { it.release() }
        return Image.wrap(bg)
    }

    @Test
    fun everyStsTypeHasTemplatesAndAStaticMask() {
        for (type in types) {
            PageRegistrar(type).use { reg ->
                assertEquals(listOf("card"), reg.pageNames, type)
                for (tpl in reg.pages[0].second) {
                    val static = Core.mean(tpl.mask).`val`[0] / 255.0
                    assertTrue(static > 0.3 && static < 0.8, "$type ${tpl.name}: mask keeps $static of the card")
                    assertTrue(tpl.keypoints.rows() > 500, "$type: too few features on the printed blank")
                }
            }
        }
    }

    @Test
    fun aCardInPerspectiveIsFoundToAFewPixels() {
        val corners = arrayOf(Pt(260.0, 180.0), Pt(1320.0, 240.0), Pt(1380.0, 1700.0), Pt(210.0, 1640.0))
        for (type in types) {
            PageRegistrar(type).use { reg ->
                photoOf(reg.pages[0].second[0].gray, corners).use { photo ->
                    val r = reg.register(photo, null).single()
                    assertTrue(r.ok && r.inliers >= 40, "$type: not found (${r.inliers} inliers)")
                    val err = r.quad!!.indices.maxOf { hypot(r.quad!![it].x - corners[it].x, r.quad!![it].y - corners[it].y) }
                    assertTrue(err < 4.0, "$type: card corners off by $err px")
                }
            }
        }
    }

    @Test
    fun theFillPaintsPastThePhotoInsteadOfSmearingItsEdge() {
        PageRegistrar("STS_2019").use { reg ->
            val photo = Mat(200, 100, CvType.CV_8UC3, Scalar(255.0, 255.0, 255.0))      // the "card" half
            val identity = arrayOf(doubleArrayOf(1.0, 0.0, 0.0), doubleArrayOf(0.0, 1.0, 0.0), doubleArrayOf(0.0, 0.0, 1.0))
            Image.wrap(photo).use { img ->
                reg.warpMatrix(img, identity, 0.2).use { smeared ->
                    assertEquals(255.0, smeared.mat.get(10, 190)[0], "the default repeats the edge (the passport path)")
                }
                reg.warpMatrix(img, identity, 0.2, doubleArrayOf(10.0, 20.0, 30.0)).use { painted ->
                    val px = painted.mat.get(10, 190)
                    assertEquals(listOf(10.0, 20.0, 30.0), px.toList())
                }
                // a fractional colour is truncated to whole levels, as `int(v)` of the reference
                reg.warpMatrix(img, identity, 0.2, doubleArrayOf(10.5, 20.9, 30.0)).use { painted ->
                    assertEquals(listOf(10.0, 20.0, 30.0), painted.mat.get(10, 190).toList())
                }
            }
        }
    }

    @Test
    fun theWarpMatricesComeFromOneFormula() {
        PageRegistrar("STS_1996").use { reg ->
            // a quad that IS the canonical page maps to the identity up to the cushion offset
            val m = reg.margin.toDouble()
            val quad = arrayOf(Pt(m, m), Pt(m + reg.pageW, m), Pt(m + reg.pageW, m + reg.pageH), Pt(m, m + reg.pageH))
            val q = reg.quadMatrix(quad, 1.0)
            for (i in 0 until 3) for (j in 0 until 3) {
                assertEquals(if (i == j) 1.0 else 0.0, q[i][j], 1e-9, "[$i][$j]")
            }
        }
    }

    @Test
    fun theTemplatesCarryNoFullColourPrint() {
        // Shipped grey: the red blank number and the stamp were cleaned and colour is not needed.
        val dir = File(File(ModelPaths.root(), "document_processing"), "pipeline_modules/page_registration/templates")
        val jpgs = dir.listFiles { f -> f.name.startsWith("sts") && f.name.endsWith(".jpg") }!!
        assertTrue(jpgs.isNotEmpty())
        for (f in jpgs) {
            val img = Imgcodecs.imread(f.path, Imgcodecs.IMREAD_UNCHANGED)
            assertEquals(1, img.channels(), f.name)
            img.release()
        }
    }
}
