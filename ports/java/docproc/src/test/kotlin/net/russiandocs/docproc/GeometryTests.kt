package net.russiandocs.docproc

import net.russiandocs.docproc.geometry.Chain
import net.russiandocs.docproc.geometry.Homography
import net.russiandocs.docproc.geometry.Offset
import net.russiandocs.docproc.geometry.Pieces
import net.russiandocs.docproc.geometry.PointMap
import net.russiandocs.docproc.geometry.QuarterTurns
import net.russiandocs.docproc.geometry.Scale
import net.russiandocs.docproc.geometry.Unknown
import net.russiandocs.docproc.geometry.VerticalRemap
import net.russiandocs.docproc.imaging.Geometry
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.imaging.StackDirection
import net.russiandocs.docproc.modules.DocDeskewer
import net.russiandocs.docproc.modules.LineDewarp
import net.russiandocs.docproc.modules.LineRefine
import net.russiandocs.docproc.pipeline.FieldQuads
import net.russiandocs.docproc.pipeline.KeptField
import net.russiandocs.docproc.modules.Field
import net.russiandocs.docproc.postprocess.Box
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point as CvPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.tan
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Every stage that changes the image knows the way back to its input — the cases of `tests/test_geometry.py`.
 *
 * Checked with the library's own functions and OpenCV on drawings with a mark: the mark found on the output of a
 * stage must land where it was drawn after the map is applied. A synthetic mark is the only ground truth that does
 * not come from the maps themselves.
 *
 * COORDINATES ARE CONTINUOUS: the pixel with index (i, j) is the unit square centred at (i + 0.5, j + 0.5). Tests
 * state the expected point with that half added, which is also what keeps them honest about the half-pixel shift
 * OpenCV's warps need.
 */
class GeometryTests {

    @BeforeTest
    fun loadNatives() {
        NativeLibraries.load()
    }

    private val background = doubleArrayOf(40.0, 70.0, 110.0)
    private val paper = doubleArrayOf(235.0, 235.0, 235.0)
    private val mark = doubleArrayOf(230.0, 20.0, 20.0)

    private fun canvas(width: Int, height: Int, colour: DoubleArray = background): Image =
        Image.wrap(Mat(height, width, CvType.CV_8UC3, Scalar(colour[0], colour[1], colour[2])))

    /** Centres of the red marks, one per connected blob, in continuous coordinates. */
    private fun marks(image: Image): List<Pt> {
        val rgb = image.mat
        val mask = Mat()
        val lo = Scalar(mark[0] - 70, mark[1] - 70, mark[2] - 70)
        val hi = Scalar(mark[0] + 70, mark[1] + 70, mark[2] + 70)
        Core.inRange(rgb, lo, hi, mask)
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val count = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids)
        val found = ArrayList<Pt>()
        for (i in 1 until count) {
            if (stats.get(i, Imgproc.CC_STAT_AREA)[0] >= 5.0) {
                found += Pt(centroids.get(i, 0)[0] + 0.5, centroids.get(i, 1)[0] + 0.5)
            }
        }
        listOf(mask, labels, stats, centroids).forEach { it.release() }
        assertTrue(found.isNotEmpty(), "no mark on the image")
        return found
    }

    private fun circle(image: Image, x: Int, y: Int, r: Int, colour: DoubleArray = mark) {
        Imgproc.circle(image.mat, CvPoint(x.toDouble(), y.toDouble()), r, Scalar(colour[0], colour[1], colour[2]), -1)
    }

    private fun assertLands(found: List<Pt>, expected: List<Pt>, atol: Double) {
        for (e in expected) {
            val nearest = found.minByOrNull { hypot(it.x - e.x, it.y - e.y) }!!
            assertTrue(abs(nearest.x - e.x) <= atol && abs(nearest.y - e.y) <= atol,
                "expected ($e) but the nearest mark landed at ($nearest)")
        }
    }

    private fun one(map: PointMap, p: Pt): Pt = map.toInput(listOf(p))!![0]

    @Test
    fun quarterTurnsPutThePixelBack() {
        // Against Core.rotate itself: a turn count of its own is the easiest thing to get wrong by one.
        for (turns in 0..3) {
            var image = Image.wrap(Mat.zeros(4, 7, CvType.CV_8UC1))
            image.mat.put(1, 5, 255.0)
            repeat(turns) {
                val next = Io.rot90(image, 1)
                image.close()
                image = next
            }
            val points = Mat()
            Core.findNonZero(image.mat, points)
            val at = points.get(0, 0)
            val back = one(QuarterTurns(7, 4, turns), Pt(at[0] + 0.5, at[1] + 0.5))
            assertEquals(5.5, back.x, 1e-9, "turns=$turns")
            assertEquals(1.5, back.y, 1e-9, "turns=$turns")
            points.release()
            image.close()
        }
    }

    @Test
    fun chainUndoesTheLastStageFirst() {
        val chain = Chain(listOf(Scale(0.5, 0.25), Offset(10.0, 20.0)))
        val back = one(chain, Pt(15.0, 45.0))
        assertEquals(10.0, back.x, 1e-12)
        assertEquals(100.0, back.y, 1e-12)
        // The same stages in the other order are a DIFFERENT map: a chain built "forward" sends a box off the photo.
        val forward = one(Chain(listOf(Offset(10.0, 20.0), Scale(0.5, 0.25))), Pt(15.0, 45.0))
        assertTrue(abs(forward.x - back.x) > 1.0 || abs(forward.y - back.y) > 1.0)
        // A stage that passed its image on unchanged adds nothing to the chain.
        assertTrue(chain.then(null) === chain)
    }

    @Test
    fun aStageWithNoWayBackMakesTheWholeChainAnswerNull() {
        // A map that cannot be inverted must not be silently dropped: the stages around it are still correct, so
        // the chain would keep answering — with a point that quietly belongs to another place on the photo.
        val chain = Chain(listOf(Scale(0.5, 0.5), Unknown, Offset(3.0, 4.0)))
        assertNull(chain.toInput(listOf(Pt(10.0, 10.0))))
        // Not identity: the stage did change the image, it is the way back that is gone.
        assertNull(chain.then(Unknown).toInput(listOf(Pt(10.0, 10.0))))
        assertNull(Pieces(listOf(Pieces.Piece(0.0, 0.0, 10.0, 10.0, Unknown))).toInput(listOf(Pt(1.0, 1.0))))
        // And a chain without it still answers.
        assertEquals(20.0, one(Chain(listOf(Scale(0.5, 0.5))), Pt(10.0, 10.0)).x, 1e-12)
    }

    private fun page(image: Image, quad: List<Pair<Int, Int>>): List<Pt> {
        val poly = org.opencv.core.MatOfPoint(*quad.map { CvPoint(it.first.toDouble(), it.second.toDouble()) }.toTypedArray())
        Imgproc.fillPoly(image.mat, listOf(poly), Scalar(paper[0], paper[1], paper[2]))
        poly.release()
        return quad.map { Pt(it.first.toDouble(), it.second.toDouble()) }
    }

    @Test
    fun aStraightenedPagePutsTheMarkBackOnThePhoto() {
        canvas(900, 700).use { image ->
            val contour = page(image, listOf(120 to 90, 760 to 140, 700 to 610, 90 to 560))
            circle(image, 400, 300, 5)

            val result = Geometry.fixPerspective(image, listOf(contour), StackDirection.AUTO, Geometry.DOC_MARGIN_FRACTION)

            result.canvas!!.use { warped ->
                assertLands(marks(warped).map { one(result.geometry, it) }, listOf(Pt(400.5, 300.5)), 1.0)
            }
        }
    }

    @Test
    fun aStitchedSpreadPutsTheMarkOfEachPageBack() {
        // Both pages, because a stitched canvas picks the piece by the centroid of the shape. Two pages of different
        // size are resized to a common side before the stitch, so a box mapped through the wrong piece lands
        // plausibly — on the other page.
        val cases = listOf(
            Triple(1024 to 760,
                listOf(listOf(40 to 60, 430 to 80, 420 to 640, 50 to 620), listOf(480 to 90, 960 to 70, 980 to 700, 470 to 660)),
                listOf(200 to 300, 700 to 400)),
            Triple(800 to 1040,
                listOf(listOf(60 to 40, 700 to 60, 690 to 420, 70 to 400), listOf(80 to 470, 720 to 460, 740 to 1000, 60 to 980)),
                listOf(300 to 200, 400 to 700)),
        )
        for ((size, quads, points) in cases) {
            canvas(size.first, size.second).use { image ->
                val contours = quads.map { page(image, it) }
                for ((x, y) in points) circle(image, x, y, 5)

                val result = Geometry.fixPerspective(image, contours, StackDirection.AUTO, Geometry.DOC_MARGIN_FRACTION)

                assertTrue(result.geometry is Pieces)
                result.canvas!!.use { warped ->
                    val found = marks(warped).map { one(result.geometry, it) }
                    assertEquals(2, found.size)
                    assertLands(found, points.map { Pt(it.first + 0.5, it.second + 0.5) }, 1.5)
                }
                result.pages.forEach { it.close() }
            }
        }
    }

    private fun lines(image: Image, top: Int, bottom: Int, degrees: Double) {
        var row = top
        while (row < bottom) {
            val rise = Math.rint(640 * tan(Math.toRadians(degrees))).toInt()
            Imgproc.line(image.mat, CvPoint(80.0, row.toDouble()), CvPoint(720.0, (row + rise).toDouble()),
                Scalar(20.0, 20.0, 20.0), 5)
            row += 28
        }
    }

    @Test
    fun deskewPutsTheMarkBack() {
        canvas(800, 600, doubleArrayOf(245.0, 245.0, 245.0)).use { image ->
            lines(image, 40, 540, 5.0)
            circle(image, 433, 287, 5)
            val deskewer = DocDeskewer.forPipeline()

            val (straight, _, geometry) = deskewer.deskewWithGeometry(image)

            assertTrue(geometry != null, "5 degrees is above min_angle, so the image is turned")
            assertLands(marks(straight).map { one(geometry!!, it) }, listOf(Pt(433.5, 287.5)), 1.0)
            straight.close()

            // Nothing to turn: the image is passed on as it is, and the stage adds no map.
            canvas(800, 600, doubleArrayOf(245.0, 245.0, 245.0)).use { flat ->
                val (same, _, none) = deskewer.deskewWithGeometry(flat)
                assertNull(none)
                same.close()
            }
        }
    }

    /** A smooth vertical displacement map, the shape line_dewarp fits: a bow across the page. */
    private fun bend(width: Int, height: Int, amplitude: Float): FloatArray {
        val v = FloatArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val xs = (x.toFloat() - width / 2f) / (width / 2f)
                val ys = (y.toFloat() - height / 2f) / (width / 2f)
                v[y * width + x] = amplitude * (1f - xs * xs) * (0.5f + ys)
            }
        }
        return v
    }

    private fun matOf(v: FloatArray, height: Int, width: Int): Mat = Mat(height, width, CvType.CV_32F).also { it.put(0, 0, v) }

    @Test
    fun anUnbentPagePutsTheMarkBackThroughTheBendMap() {
        // Against remap itself: the bend map is the way back, read at the right pixel. The marks are off-centre on
        // purpose: the map is not uniform, and a mark read at a wrong pixel of it lands a pixel or two away.
        for (amplitude in listOf(4f, 12f)) {
            canvas(600, 400, doubleArrayOf(235.0, 235.0, 235.0)).use { image ->
                circle(image, 170, 290, 4)
                circle(image, 450, 110, 4)
                val v = bend(600, 400, amplitude)
                val vMat = matOf(v, 400, 600)

                LineDewarp.applyDewarp(image, vMat).use { unbent ->
                    val map = VerticalRemap(v, 400, 600)
                    assertLands(marks(unbent).map { one(map, it) }, listOf(Pt(170.5, 290.5), Pt(450.5, 110.5)), 1.0)
                }
                vMat.release()
            }
        }
    }

    @Test
    fun aStraightenedAndUnbentPageChainsBothMaps() {
        // The registration's straightening: a homography, then the bend map on its result.
        canvas(800, 500, doubleArrayOf(235.0, 235.0, 235.0)).use { image ->
            circle(image, 300, 260, 4)
            val src = MatOfPoint2f(CvPoint(0.0, 0.0), CvPoint(800.0, 0.0), CvPoint(800.0, 500.0), CvPoint(0.0, 500.0))
            val dst = MatOfPoint2f(CvPoint(6.0, 3.0), CvPoint(792.0, -4.0), CvPoint(805.0, 503.0), CvPoint(-3.0, 496.0))
            val hMat = Imgproc.getPerspectiveTransform(src, dst)
            val hm = Array(3) { r -> DoubleArray(3).also { hMat.get(r, 0, it) } }
            val v = bend(800, 500, 8f)
            val vMat = matOf(v, 500, 800)

            LineRefine.applyRefinement(image, hm).use { straightened ->
                LineDewarp.applyDewarp(straightened, vMat).use { page ->
                    val chain = Chain(listOf(Homography(hm), VerticalRemap(v, 500, 800)))
                    assertLands(marks(page).map { one(chain, it) }, listOf(Pt(300.5, 260.5)), 1.0)
                }
            }
            listOf(src, dst, hMat, vMat).forEach { it.release() }
        }
    }

    // ---- the quadrilaterals of the fields ------------------------------------------------------------------

    private fun box(x1: Int, y1: Int, x2: Int, y2: Int, label: String): Box = Box().also {
        it.x1 = x1.toDouble(); it.y1 = y1.toDouble(); it.x2 = x2.toDouble(); it.y2 = y2.toDouble()
        it.conf = 0.9; it.label = label
    }

    @Test
    fun aFieldAndItsWordLandWhereTheyWereCutFrom() {
        // A canvas that is the photo shrunk by half and shifted: the quadrilateral of a field cut at (100, 50)
        // must come back on the photo twice as large and shifted.
        val map = Chain(listOf(Offset(-10.0, -20.0), Scale(0.5, 0.5)))
        Image.wrap(Mat.zeros(20, 40, CvType.CV_8UC3)).use { patch ->
            val field = Field(box(100, 50, 140, 70, "Last_name_ru"), patch, Offset(-100.0, -50.0))
            val kept = listOf(KeptField(field, listOf(box(5, 2, 15, 12, "word"))))

            val quads = FieldQuads.compute(kept, false) { map.toInput(it) }

            // the stages ran: shift by (-10, -20), then shrink by half — so the photo point is twice the canvas one
            // plus (10, 20)
            val f = quads.fields!!["Last_name_ru"]!![0]
            assertEquals(Pt(100.0 * 2 + 10, 50.0 * 2 + 20), f[0])              // top-left, then clockwise
            assertEquals(Pt(140.0 * 2 + 10, 50.0 * 2 + 20), f[1])
            assertEquals(Pt(140.0 * 2 + 10, 70.0 * 2 + 20), f[2])
            assertEquals(Pt(100.0 * 2 + 10, 70.0 * 2 + 20), f[3])
            val w = quads.words!!["Last_name_ru"]!![0]
            assertEquals(Pt(105.0 * 2 + 10, 52.0 * 2 + 20), w[0])
            assertEquals(Pt(115.0 * 2 + 10, 62.0 * 2 + 20), w[2])
        }
    }

    @Test
    fun aFieldThatWasNotSplitHasItsOwnQuadrilateralAsTheWord() {
        Image.wrap(Mat.zeros(20, 40, CvType.CV_8UC3)).use { patch ->
            val field = Field(box(10, 10, 50, 30, "Sex_ru"), patch, Offset(-10.0, -10.0))
            val quads = FieldQuads.compute(listOf(KeptField(field, null)), false) { it }
            assertEquals(quads.fields!!["Sex_ru"], quads.words!!["Sex_ru"])
        }
    }

    @Test
    fun theSeriesNumberWordLandsOnItsOwnEndOfTheField() {
        // The series/number patch is turned once before the words are found on it, so its word must land on its own
        // end of the field — not on the other end. Drawn, turned with the library's own rotation, and found again.
        val cut = Image.wrap(Mat.zeros(20, 40, CvType.CV_8UC3))            // as cut from the canvas: 40 wide, 20 high
        Imgproc.rectangle(cut.mat, CvPoint(30.0, 4.0), CvPoint(33.0, 8.0), Scalar(255.0, 255.0, 255.0), -1)
        val turned = Io.rot90(cut, 1)                                       // what the words were found on: 20 x 40
        cut.close()
        turned.use { patch ->
            val gray = Mat()
            Imgproc.cvtColor(patch.mat, gray, Imgproc.COLOR_RGB2GRAY)
            val pts = Mat()
            Core.findNonZero(gray, pts)
            val bounds = Imgproc.boundingRect(pts)
            gray.release(); pts.release()
            val word = box(bounds.x, bounds.y, bounds.x + bounds.width, bounds.y + bounds.height, "word")

            val field = Field(box(100, 50, 140, 70, "Licence_number"), patch, Offset(-100.0, -50.0))
            val quads = FieldQuads.compute(listOf(KeptField(field, listOf(word))), true) { it }

            val w = quads.words!!["Licence_number"]!![0]
            val cx = w.sumOf { it.x } / 4
            val cy = w.sumOf { it.y } / 4
            // the mark covers pixels x 30..33, y 4..8 of the cut patch, i.e. [30, 34] x [4, 9], which sits at (100, 50)
            assertEquals(100.0 + 32.0, cx, 1e-6)
            assertEquals(50.0 + 6.5, cy, 1e-6)
            val f = quads.fields!!["Licence_number"]!![0]
            assertEquals(Pt(100.0, 50.0), f[0])
            assertEquals(Pt(140.0, 70.0), f[2], "the field quadrilateral is the patch as cut, not as turned")
        }
    }

    @Test
    fun noWayBackLeavesNoQuadrilaterals() {
        Image.wrap(Mat.zeros(20, 40, CvType.CV_8UC3)).use { patch ->
            val field = Field(box(10, 10, 50, 30, "Sex_ru"), patch, Offset(-10.0, -10.0))
            val quads = FieldQuads.compute(listOf(KeptField(field, null)), false) { null }
            assertNull(quads.fields)
            assertNull(quads.words)
            assertEquals("{\"fields\":null,\"words\":null}", quads.toJson().toString())
        }
    }
}
