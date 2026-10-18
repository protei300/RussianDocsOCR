package net.russiandocs.docproc

import net.russiandocs.docproc.geometry.corners
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.modules.AddressLinesDetector
import net.russiandocs.docproc.modules.AddressTextKindClassifier
import net.russiandocs.docproc.pipeline.AddressPath
import net.russiandocs.docproc.pipeline.FieldQuads
import net.russiandocs.docproc.pipeline.RegistrationStamp
import net.russiandocs.docproc.pipeline.Stamps
import net.russiandocs.docproc.postprocess.Context
import net.russiandocs.docproc.postprocess.ObbBox
import net.russiandocs.docproc.postprocess.YoloObbDetector
import net.russiandocs.docproc.tensors.NdArray
import net.russiandocs.docproc.viewmodel.Builder
import net.russiandocs.docproc.viewmodel.Input
import net.russiandocs.docproc.viewmodel.RawAddressLine
import net.russiandocs.docproc.viewmodel.RawObb
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import org.opencv.core.CvType
import org.opencv.core.Mat
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The registration page (INTPASSPORTADDR): the oriented-box decoder, the crop and its way back, the stamps and
 * the address view model. Pure logic and OpenCV — no weights needed; the end-to-end check is conformance case
 * `INTPASSPORTADDR_ALL-synth_01_INTPASSPORTADDR_ALL`.
 */
class AddressPathTests {

    @BeforeTest
    fun loadNatives() {
        NativeLibraries.load()
    }

    private fun quad(x0: Double, y0: Double, x1: Double, y1: Double): List<Pt> = corners(x0, y0, x1, y1)

    private fun stamp(kind: String) = RegistrationStamp(kind, 0.9, null)

    @Test
    fun aNumberTokenIsDigitDominated() {
        assertTrue(AddressPath.isNumberToken("16"))
        assertTrue(AddressPath.isNumberToken("12A"))      // two digits, one letter
        assertTrue(AddressPath.isNumberToken("1А"))       // equal counts count as a number
        assertFalse(AddressPath.isNumberToken("УЛ"))
        assertFalse(AddressPath.isNumberToken("К1А"))     // one digit, two letters
        assertFalse(AddressPath.isNumberToken(""))
        assertFalse(AddressPath.isNumberToken(null))
        assertFalse(AddressPath.isNumberToken("..."))
    }

    @Test
    fun aDeregisteredStampCancelsTheNearestEarlierRegistration() {
        // freshest first: «снят», registered, «снят», registered, registered
        val stamps = listOf("deregistered", "registered", "deregistered", "registered", "registered").map(::stamp)
        Stamps.tieCancellations(stamps)
        assertEquals(1, stamps[0].cancels)
        assertEquals(3, stamps[2].cancels)
        assertNull(stamps[1].cancels)
        // the registrations 1 and 3 are cancelled; 4 is the freshest one nothing cancels
        assertEquals(listOf(false, false, false, false, true), stamps.map { it.current })
    }

    @Test
    fun aDeregisteredStampThatOpensThePageCancelsNothingOnIt() {
        val stamps = listOf("deregistered", "unclear").map(::stamp)
        Stamps.tieCancellations(stamps)
        assertNull(stamps[0].cancels)
        assertTrue(stamps.none { it.current })
    }

    @Test
    fun stampsAreOrderedFreshestFirstLeftPageBeforeRight() {
        // a spread: 2000 wide, 700 high, split down the middle at x = 1000
        val leftUpper = quad(100.0, 50.0, 300.0, 150.0)
        val leftLower = quad(100.0, 400.0, 300.0, 500.0)
        val rightUpper = quad(1200.0, 50.0, 1400.0, 150.0)
        val order = Stamps.orderStamps(listOf(leftUpper, rightUpper, leftLower), 2000, 700)
        // filled in order: leftUpper (0), leftLower (2), rightUpper (1); the freshest is the last one filled
        assertEquals(listOf(1, 2, 0), order)
    }

    @Test
    fun aSinglePageIsOrderedTopToBottom() {
        val upper = quad(100.0, 50.0, 400.0, 150.0)
        val lower = quad(100.0, 400.0, 400.0, 500.0)
        // 700 wide, 800 high: not a spread, so the x of the stamps does not make a "right page"
        assertEquals(listOf(1, 0), Stamps.orderStamps(listOf(upper, lower), 700, 800))
    }

    @Test
    fun aLineBelongsToTheStampItsCentreIsIn() {
        val a = quad(0.0, 0.0, 100.0, 100.0)
        val b = quad(60.0, 0.0, 200.0, 100.0)
        assertEquals(0, Stamps.containingStamp(Pt(20.0, 50.0), listOf(a, b)))
        assertEquals(1, Stamps.containingStamp(Pt(150.0, 50.0), listOf(a, b)))
        // in both: the one whose centre is nearer (a: 50, b: 130)
        assertEquals(0, Stamps.containingStamp(Pt(70.0, 50.0), listOf(a, b)))
        assertEquals(1, Stamps.containingStamp(Pt(120.0, 50.0), listOf(a, b)))
        assertNull(Stamps.containingStamp(Pt(500.0, 500.0), listOf(a, b)))
    }

    @Test
    fun theCropOfAnUnrotatedBoxIsTheWindowAtItsCentreAndTheMapGoesBack() {
        val img = Mat(80, 100, CvType.CV_8UC3)
        val row = ByteArray(100 * 3)
        for (y in 0 until 80) {
            for (x in 0 until 100) {
                row[x * 3] = (x * 2).toByte()
                row[x * 3 + 1] = y.toByte()
                row[x * 3 + 2] = 7
            }
            img.put(y, 0, row)
        }
        Image.wrap(img).use { image ->
            val box = ObbBox(50.0, 40.0, 21.0, 11.0, 0.0, 0.9, 0, "Living_region_ru")
            AddressLinesDetector.cropRotated(image, box).use { patch ->
                assertEquals(21, patch.width)
                assertEquals(11, patch.height)
                // patch pixel (0, 0) is image pixel (cx - 10, cy - 5)
                val px = patch.mat.get(0, 0)
                assertEquals(40.0 * 2, px[0])
                assertEquals(35.0, px[1])
            }
            val back = AddressLinesDetector.cropGeometry(box).toInput(corners(0.0, 0.0, 21.0, 11.0))!!
            assertEquals(40.0, back[0].x, 1e-9)
            assertEquals(35.0, back[0].y, 1e-9)
            assertEquals(61.0, back[2].x, 1e-9)
            assertEquals(46.0, back[2].y, 1e-9)
        }
    }

    @Test
    fun theCentreOfARotatedCropStaysAtTheCentreOfTheBox() {
        val box = ObbBox(300.5, 120.25, 90.0, 20.0, 0.3, 0.9, 0, "Living_region_ru")
        val centre = AddressLinesDetector.cropGeometry(box).toInput(listOf(Pt(45.0, 10.0)))!![0]
        // pixel centres sit at +0.5 in continuous coordinates, and the rotation is about the centre
        assertEquals(box.cx + 0.5, centre.x, 1e-9)
        assertEquals(box.cy + 0.5, centre.y, 1e-9)
        // and the whole patch is tilted by the angle: the top edge runs along (cos, sin)
        val edge = AddressLinesDetector.cropGeometry(box).toInput(listOf(Pt(0.0, 0.0), Pt(90.0, 0.0)))!!
        assertEquals(Math.cos(0.3) * 90.0, edge[1].x - edge[0].x, 1e-9)
        assertEquals(Math.sin(0.3) * 90.0, edge[1].y - edge[0].y, 1e-9)
    }

    @Test
    fun theTextKindClassifierSeesALineCentredOnAGreySquare() {
        val line = Io.newFilled(4, 10, 0.0, 0.0, 0.0)
        line.use {
            AddressTextKindClassifier.letterboxSquare(line).use { square ->
                assertEquals(10, square.width)
                assertEquals(10, square.height)
                // (10 - 4) // 2 = 3 rows of grey above, 3 below
                assertEquals(114.0, square.mat.get(2, 5)[0])
                assertEquals(0.0, square.mat.get(3, 5)[0])
                assertEquals(0.0, square.mat.get(6, 5)[0])
                assertEquals(114.0, square.mat.get(7, 5)[0])
            }
        }
    }

    /** [features, anchors] float32 head with `nc = 1`: cx, cy, w, h, score, angle. */
    private fun head(rows: List<FloatArray>): NdArray {
        // padded with empty anchors: a head with fewer anchors than features would read as the transposed layout
        val anchors = rows + List(16) { floatArrayOf(0f, 0f, 1f, 1f, 0f, 0f) }
        val n = anchors.size
        val out = FloatArray(6 * n)
        for ((a, row) in anchors.withIndex()) {
            for (f in 0 until 6) {
                out[f * n + a] = row[f]
            }
        }
        return NdArray.fromFloat32(out, 1, 6, n)
    }

    @Test
    fun theDecoderKeepsTheBestOfTwoOverlappingBoxesAndOrdersTheRest() {
        val decoder = YoloObbDetector(listOf("Living_region_ru"), 0.45, 0.25)
        val output = head(
            listOf(
                floatArrayOf(300f, 400f, 200f, 40f, 0.60f, 0.0f),   // lower, kept
                floatArrayOf(302f, 401f, 200f, 40f, 0.90f, 0.0f),   // its better twin: it wins
                floatArrayOf(300f, 100f, 200f, 40f, 0.80f, 0.1f),   // upper
                floatArrayOf(300f, 250f, 200f, 40f, 0.10f, 0.0f),   // under the threshold
            ),
        )
        // letterboxed at 0.5 with 10 px of padding on x: undo both
        val boxes = decoder.decode(output, Context(ratio = 0.5, padLetter = doubleArrayOf(10.0, 0.0), resize = true))
        assertEquals(2, boxes.size)
        // reading order: the upper box first
        assertEquals((100.0 - 0.0) / 0.5, boxes[0].cy, 1e-4)
        assertEquals((300.0 - 10.0) / 0.5, boxes[0].cx, 1e-4)
        assertEquals(400.0, boxes[0].w, 1e-4)
        assertEquals(0.1, boxes[0].angle, 1e-6)
        assertEquals("Living_region_ru", boxes[0].label)
        // the better of the two twins survived
        assertEquals(0.90, boxes[1].conf, 1e-6)
        assertEquals((302.0 - 10.0) / 0.5, boxes[1].cx, 1e-4)
    }

    @Test
    fun aDecoderGivenTheTransposedHeadReadsTheSameBoxes() {
        val decoder = YoloObbDetector(listOf("a", "b"), 0.45, 0.25)
        // [anchors, features] with 8 features x 3 anchors... anchors > features is the transposed layout
        val rows = List(20) { i -> floatArrayOf(50f + i * 100, 50f, 40f, 20f, if (i == 3) 0.1f else 0.0f, if (i == 5) 0.9f else 0.0f, 0f) }
        val data = FloatArray(20 * 7)
        for ((a, r) in rows.withIndex()) r.copyInto(data, a * 7)
        val boxes = decoder.decode(NdArray.fromFloat32(data, 1, 20, 7), Context(ratio = 1.0, resize = true))
        assertEquals(1, boxes.size)
        assertEquals("b", boxes[0].label)
        assertEquals(1, boxes[0].cls)
        assertEquals(550.0, boxes[0].cx, 1e-4)
    }

    @Test
    fun theQuadsStageCarriesTheAddressLinesOnlyOnARegistrationPage() {
        val plain = FieldQuads(emptyMap(), emptyMap()).toJson() as JsonObject
        assertEquals(setOf("fields", "words"), plain.keys)
        val onPage = FieldQuads(emptyMap(), emptyMap()).withAddressLines(listOf(quad(0.0, 0.0, 1.0, 1.0))).toJson() as JsonObject
        assertEquals(setOf("fields", "words", "address_lines"), onPage.keys)
        assertEquals(1, (onPage["address_lines"] as JsonArray).size)
        // unknown way back: the key is there, with null
        val unknown = FieldQuads(null, null).withAddressLines(null).toJson() as JsonObject
        assertTrue(unknown.containsKey("address_lines"))
    }

    @Test
    fun theViewModelAddressDropsTheGeometryWhenTheListsDoNotLineUp() {
        val box = RawObb(10.123456789, 20.0, 30.0, 40.0, 0.0156034, 0.8581, "Living_region_ru")
        val lines = listOf(RawAddressLine("printed", "Г. ПСКОВ", 0.0), RawAddressLine("handwritten", null, 0.99997908))

        val aligned = Builder.build(Input(docType = "INTPASSPORTADDR_ALL", addressBoxes = listOf(box, box), addressLines = lines), false)
        val address = assertNotNull(aligned.address)
        assertTrue(address.aligned)
        assertEquals("o1", address.lines[1].id)
        assertEquals(10.1235, address.lines[0].obbox!!.cx)
        assertEquals(0.015603, address.lines[0].obbox!!.angleRad)
        assertEquals(1.0, address.lines[1].pHandwritten)

        // one detection had an empty patch: the text stays, the boxes go
        val skewed = Builder.build(Input(docType = "INTPASSPORTADDR_ALL", addressBoxes = listOf(box, box, box), addressLines = lines), false)
        assertFalse(assertNotNull(skewed.address).aligned)
        assertNull(skewed.address!!.lines[0].obbox)
        assertEquals("Г. ПСКОВ", skewed.address!!.lines[0].text)

        // not a registration page, or nothing found: no address at all
        assertNull(Builder.build(Input(docType = "DL_2011"), false).address)
        assertNull(Builder.build(Input(addressBoxes = emptyList(), addressLines = emptyList()), false).address)
    }
}
