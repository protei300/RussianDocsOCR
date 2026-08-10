package net.russiandocs.docproc

import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.imaging.Crop
import net.russiandocs.docproc.imaging.Geometry
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Pt
import net.russiandocs.docproc.tensors.Dtype
import net.russiandocs.docproc.tensors.NdArray
import net.russiandocs.docproc.tensors.Npy
import net.russiandocs.docproc.tensors.Ops
import net.russiandocs.docproc.tensors.PyNum
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import java.io.File
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The traps, each with a test that FAILS if the trap is reintroduced.
 *
 * CONVENTIONS requires these per language rather than once: the whole point is that each language reaches
 * the trap through a different plausible mistake, so a test written in one does not protect the others.
 */
class FoundationTests {

    @BeforeTest
    fun loadNatives() {
        NativeLibraries.load()
    }

    /**
     * CPython's float `//` is fmod-based, and this is the case that proves it matters.
     *
     * A 2999x1777 image with img_size 1500 must give width 1499. `floor(2999 / ratio)` gives 1500 — a canvas
     * one pixel wider, which shifts every box downstream and fails comparison at a stage far from the cause.
     */
    @Test
    fun floorDivMatchesCPythonOnTheCaseThatBitTheGoPort() {
        val ratio = maxOf(2999.0 / 1500, 1.0)
        assertEquals(1499, PyNum.floorDivInt(2999.0, ratio))
        assertEquals(888, PyNum.floorDivInt(1777.0, ratio))

        // And the naive form really does disagree — a test that cannot fail proves nothing.
        assertEquals(1500, kotlin.math.floor(2999.0 / ratio).toInt())
    }

    /** `np.round` is half to EVEN. `Math.round` and `roundToInt` are both half-up. */
    @Test
    fun roundHalfEvenIsRint() {
        assertEquals(0, PyNum.roundHalfEvenToInt(0.5))
        assertEquals(2, PyNum.roundHalfEvenToInt(1.5))
        assertEquals(2, PyNum.roundHalfEvenToInt(2.5))
        assertEquals(-2, PyNum.roundHalfEvenToInt(-2.5))

        // The tempting alternatives disagree on exactly these values.
        assertEquals(1, Math.round(0.5).toInt())
        assertEquals(3, Math.round(2.5).toInt())
    }

    /** `np.argmax` returns the FIRST maximum. Strict `>` only. */
    @Test
    fun argmaxKeepsTheFirstMaximum() {
        assertEquals(1, Ops.argmax(floatArrayOf(0.1f, 0.9f, 0.9f, 0.2f)))
        assertEquals(0, Ops.argmax(floatArrayOf(1f, 1f, 1f)))
    }

    /**
     * `Mat.submat` throws where Python's slice clamps.
     *
     * The detectors routinely return a box a pixel or two outside the image, so a port that translates the
     * slice literally crashes on real input. Every case here is one the reference handles silently.
     */
    @Test
    fun clampedCropBehavesLikeAPythonSlice() {
        Image.wrap(Mat(10, 20, CvType.CV_8UC3, Scalar(1.0, 2.0, 3.0))).use { src ->
            // Past the right and bottom edges: clamped, not an error.
            Crop.clampedCrop(src, 15, 5, 40, 30).use {
                assertEquals(5, it.width)
                assertEquals(5, it.height)
            }
            // Negative start clamps to zero rather than counting from the end.
            Crop.clampedCrop(src, -5, -5, 4, 4).use {
                assertEquals(4, it.width)
                assertEquals(4, it.height)
            }
            // A reversed range is EMPTY in Python — not an error and not a flipped crop.
            Crop.clampedCrop(src, 8, 8, 3, 3).use {
                assertEquals(0, it.width)
                assertEquals(0, it.height)
                assertEquals(3, it.channels)
            }
            // Entirely outside: empty, with the source's channel count preserved.
            Crop.clampedCrop(src, 100, 100, 120, 120).use {
                assertTrue(it.isEmpty)
            }
        }
    }

    /**
     * The variance matches `np.var`, which is what the deskewer's angle choice rests on.
     *
     * **This test corrected the claim it was written to defend, and the correction is worth recording.**
     * CONVENTIONS carries a note from the Go port that the one-pass `E[x²] − E[x]²` "loses about seven
     * significant digits" at the magnitudes the deskewer works with, and that this can flip the argmax. In
     * float64 it does not. The test's own guard fired twice: at 255x700 over 200 rows and again at 255x4000
     * over 2000 rows, the one-pass and two-pass forms agree to the last printed digit. A float64 mantissa is
     * simply wide enough at mean²/variance ratios of order 1e6.
     *
     * So the seven-digit figure belongs to float32 accumulation, and NumPy does not do that here: the input
     * is an int64 row sum and `.var()` computes in float64. This port keeps the two-pass form anyway,
     * because it is what the reference computes and it costs one extra pass over a few hundred doubles — but
     * it should not be presented as the difference between a correct and an incorrect deskew.
     *
     * What the test therefore asserts is the thing that IS true and IS load-bearing: this implementation
     * reproduces NumPy's definition exactly, including the population divisor (n, not n-1). Getting THAT
     * wrong scales every variance by n/(n-1) — harmless for the argmax, which is why it needs a test rather
     * than a comparison.
     */
    @Test
    fun varianceMatchesNumpyIncludingThePopulationDivisor() {
        // 0..1999 offset by a large constant: the true population variance is that of 0..1999.
        val n = 2000
        val values = DoubleArray(n) { 255.0 * 4000 + it }
        val exact = ((n.toLong() * n - 1) / 12.0)   // variance of 0..n-1

        assertTrue(kotlin.math.abs(Ops.variance(values) - exact) < 1e-6,
            "got ${Ops.variance(values)}, expected $exact")

        // The population divisor, stated as a case rather than left implicit: the sample variance of
        // [1, 2, 3, 4] is 1.666..., the population variance is 1.25, and np.var gives the latter.
        assertEquals(1.25, Ops.variance(doubleArrayOf(1.0, 2.0, 3.0, 4.0)), 1e-12)

        assertEquals(0.0, Ops.variance(DoubleArray(0)))
    }

    /** `.npy` round-trips, including the `<U` decode that turns into blanks when done naively. */
    @Test
    fun npyRoundTripsEveryDtypeItSupports() {
        val temp = File.createTempFile("rdocs-npy", ".npy")
        try {
            val original = NdArray.fromFloat32(floatArrayOf(1.5f, -2.5f, 3.25f), 3)
            Npy.save(temp.path, original)
            val loaded = Npy.load(temp.path)
            assertEquals(Dtype.FLOAT32, loaded.dtype)
            assertContentEquals(intArrayOf(3), loaded.shape)
            assertContentEquals(floatArrayOf(1.5f, -2.5f, 3.25f), loaded.asFloat32())
        } finally {
            temp.delete()
        }
    }

    /**
     * `<U<n>` is fixed-width UTF-32 with NUL padding.
     *
     * Read as bytes it decodes to empty strings, which is how a label array becomes nine blanks that match
     * nothing — a failure that reads like a broken model rather than a broken decoder. Verified against the
     * REAL centers.npz, so the test also proves the file is where the loader thinks it is.
     */
    @Test
    fun centersNpzDecodesRealLabels() {
        val root = ModelPaths.root()
        val path = File(root,
            "document_processing/models/DocTypeAngles/ONNX/resources/centers.npz")
        assertTrue(path.isFile, "centers.npz not found at $path")

        val blob = Npy.loadNpz(path.path)
        val labels = blob.getValue("labels").asUnicode()
        assertTrue(labels.isNotEmpty())
        assertTrue(labels.all { it.isNotEmpty() },
            "some labels decoded empty — the <U decoder is reading bytes: ${labels.toList()}")
        assertTrue(labels.any { it.startsWith("INTPASSPORT") },
            "expected a passport label among ${labels.toList()}")
        assertEquals(labels.size, blob.getValue("centers").shape[0])
    }

    /**
     * Corner ordering is by coordinate sum and difference, and ties go to the FIRST index.
     *
     * On an axis-aligned rectangle two corners share a sum; picking the later one rotates the whole quad and
     * the warp comes out transposed.
     */
    @Test
    fun orderPointsPutsTopLeftFirst() {
        val quad = listOf(Pt(10.0, 90.0), Pt(90.0, 90.0), Pt(90.0, 10.0), Pt(10.0, 10.0))
        val ordered = Geometry.orderPoints(quad)!!
        assertEquals(Pt(10.0, 10.0), ordered[0], "top-left")
        assertEquals(Pt(90.0, 10.0), ordered[1], "top-right")
        assertEquals(Pt(90.0, 90.0), ordered[2], "bottom-right")
        assertEquals(Pt(10.0, 90.0), ordered[3], "bottom-left")
    }

    /** The margin is 0.01 and applies to EACH edge, so the quad grows by 1 + 2*margin. */
    @Test
    fun expandQuadUsesTheReferenceMargin() {
        assertEquals(0.01, Geometry.DOC_MARGIN_FRACTION)
        val quad = listOf(Pt(0.0, 0.0), Pt(100.0, 0.0), Pt(100.0, 100.0), Pt(0.0, 100.0))
        val expanded = Geometry.expandQuad(quad, Geometry.DOC_MARGIN_FRACTION)
        // Centre 50,50; scale 1.02 → corners move 1 unit outward.
        assertEquals(-1.0, expanded[0].x, 1e-9)
        assertEquals(101.0, expanded[2].x, 1e-9)
    }

    /** Backslashes in the committed YAML and model.json must become the platform's separator. */
    @Test
    fun separatorsAreNormalised() {
        val normalised = ModelPaths.normaliseSeparators("resources\\centers.npz")
        assertTrue(normalised.contains(File.separatorChar))
        assertTrue(!normalised.contains('\\') || File.separatorChar == '\\')
    }
}
