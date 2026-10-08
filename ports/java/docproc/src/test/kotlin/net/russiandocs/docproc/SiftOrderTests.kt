package net.russiandocs.docproc

import net.russiandocs.docproc.modules.SiftFeatures
import org.opencv.core.CvType
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import java.util.Random
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Template matching does not depend on the order OpenCV hands keypoints over — the cases of
 * `tests/test_sift_order.py`.
 *
 * OpenCV orders SIFT keypoints with `std::sort` and cuts its `nfeatures` budget with `std::nth_element`; the order of
 * ties depends on the C++ library it was built with. MAGSAC samples by index, so the Windows wheel and Linux reached
 * different decisions on one STS card (conformance D-07/D-08, 2026-10-08). `SiftFeatures.sort` sorts the keypoints
 * itself and cuts the budget from that order. No models needed.
 */
class SiftOrderTests {

    @BeforeTest
    fun loadNatives() {
        NativeLibraries.load()
    }

    /** 300 keypoints at random places and responses, and 20 tied on response, told apart by position only. */
    private fun keypoints(): List<KeyPoint> {
        val rng = Random(0)
        val kp = ArrayList<KeyPoint>()
        repeat(300) {
            kp += KeyPoint(rng.nextFloat() * 500f, rng.nextFloat() * 500f, 3f, 0f, rng.nextFloat(), 0)
        }
        for (x in 0 until 20) {
            kp += KeyPoint(x.toFloat(), 10f, 3f, 0f, 0.5f, 0)
        }
        return kp
    }

    /** Descriptor row i belongs to keypoint i: (4 floats, i * 4 ..). */
    private fun descriptors(n: Int): Mat {
        val m = Mat(n, 4, CvType.CV_32F)
        m.put(0, 0, FloatArray(n * 4) { it.toFloat() })
        return m
    }

    private fun shuffled(seed: Long): Pair<Array<KeyPoint>, Mat> {
        val kp = keypoints()
        val desc = descriptors(kp.size)
        val all = FloatArray(kp.size * 4)
        desc.get(0, 0, all)
        val idx = kp.indices.toMutableList()
        idx.shuffle(java.util.Random(seed))
        val out = Mat(kp.size, 4, CvType.CV_32F)
        out.put(0, 0, FloatArray(kp.size * 4) { all[idx[it / 4] * 4 + it % 4] })
        desc.release()
        return Array(kp.size) { kp[idx[it]] } to out
    }

    private fun signature(kp: Array<KeyPoint>, desc: Mat): Pair<List<Triple<Double, Double, Float>>, List<Float>> {
        val values = FloatArray(desc.rows() * desc.cols())
        desc.get(0, 0, values)
        return kp.map { Triple(it.pt.x, it.pt.y, it.response) } to values.toList()
    }

    @Test
    fun theOrderHandedOverDoesNotMatter() {
        val results = (0L until 5L).map { seed ->
            val (kp, desc) = shuffled(seed)
            SiftFeatures.sort(kp, desc, 100).use { signature(it.keypoints, it.descriptors!!) }
        }
        for (r in results.drop(1)) assertEquals(results[0], r)
    }

    @Test
    fun theBudgetKeepsTheStrongestAndDescriptorsFollowTheirKeypoints() {
        val original = keypoints()
        val (kp, desc) = shuffled(1)
        SiftFeatures.sort(kp, desc, 50).use { got ->
            val responses = got.keypoints.map { it.response }
            assertEquals(50, got.keypoints.size)
            assertEquals(responses.sortedDescending(), responses)
            assertTrue(responses.min() >= original.map { it.response }.sortedDescending()[49])
            val values = FloatArray(50 * 4)
            got.descriptors!!.get(0, 0, values)
            for ((row, k) in got.keypoints.withIndex()) {
                val i = original.indexOfFirst { it.pt.x == k.pt.x && it.pt.y == k.pt.y && it.response == k.response }
                for (c in 0 until 4) assertEquals((i * 4 + c).toFloat(), values[row * 4 + c])
            }
        }
        desc.release()
    }

    @Test
    fun tiesOnResponseAreBrokenByYThenX() {
        val kp = arrayOf(
            KeyPoint(5f, 10f, 3f, 0f, 0.5f, 0), KeyPoint(2f, 10f, 3f, 0f, 0.5f, 0),
            KeyPoint(9f, 4f, 3f, 0f, 0.5f, 0), KeyPoint(1f, 1f, 3f, 0f, 0.9f, 0),
        )
        val desc = descriptors(kp.size)
        SiftFeatures.sort(kp, desc, null).use { got ->
            // strongest first; then the smaller y; then the smaller x
            assertEquals(listOf(1.0 to 1.0, 9.0 to 4.0, 2.0 to 10.0, 5.0 to 10.0), got.keypoints.map { it.pt.x to it.pt.y })
        }
        desc.release()
    }

    @Test
    fun noKeypointsGiveNoDescriptors() {
        SiftFeatures.sort(emptyArray(), Mat(), 100).use { got ->
            assertTrue(got.keypoints.isEmpty() && got.descriptors == null)
        }
    }
}
