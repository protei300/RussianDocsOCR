package net.russiandocs.docproc.modules

import org.opencv.core.CvType
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfKeyPoint
import org.opencv.features2d.SIFT

/** SIFT keypoints with their descriptors (one row per keypoint, same order). The Mat is owned by this object. */
public class Features(public val keypoints: Array<KeyPoint>, public val descriptors: Mat?) : AutoCloseable {
    override fun close() {
        descriptors?.release()
    }
}

/**
 * SIFT keypoints and descriptors in an order every platform agrees on. Port of `detect_features`
 * (`page_registration.py`, 2026-10-08).
 *
 * OpenCV orders its keypoints with `std::sort` (duplicate removal) and cuts an `nfeatures` budget with
 * `std::nth_element`; both leave ties in an order that depends on the C++ library OpenCV was built with. MAGSAC
 * samples matches by index, so the same keypoints in another order give another homography: on an STS of the 2019
 * form the Windows wheel kept the Borders canvas (skew 0.0076) and the same code on Linux re-cut it from the
 * template (0.0118) — conformance D-07/D-08. So the detector runs WITHOUT a budget (`SIFT.create(0)`), the
 * keypoints are sorted here by response (descending) and then by y, x, size, angle and octave, and the budget is
 * cut from that order. The comparison is on float64 values and the sort is stable, as `np.lexsort` is.
 */
public object SiftFeatures {

    /** Detects with [sift] (created with `nfeatures = 0`), sorts, and cuts [budget]. */
    public fun detect(sift: SIFT, gray: Mat, mask: Mat?, budget: Int?): Features {
        val kpMat = MatOfKeyPoint()
        val desc = Mat()
        try {
            sift.detectAndCompute(gray, mask ?: Mat(), kpMat, desc)
            return sort(kpMat.toArray(), desc, budget)
        } finally {
            kpMat.release()
            desc.release()
        }
    }

    /**
     * The keypoints in the platform-free order with their descriptor rows, the budget cut from that order. [desc]
     * is only read; the result owns its own descriptor Mat. An empty detection gives no keypoints and no
     * descriptors, as the reference does.
     */
    public fun sort(kp: Array<KeyPoint>, desc: Mat?, budget: Int?): Features {
        if (kp.isEmpty() || desc == null || desc.empty()) {
            return Features(emptyArray(), null)
        }
        // Compared as `np.lexsort` over float64 keys does: the main key is -response, then y, x, size, angle,
        // octave. `<` and `>` rather than Double.compareTo, which tells -0.0 from 0.0 and NaN from itself.
        fun cmp(a: Double, b: Double): Int = if (a < b) -1 else if (a > b) 1 else 0
        val order = kp.indices.sortedWith { i, j ->
            val a = kp[i]
            val b = kp[j]
            var c = cmp(-a.response.toDouble(), -b.response.toDouble())
            if (c == 0) c = cmp(a.pt.y, b.pt.y)
            if (c == 0) c = cmp(a.pt.x, b.pt.x)
            if (c == 0) c = cmp(a.size.toDouble(), b.size.toDouble())
            if (c == 0) c = cmp(a.angle.toDouble(), b.angle.toDouble())
            if (c == 0) c = cmp(a.octave.toDouble(), b.octave.toDouble())
            c
        }.let { if (budget != null && budget > 0 && budget < it.size) it.subList(0, budget) else it }

        val cols = desc.cols()
        val source = FloatArray(desc.rows() * cols)
        desc.get(0, 0, source)
        val sorted = FloatArray(order.size * cols)
        for ((row, i) in order.withIndex()) {
            System.arraycopy(source, i * cols, sorted, row * cols, cols)
        }
        val out = Mat(order.size, cols, CvType.CV_32F)
        out.put(0, 0, sorted)
        return Features(Array(order.size) { kp[order[it]] }, out)
    }
}
