package net.russiandocs.docproc.tensors

import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * The small numeric helpers the pipeline needs, with NumPy's exact semantics.
 *
 * Every one of these has a plausible alternative that is subtly different, and each difference changes
 * a discrete output rather than a float.
 */
public object Ops {

    /**
     * `np.argmax`: the index of the FIRST maximum.
     *
     * **Strict `>`, never `>=`.** On a tie NumPy keeps the earlier index, and reversing that flips a CTC
     * timestep and changes a character — an exact-match failure with no float anywhere near it.
     */
    public fun argmax(values: FloatArray): Int {
        require(values.isNotEmpty()) { "ops: argmax of an empty vector" }
        var best = 0
        for (i in 1 until values.size) {
            if (values[i] > values[best]) {
                best = i
            }
        }
        return best
    }

    /** `np.argmax` over one row of a flat `[rows, cols]` buffer. */
    public fun argmaxRow(values: FloatArray, offset: Int, count: Int): Int {
        require(count > 0) { "ops: argmax of an empty row" }
        var best = 0
        for (i in 1 until count) {
            if (values[offset + i] > values[offset + best]) {
                best = i
            }
        }
        return best
    }

    public fun max(values: FloatArray): Double {
        require(values.isNotEmpty()) { "ops: max of an empty vector" }
        var best = values[0]
        for (v in values) {
            if (v > best) {
                best = v
            }
        }
        return best.toDouble()
    }

    /**
     * Cosine distance, `1 - cos(a, b)`, as scipy and sklearn define it.
     *
     * A zero-norm vector gives distance 1 rather than NaN, matching scipy: an embedding of all zeros is
     * maximally distant from everything, not undefined. NaN here would propagate into the nearest-centroid
     * comparison and make every branch false, so the document type would come back NONE for a reason
     * nothing reports.
     */
    public fun cosineDistance(a: FloatArray, b: FloatArray): Double {
        require(a.size == b.size) { "ops: cosine of ${a.size} and ${b.size}" }
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (i in a.indices) {
            val x = a[i].toDouble()
            val y = b[i].toDouble()
            dot += x * y
            na += x * x
            nb += y * y
        }
        if (na == 0.0 || nb == 0.0) {
            return 1.0
        }
        return 1.0 - dot / (sqrt(na) * sqrt(nb))
    }

    public fun euclideanDistance(a: FloatArray, b: FloatArray): Double {
        require(a.size == b.size) { "ops: euclidean of ${a.size} and ${b.size}" }
        var sum = 0.0
        for (i in a.indices) {
            val d = a[i].toDouble() - b[i].toDouble()
            sum += d * d
        }
        return sqrt(sum)
    }

    /**
     * `np.round(x, n)`: half to even, at `n` decimal places.
     *
     * Scaling and unscaling by a power of ten is what NumPy does, and it is reproduced rather than
     * improved: a decimal-based implementation would round some values differently, and the results are
     * compared to three digits.
     */
    public fun roundHalfEven(value: Double, digits: Int): Double {
        val scale = 10.0.pow(digits)
        return Math.rint(value * scale) / scale
    }

    /**
     * The sigmoid, in float32.
     *
     * **`exp` on the JVM takes and returns Double (D-05).** Every accumulation in this port stays
     * float32, so the promotion is made explicit and narrowed straight back — `exp(x.toDouble()).toFloat()`
     * — rather than being allowed to widen a whole computation silently. Where a whole Mat needs it,
     * `Core.exp` on a CV_32F Mat keeps the depth and is preferred.
     */
    public fun sigmoid(x: Float): Float = (1.0 / (1.0 + kotlin.math.exp(-x.toDouble()))).toFloat()

    /**
     * Softmax over a vector, in float32, with the max subtracted first.
     *
     * Subtracting the max is what NumPy-based code does for stability and is not optional here: the
     * detector logits reach magnitudes where a bare `exp` overflows to infinity, and the resulting NaNs
     * make every comparison false.
     */
    public fun softmax(values: FloatArray): FloatArray {
        if (values.isEmpty()) {
            return values
        }
        var maximum = values[0]
        for (v in values) {
            if (v > maximum) {
                maximum = v
            }
        }
        val out = FloatArray(values.size)
        var sum = 0.0f
        for (i in values.indices) {
            val e = kotlin.math.exp((values[i] - maximum).toDouble()).toFloat()
            out[i] = e
            sum += e
        }
        if (sum != 0.0f) {
            for (i in out.indices) {
                out[i] = out[i] / sum
            }
        }
        return out
    }

    /**
     * `np.var`, TWO-PASS.
     *
     * **Not `E[x²] − E[x]²`.** NumPy subtracts the mean first, and the one-pass form loses roughly seven
     * significant digits at the magnitudes the deskewer works with (255·W). That is enough to flip the
     * argmax between two adjacent, nearly-equal angles — and the output is a discrete choice of angle
     * that rotates the image, so the error is not small anywhere downstream. Predicted before it could
     * happen, in the Go port's T10.
     */
    public fun variance(values: DoubleArray): Double {
        if (values.isEmpty()) {
            return 0.0
        }
        var mean = 0.0
        for (v in values) {
            mean += v
        }
        mean /= values.size
        var sum = 0.0
        for (v in values) {
            val d = v - mean
            sum += d * d
        }
        return sum / values.size
    }

    /** Whether two doubles agree within an absolute tolerance. Used by tests, not by the pipeline. */
    public fun close(a: Double, b: Double, atol: Double = 1e-9): Boolean = abs(a - b) <= atol
}
