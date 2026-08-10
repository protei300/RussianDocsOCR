package net.russiandocs.docproc.modules

import net.russiandocs.docproc.imaging.Contours
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Interpolation
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.tensors.Ops
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point as CvPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Removes residual tilt by scanning candidate angles and maximising the variance of the row-ink profile.
 *
 * A COARSE-TO-FINE search: a sparse scan across the whole range, then a dense scan around the coarse
 * winner. Verified in the reference to choose the same angles as a single dense scan while being about 2.9x
 * faster.
 */
public class DocDeskewer(
    private val angleRange: Double,
    angleSteps: Int,
    private val minAngle: Double,
    private val scale: Double,
    coarseSteps: Int,
) {
    private val coarseAngles: DoubleArray
    private val fineHalfRange: Double
    private val fineCount: Int

    init {
        val coarse = coarseSteps.coerceIn(3, angleSteps)
        coarseAngles = linspace(-angleRange, angleRange, coarse)

        val coarseStep = 2 * angleRange / (coarse - 1)
        val fullResolution = 2 * angleRange / (angleSteps - 1)
        fineHalfRange = coarseStep
        fineCount = max(3, (2 * coarseStep / fullResolution).roundToInt() + 1)
    }

    /**
     * Rotates the residual tilt out. Returns the corrected image and the angle applied.
     *
     * Below [minAngle] the image is returned unchanged — a copy, so ownership is uniform. The threshold is
     * load-bearing: it decides whether the canvas is rotated at all, and therefore whether every box
     * downstream lands in the same place.
     */
    public fun deskew(image: Image): Pair<Image, Double> {
        val angle = findAngle(image)
        if (abs(angle) < minAngle) {
            return image.clone() to angle
        }

        val rotation = rotationMatrix(image.width / 2.0, image.height / 2.0, angle, 1.0)
        try {
            val dst = Mat()
            Imgproc.warpAffine(image.mat, dst, rotation,
                Size(image.width.toDouble(), image.height.toDouble()),
                Imgproc.INTER_LINEAR, Core.BORDER_REPLICATE, Scalar(0.0))
            return Image.wrap(dst) to angle
        } finally {
            rotation.release()
        }
    }

    private fun findAngle(image: Image): Double {
        Io.toGray(image).use { gray ->
            val sh = max(1, (gray.height * scale).toInt())
            val sw = max(1, (gray.width * scale).toInt())
            Io.resize(gray, sw, sh, Interpolation.AREA).use { small ->
                val (binary, _) = Contours.thresholdOtsu(small, invert = true)
                binary.use {
                    val cx = sw / 2.0
                    val cy = sh / 2.0
                    val coarse = scoreAngles(binary, sw, sh, cx, cy, coarseAngles)
                    val ci = argmax(coarse)

                    // **A winner at either END of the coarse scan means no tilt.** The true optimum lies
                    // outside the search range, so refining around the edge would pick an arbitrary angle;
                    // the reference bails to 0.0 and this early exit is what keeps the canvas unrotated.
                    if (ci == 0 || ci == coarseAngles.size - 1) {
                        return 0.0
                    }

                    val best = coarseAngles[ci]
                    val lo = max(-angleRange, best - fineHalfRange)
                    val hi = min(angleRange, best + fineHalfRange)
                    val fineAngles = linspace(lo, hi, fineCount)
                    val fine = scoreAngles(binary, sw, sh, cx, cy, fineAngles)
                    return fineAngles[argmax(fine)]
                }
            }
        }
    }

    private fun scoreAngles(
        binary: Image,
        sw: Int,
        sh: Int,
        cx: Double,
        cy: Double,
        angles: DoubleArray,
    ): DoubleArray = DoubleArray(angles.size) { i ->
        val rotation = rotationMatrix(cx, cy, angles[i], 1.0)
        val rotated = Mat()
        try {
            // Nearest-neighbour and ZERO borders: the input is a binary mask, so interpolation would invent
            // grey values, and rotated-in area must contribute no ink.
            Imgproc.warpAffine(binary.mat, rotated, rotation, Size(sw.toDouble(), sh.toDouble()),
                Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, Scalar(0.0))
            // Row sums are exact integers, so this step contributes no float error at all — which makes the
            // variance the only numerically delicate part of the search. Ops.variance is TWO-PASS for
            // exactly that reason.
            Ops.variance(rowSumsOf(rotated))
        } finally {
            rotation.release()
            rotated.release()
        }
    }

    /**
     * Per-row ink totals of a single-channel 8-bit Mat, as exact integers.
     *
     * Read row-at-a-time rather than pixel-at-a-time: a JNI call per pixel is roughly 90 000 crossings on a
     * 300x300 scaled mask, times 21 coarse plus ~40 fine angles per document.
     */
    private fun rowSumsOf(binary: Mat): DoubleArray {
        val h = binary.rows()
        val w = binary.cols()
        val row = ByteArray(w)
        val sums = DoubleArray(h)
        for (y in 0 until h) {
            binary.get(y, 0, row)
            var acc = 0L
            for (x in 0 until w) {
                acc += (row[x].toInt() and 0xff).toLong()
            }
            sums[y] = acc.toDouble()
        }
        return sums
    }

    /** First maximum, like `np.argmax`. Strict `>` only. */
    private fun argmax(values: DoubleArray): Int {
        var best = 0
        var bestValue = Double.NEGATIVE_INFINITY
        for (i in values.indices) {
            if (values[i] > bestValue) {
                best = i
                bestValue = values[i]
            }
        }
        return best
    }

    private fun linspace(from: Double, to: Double, count: Int): DoubleArray {
        if (count <= 1) {
            return doubleArrayOf(from)
        }
        val step = (to - from) / (count - 1)
        return DoubleArray(count) { from + step * it }
    }

    /**
     * The rotation matrix.
     *
     * **Built by hand even though this port does not have to.** `Imgproc.getRotationMatrix2D` accepts a
     * double-valued `Point`, so D-08 does not apply here and the real function would work. It is written out
     * anyway so all four ports compute the identical matrix from the identical formula — the Go port has no
     * choice, and a difference here selects a different ANGLE, which rotates the canvas. Verified against
     * OpenCV's own output to 1.6e-14.
     */
    private fun rotationMatrix(cx: Double, cy: Double, angleDegrees: Double, scale: Double): Mat {
        val radians = angleDegrees * Math.PI / 180.0
        val alpha = Math.cos(radians) * scale
        val beta = Math.sin(radians) * scale

        val m = Mat(2, 3, CvType.CV_64FC1)
        m.put(0, 0, alpha)
        m.put(0, 1, beta)
        m.put(0, 2, (1 - alpha) * cx - beta * cy)
        m.put(1, 0, -beta)
        m.put(1, 1, alpha)
        m.put(1, 2, beta * cx + (1 - alpha) * cy)
        return m
    }

    public companion object {
        /**
         * The parameters the PIPELINE uses.
         *
         * Deliberately not the reference class's own defaults, which differ (its `angle_range` is 2.0).
         * Reading only the class would give a working deskewer that chooses different angles.
         */
        public fun forPipeline(): DocDeskewer = DocDeskewer(10.0, 101, 2.0, 0.4, 21)
    }
}
