package net.russiandocs.docproc.imaging

import net.russiandocs.docproc.tensors.PyNum
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfInt
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File

/** The interpolation modes this port uses, by their OpenCV values. */
public enum class Interpolation(public val value: Int) {
    /** `cv2.INTER_LINEAR` — OpenCV's default and what every resize here uses. */
    LINEAR(Imgproc.INTER_LINEAR),

    /** `cv2.INTER_AREA` — only the thumbnail. */
    AREA(Imgproc.INTER_AREA),
}

/**
 * Decode, resize and the other pixel-level primitives.
 *
 * **Everything goes through OpenCV, never through a JVM imaging library.** Measured in the Go spike:
 * OpenCV and a non-OpenCV JPEG decoder disagree by up to 14 LSB on 58–83% of pixels, because
 * libjpeg-turbo's IDCT differs from other implementations — enough that "1e-3 on numeric outputs" is
 * unreachable before inference even starts. The same applies to resize: `cv2.resize` with INTER_LINEAR
 * computes in FIXED POINT with 11-bit coefficients, not in float, and cannot be reproduced by hand.
 *
 * On the JVM the temptation is `javax.imageio`, and it is exactly the same mistake.
 */
public object Io {

    /**
     * Decodes to RGB.
     *
     * This is `BasePreprocessing.__call__` and the first half of `Pipeline._prepare_image`:
     * `imdecode(IMREAD_COLOR)` gives BGR, and the pipeline works in RGB. Format is sniffed from the
     * content, not from the extension — `tests/images/OCRv2` contains a file named `.png` that is not
     * one, and cv2 never cared because it sniffs too.
     */
    public fun decodeRgb(data: ByteArray): Image {
        val buffer = MatOfByte(*data)
        val bgr = try {
            Imgcodecs.imdecode(buffer, Imgcodecs.IMREAD_COLOR)
        } finally {
            buffer.release()
        }
        if (bgr == null || bgr.empty()) {
            bgr?.release()
            throw IllegalArgumentException("imaging: decoded an empty image (${data.size} bytes)")
        }
        return try {
            val rgb = Mat()
            Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB)
            Image.wrap(rgb)
        } finally {
            bgr.release()
        }
    }

    /**
     * Width and height of an encoded image, or `null` if it cannot be decoded.
     *
     * Still a FULL decode, on purpose: the caller uses this to reject an undecodable upload with an
     * immediate, actionable error rather than letting it become a mysterious failed job, and only a
     * real decode proves decodability. What it skips is the BGR-to-RGB conversion that [decodeRgb] owes
     * the pipeline and a caller wanting two integers does not — a second full pass, roughly 36 MB of
     * pointless copying on a phone photo.
     */
    public fun decodeSize(data: ByteArray): Pair<Int, Int>? {
        val buffer = MatOfByte(*data)
        var mat: Mat? = null
        return try {
            mat = Imgcodecs.imdecode(buffer, Imgcodecs.IMREAD_COLOR)
            if (mat == null || mat.empty()) null else mat.cols() to mat.rows()
        } catch (e: Exception) {
            // OpenCV's Java binding wraps a native error in CvException, which extends RuntimeException;
            // catching broadly here is deliberate, because "cannot decode" must never propagate.
            null
        } finally {
            mat?.release()
            buffer.release()
        }
    }

    public fun loadRgb(path: String): Image = decodeRgb(File(path).readBytes())

    /**
     * Resizes to an exact size.
     *
     * Argument order follows `cv2.resize`'s `dsize=(w, h)`, NOT numpy's `(h, w)`. The shipped model
     * input sizes are square, which hides an axis swap — the conformance suite therefore includes a
     * deliberately non-square resize.
     */
    public fun resize(src: Image, width: Int, height: Int, interp: Interpolation): Image {
        val dst = Mat()
        Imgproc.resize(src.mat, dst, Size(width.toDouble(), height.toDouble()), 0.0, 0.0, interp.value)
        return Image.wrap(dst)
    }

    /**
     * Shrinks so the longest side is at most [imgSize].
     *
     * The second half of `Pipeline._prepare_image`: `ratio = max(max(h, w) / img_size, 1)` — so it only
     * ever SHRINKS — then `int(w // ratio), int(h // ratio)`.
     *
     * The floor divisions go through [PyNum.floorDivInt], and that is load-bearing: for 2999x1777 the
     * correct answer is 1499, while `floor(2999 / ratio)` gives 1500. A canvas one pixel wider shifts
     * every box downstream, and the failure surfaces at a stage far from its cause. A consequence worth
     * stating because it surprises: this does NOT guarantee the long side equals [imgSize].
     */
    public fun fitToLongestSide(src: Image, imgSize: Int): Image {
        val h = src.height
        val w = src.width
        val ratio = maxOf(maxOf(h, w).toDouble() / imgSize, 1.0)
        if (ratio == 1.0) {
            // No resize at all when it already fits — not a resize by 1.0, which would still run the
            // interpolator and could perturb pixels.
            return src.clone()
        }
        return resize(src, PyNum.floorDivInt(w.toDouble(), ratio),
            PyNum.floorDivInt(h.toDouble(), ratio), Interpolation.LINEAR)
    }

    public fun toBgr(src: Image): Image {
        val dst = Mat()
        Imgproc.cvtColor(src.mat, dst, Imgproc.COLOR_RGB2BGR)
        return Image.wrap(dst)
    }

    public fun toGray(src: Image): Image {
        val dst = Mat()
        Imgproc.cvtColor(src.mat, dst, Imgproc.COLOR_RGB2GRAY)
        return Image.wrap(dst)
    }

    public fun copyMakeBorderConstant(
        src: Image,
        top: Int,
        bottom: Int,
        left: Int,
        right: Int,
        r: Double,
        g: Double,
        b: Double,
    ): Image {
        val dst = Mat()
        Core.copyMakeBorder(src.mat, dst, top, bottom, left, right, Core.BORDER_CONSTANT,
            Scalar(r, g, b))
        return Image.wrap(dst)
    }

    public fun newFilled(height: Int, width: Int, r: Double, g: Double, b: Double): Image =
        Image.wrap(Mat(height, width, CvType.CV_8UC3, Scalar(r, g, b)))

    /**
     * Rotates by a multiple of 90 degrees, counter-clockwise, as `np.rot90` does.
     *
     * `Core.rotate` takes a code rather than an angle, and the codes are CLOCKWISE while `np.rot90` is
     * counter-clockwise — so `k = 1` maps to `ROTATE_90_COUNTERCLOCKWISE`. Getting the direction
     * backwards produces an upside-down document for `k = 2` only, which passes a casual look at one
     * sample and fails the other three angles.
     */
    public fun rot90(src: Image, k: Int): Image {
        val turns = ((k % 4) + 4) % 4
        if (turns == 0) {
            return src.clone()
        }
        val dst = Mat()
        val code = when (turns) {
            1 -> Core.ROTATE_90_COUNTERCLOCKWISE
            2 -> Core.ROTATE_180
            else -> Core.ROTATE_90_CLOCKWISE
        }
        Core.rotate(src.mat, dst, code)
        return Image.wrap(dst)
    }

    /** Writes a PNG. The input is RGB; the encoder expects BGR, so it converts first. */
    public fun writePngFromRgb(path: String, rgb: Image) {
        toBgr(rgb).use { bgr ->
            File(path).parentFile?.mkdirs()
            if (!Imgcodecs.imwrite(path, bgr.mat)) {
                throw java.io.IOException("imaging: could not write $path")
            }
        }
    }

    /**
     * Writes a JPEG at the given quality. The input is RGB; the encoder expects BGR.
     *
     * Used only for the list-page thumbnail, where a lossy 96-px-wide image is the point. Nothing the
     * conformance suite compares goes through here.
     */
    public fun writeJpegFromRgb(path: String, rgb: Image, quality: Int) {
        toBgr(rgb).use { bgr ->
            File(path).parentFile?.mkdirs()
            val params = MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality)
            try {
                if (!Imgcodecs.imwrite(path, bgr.mat, params)) {
                    throw java.io.IOException("imaging: could not write $path")
                }
            } finally {
                params.release()
            }
        }
    }
}
