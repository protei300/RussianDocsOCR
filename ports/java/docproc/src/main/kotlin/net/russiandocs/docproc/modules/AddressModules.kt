package net.russiandocs.docproc.modules

import net.russiandocs.docproc.config.ModelPaths
import net.russiandocs.docproc.geometry.Chain
import net.russiandocs.docproc.geometry.Homography
import net.russiandocs.docproc.geometry.Offset
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.models.Loader
import net.russiandocs.docproc.models.Model
import net.russiandocs.docproc.models.ObbDetectionModel
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.postprocess.ClassResult
import net.russiandocs.docproc.postprocess.ObbBox
import net.russiandocs.docproc.tensors.PyNum
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.max

/**
 * Finds the oriented address lines on the residence registration page (INTPASSPORTADDR). Port of
 * `pipeline_modules/address_lines_detector/address_lines_detector.py`.
 *
 * A YOLO-OBB detector: a tilted or skewed line is captured directly, with no deskew in front of it. Every line is
 * cut out and turned upright ([cropRotated]), in reading order, ready for the word split.
 */
public class AddressLinesDetector(
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
) : AutoCloseable {

    private val model = ObbDetectionModel(
        File(ModelPaths.resolve(root, paths, "AddressLinesDetector"), "ONNX").path, device, threads, root)

    /**
     * The oriented boxes and, one for each, the upright patch (`predict_transform`). The caller closes the patches.
     * On any failure the patches cut so far are closed before the exception leaves.
     */
    public fun predictTransform(canvas: Image): Pair<List<ObbBox>, List<Image>> {
        val boxes = model.predict(canvas)
        val patches = ArrayList<Image>(boxes.size)
        try {
            for (box in boxes) {
                patches += cropRotated(canvas, box)
            }
            return boxes to patches
        } catch (e: Throwable) {
            patches.forEach { it.close() }
            throw e
        }
    }

    override fun close(): Unit = model.close()

    public companion object {
        /** `max(1, int(round(v)))`: Python's `round` on a float is half to even. */
        internal fun side(v: Double): Int = max(1, PyNum.roundHalfEvenToInt(v))

        /** `cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), 1.0)` as a 2x3 array. */
        internal fun rotationMatrix(box: ObbBox): Array<DoubleArray> {
            val m = Imgproc.getRotationMatrix2D(Point(box.cx, box.cy), Math.toDegrees(box.angle), 1.0)
            try {
                return Array(2) { r -> DoubleArray(3) { c -> m.get(r, c)[0] } }
            } finally {
                m.release()
            }
        }

        /**
         * Cuts a rotated rectangle and returns it upright (`crop_rotated`): the same rotation about the centre,
         * then a sub-pixel window of the rounded size.
         */
        public fun cropRotated(img: Image, box: ObbBox): Image {
            val wi = side(box.w)
            val hi = side(box.h)
            val m = Imgproc.getRotationMatrix2D(Point(box.cx, box.cy), Math.toDegrees(box.angle), 1.0)
            val rotated = Mat()
            val patch = Mat()
            try {
                Imgproc.warpAffine(img.mat, rotated, m, Size(img.width.toDouble(), img.height.toDouble()),
                    Imgproc.INTER_LINEAR)
                Imgproc.getRectSubPix(rotated, Size(wi.toDouble(), hi.toDouble()), Point(box.cx, box.cy), patch)
                return Image.wrap(patch)
            } catch (e: Throwable) {
                patch.release()
                throw e
            } finally {
                m.release()
                rotated.release()
            }
        }

        /**
         * The map from a [cropRotated] patch back to the image it was cut from (`crop_geometry`): the same rotation
         * and the same sub-pixel window, so a box found on the patch lands on the image. `getRectSubPix` samples
         * patch pixel (u, v) at (u + cx - (wi - 1) / 2, v + cy - (hi - 1) / 2), pixel centres on integers.
         */
        public fun cropGeometry(box: ObbBox): Chain {
            val wi = side(box.w)
            val hi = side(box.h)
            val window = Offset(-(box.cx - (wi - 1) / 2.0), -(box.cy - (hi - 1) / 2.0))
            return Chain(listOf(Homography(rotationMatrix(box)), window))
        }
    }
}

/**
 * Printed or handwritten, for one address-line crop. Port of
 * `pipeline_modules/address_textkind_classifier/address_textkind_classifier.py`.
 *
 * A line is wide and short, so the crop is put on a grey square (aspect kept) before the classifier's own square
 * resize — the glyph proportions are not distorted.
 */
public class AddressTextKindClassifier(
    root: String,
    paths: Map<String, String>,
    device: Device,
    threads: Int,
) : AutoCloseable {

    private val model: Model = Loader.load(
        File(ModelPaths.resolve(root, paths, "AddressTextKindClassifier"), "ONNX").path, device, threads, root)

    /** `('printed' | 'handwritten', P(handwritten))`. The probability is the RAW score, whichever label won. */
    public fun predict(patch: Image): Pair<String, Double> {
        letterboxSquare(patch).use { square ->
            val result = model.predict(square)[0] as? ClassResult
                ?: throw IllegalStateException("modules: AddressTextKind output is not a ClassResult")
            return result.label to result.score
        }
    }

    override fun close(): Unit = model.close()

    internal companion object {
        /** `letterbox_square`: a (114, 114, 114) square, the crop centred (floor division for the offset). */
        fun letterboxSquare(img: Image): Image {
            val h = img.height
            val w = img.width
            val size = max(h, w)
            val canvas = Mat(size, size, img.mat.type(), org.opencv.core.Scalar(114.0, 114.0, 114.0, 0.0))
            val y0 = (size - h) / 2
            val x0 = (size - w) / 2
            val window = canvas.submat(Rect(x0, y0, w, h))
            try {
                img.mat.copyTo(window)
            } finally {
                window.release()
            }
            return Image.wrap(canvas)
        }
    }
}

/**
 * Finds the registration stamps on the residence registration page, issue #27. Port of
 * `pipeline_modules/registration_stamps/registration_stamps.py`.
 *
 * The same family as [AddressLinesDetector] (YOLO-OBB), run on the same canvas. One oriented box per stamp
 * impression, classed `registered`, `deregistered` or `unclear`. The stamp is located, not read.
 */
public class RegistrationStamps private constructor(private val model: ObbDetectionModel) : AutoCloseable {

    public fun predict(canvas: Image): List<ObbBox> = model.predict(canvas)

    override fun close(): Unit = model.close()

    public companion object {
        /** Model label to the kind reported in the results. */
        public val KINDS: Map<String, String> = mapOf(
            "Stamp_registered" to "registered",
            "Stamp_deregistered" to "deregistered",
            "Stamp_unclear" to "unclear",
        )

        /**
         * Opens the detector, or returns null when this weight set has none (models-v10 and older): the address is
         * then read as before, without stamps. Said once, on stderr — `System.out` belongs to the conformance payload.
         */
        public fun openOrNull(root: String, paths: Map<String, String>, device: Device, threads: Int): RegistrationStamps? {
            val dir = paths["RegistrationStamps"]?.let { File(ModelPaths.resolve(root, paths, "RegistrationStamps"), "ONNX") }
            if (dir == null || !File(dir, "model.json").isFile || !File(dir, "model.onnx").isFile) {
                System.err.println(
                    "[!] RegistrationStamps weights not found (models/RegistrationStamps): the registration page " +
                        "is read without stamps. Run scripts/fetch_models.py for a weight set that has them.",
                )
                return null
            }
            return RegistrationStamps(ObbDetectionModel(dir.path, device, threads, root))
        }
    }
}
