package net.russiandocs.docproc.preprocess

import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.models.ModelInput
import net.russiandocs.docproc.tensors.Dtype
import net.russiandocs.docproc.tensors.NdArray

/**
 * The input of an oriented-box detector: the YOLO letterbox, handed over as float32 NCHW scaled to [0, 1].
 * Port of `processing/preprocessing.py::OBBPreprocessing`.
 *
 * The letterbox is [Yolo]'s, unchanged — the reference's class inherits it. What differs is the tensor: the
 * standard ultralytics OBB export expects channels first and divided by 255, where the baked-normalisation
 * detectors take NHWC 0-255. (`AddressLines` is the one deployed model still on that convention; see CLAUDE.md,
 * «ONNX export convention».)
 */
public class YoloObb(input: ModelInput) : Preprocessor {

    private val letterbox = Yolo(input)

    override fun apply(image: Image): Pair<NdArray, Meta> {
        val (nhwc, meta) = letterbox.apply(image)
        require(nhwc.dtype == Dtype.UINT8 && nhwc.shape.size == 4) {
            "preprocess: YOLOOBB expects a uint8 NHWC tensor from the letterbox, got $nhwc"
        }
        val height = nhwc.shape[1]
        val width = nhwc.shape[2]
        val channels = nhwc.shape[3]
        val pixels = nhwc.asUInt8()

        // `tensor.astype(np.float32).transpose(0, 3, 1, 2) / 255.0`: the division is float32.
        val plane = height * width
        val out = FloatArray(channels * plane)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val base = (y * width + x) * channels
                for (c in 0 until channels) {
                    out[c * plane + y * width + x] = (pixels[base + c].toInt() and 0xff).toFloat() / 255.0f
                }
            }
        }
        return NdArray.fromFloat32(out, 1, channels, height, width) to meta
    }
}
