package net.russiandocs.docproc.inference

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import net.russiandocs.docproc.pipeline.Device
import net.russiandocs.docproc.tensors.Dtype
import net.russiandocs.docproc.tensors.NdArray
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * One ONNX Runtime session, plus the two rules that make it safe to share.
 *
 * ### The GPU lock
 *
 * Concurrent `run` calls on ONE CUDA session wedge the device. Measured, not theoretical: Python hit
 * `cudaErrorIllegalAddress` after roughly 200 calls, and the Go port degraded by more than 90x — 6.6 s
 * with the lock against over 600 s without, never finishing. The lock is held around `run` ONLY, so
 * five different models keep their parallelism, and the results are read after releasing it.
 *
 * **CPU sessions take no lock, deliberately.** ONNX Runtime's CPU path is re-entrant, and locking it
 * would serialise the quality group that exists precisely to run four classifiers at once.
 *
 * ### Casting to the declared dtype
 *
 * Each input is cast to what the model declares. The detectors want float32 and the OCR nets want
 * uint8; feeding the wrong one is not a graceful failure — ONNX Runtime either refuses or reinterprets
 * the bytes.
 */
public class Session(
    modelPath: String,
    public val device: Device,
    intraOpThreads: Int,
) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputNames: List<String>
    private val inputTypes: Map<String, OnnxJavaType>

    /** Output names in the order the model DECLARES them. */
    public val outputNames: List<String>

    /** Null on CPU: see the type note. */
    private val gate: Any?

    init {
        val options = OrtSession.SessionOptions()
        try {
            if (device == Device.GPU) {
                // Throws if the provider is unavailable, which the caller's [gpu, cpu] attempt loop
                // handles. What it CANNOT catch is a container started without --gpus, where the
                // provider segfaults instead — hence the device-node probe before we ever get here.
                options.addCUDA(0)
            }
            if (intraOpThreads > 0) {
                // Pinned for conformance runs; see the note on the CLI's IntraOpThreads.
                options.setIntraOpNumThreads(intraOpThreads)
            }
            session = env.createSession(modelPath, options)
        } catch (e: Throwable) {
            options.close()
            throw e
        }

        // LinkedHashMap iteration order is the declaration order the binding reports, and that is what
        // "inputs are positional" below relies on.
        inputNames = session.inputNames.toList()
        inputTypes = session.inputInfo.mapValues { (_, info) ->
            (info.info as TensorInfo).type
        }
        outputNames = session.outputNames.toList()
        gate = if (device == Device.GPU) Any() else null
    }

    /**
     * Runs the model. Inputs are positional; the caller supplies them in declaration order.
     *
     * Outputs are collected **by declared name**, not by iteration order: `DocTypeAngles` has two
     * outputs whose meanings are not interchangeable — embeddings and an angle — and an unexpected
     * reordering must be visible rather than quietly swapping two heads.
     */
    public fun run(inputs: List<NdArray>): List<NdArray> {
        require(inputs.size == inputNames.size) {
            "inference: model wants ${inputNames.size} inputs, got ${inputs.size}"
        }

        val feeds = LinkedHashMap<String, OnnxTensor>(inputs.size)
        try {
            for ((i, name) in inputNames.withIndex()) {
                feeds[name] = toTensor(name, inputs[i], inputTypes.getValue(name))
            }

            val result = if (gate != null) {
                // Acquire, run, release — and read the results AFTER releasing, outside the lock.
                synchronized(gate) { session.run(feeds) }
            } else {
                session.run(feeds)
            }

            result.use { outputs ->
                return outputNames.map { name ->
                    fromTensor(outputs.get(name).orElseThrow {
                        IllegalStateException("inference: model declared output $name but did not " +
                            "return it")
                    } as OnnxTensor)
                }
            }
        } finally {
            feeds.values.forEach { it.close() }
        }
    }

    private fun toTensor(name: String, array: NdArray, want: OnnxJavaType): OnnxTensor {
        val shape = array.shape.map { it.toLong() }.toLongArray()
        return when (want) {
            OnnxJavaType.FLOAT -> {
                val values = when (array.dtype) {
                    Dtype.FLOAT32 -> array.asFloat32()
                    // uint8 to float32 WITHOUT scaling: normalisation is baked into the graphs, and
                    // dividing here would silently double-normalise.
                    Dtype.UINT8 -> FloatArray(array.length) { i ->
                        (array.data[i].toInt() and 0xff).toFloat()
                    }
                    Dtype.FLOAT64 -> {
                        val src = array.asFloat64()
                        FloatArray(src.size) { src[it].toFloat() }
                    }
                    else -> throw IllegalStateException(
                        "inference: $name wants float32, cannot cast from ${array.dtype}")
                }
                OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(values), shape)
            }

            OnnxJavaType.UINT8 -> {
                val values = when (array.dtype) {
                    Dtype.UINT8 -> array.asUInt8()
                    else -> throw IllegalStateException(
                        "inference: $name wants uint8, cannot cast from ${array.dtype}")
                }
                // The UINT8 overload must be named explicitly: without it the binding infers INT8 from
                // a ByteBuffer and the model rejects the feed with a type error that names neither.
                OnnxTensor.createTensor(env, ByteBuffer.wrap(values), shape, OnnxJavaType.UINT8)
            }

            OnnxJavaType.INT64 -> {
                val values = when (array.dtype) {
                    Dtype.INT64 -> array.asInt64()
                    else -> throw IllegalStateException(
                        "inference: $name wants int64, cannot cast from ${array.dtype}")
                }
                OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(values), shape)
            }

            else -> throw IllegalStateException("inference: $name declares unsupported type $want")
        }
    }

    /**
     * Copies an output tensor into an [NdArray].
     *
     * Through the raw ByteBuffer rather than `getValue()`, and that is not an optimisation: `getValue()`
     * materialises a nested `Object[]` of boxed arrays — for a `[1, 8400, 26]` detector head that is
     * 8400 float arrays plus the boxing, per call, on the hot path. The buffer copy is one pass.
     */
    private fun fromTensor(tensor: OnnxTensor): NdArray {
        val info = tensor.info
        val shape = info.shape.map { it.toInt() }.toIntArray()
        val buffer = tensor.byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return when (info.type) {
            OnnxJavaType.FLOAT -> NdArray(bytes, shape, Dtype.FLOAT32, 4)
            OnnxJavaType.DOUBLE -> NdArray(bytes, shape, Dtype.FLOAT64, 8)
            OnnxJavaType.UINT8, OnnxJavaType.INT8 -> NdArray(bytes, shape, Dtype.UINT8, 1)
            OnnxJavaType.INT64 -> NdArray(bytes, shape, Dtype.INT64, 8)
            else -> throw IllegalStateException(
                "inference: unsupported output type ${info.type}")
        }
    }

    override fun close() {
        session.close()
        // The environment is a process-wide singleton in the JVM binding — closing it here would tear
        // down every other session. Left alone deliberately.
    }
}
