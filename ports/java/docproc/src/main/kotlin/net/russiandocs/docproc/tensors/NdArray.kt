package net.russiandocs.docproc.tensors

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** The element types the `.npy` subset allows. Names match the NumPy descriptors. */
public enum class Dtype {
    /** `<f4` — float32. What every model input and output uses. */
    FLOAT32,

    /** `<f8` — float64. Appears in intermediates, never in a model tensor. */
    FLOAT64,

    /** `|u1` — uint8. Images and the OCR input. */
    UINT8,

    /** `<i8` — int64. Label indices. */
    INT64,

    /** `<U<n>` — fixed-width UTF-32 strings. Only `centers.npz` uses it. */
    UNICODE,
}

/**
 * A dense, C-contiguous N-dimensional array — the port's stand-in for `numpy.ndarray`.
 *
 * Bytes rather than a typed array, deliberately. The dtype is not known until a `.npy` header or a
 * `model.json` has been read, so a generic `NdArray<T>` would need reflection at every load site;
 * all three preceding ports reached the same conclusion and settled on a byte payload plus a dtype
 * tag. Keeping that identical across languages is worth more here than type safety at the seams.
 *
 * **C-contiguous only.** Fortran order is an error, not a mode: the reference never produces it, and
 * silently accepting it would let a transposed tensor reach a model and be graded as a numeric
 * divergence instead of a shape bug.
 */
public class NdArray(
    /** Raw little-endian payload, C-contiguous. */
    public val data: ByteArray,
    /** Dimensions. Empty means a SCALAR — one element, not zero. */
    public val shape: IntArray,
    public val dtype: Dtype,
    /**
     * Bytes per element. For [Dtype.UNICODE] this is `4 * n`, because NumPy stores `<U<n>` as
     * fixed-width UTF-32 padded with NULs — the trap that turns a naive byte-slicing label decoder
     * into a list of empty strings.
     */
    public val itemSize: Int,
) {
    init {
        val expected = count(shape).toLong() * itemSize
        require(data.size.toLong() == expected) {
            "tensor: payload is ${data.size} bytes, shape ${describe(shape)} of " +
                "$itemSize-byte items needs $expected"
        }
    }

    public val length: Int get() = count(shape)

    /** float32 view. Throws on any other dtype rather than reinterpreting. */
    public fun asFloat32(): FloatArray {
        check(dtype == Dtype.FLOAT32) { "tensor: dtype is $dtype, not FLOAT32" }
        val out = FloatArray(length)
        buffer().asFloatBuffer().get(out)
        return out
    }

    public fun asFloat64(): DoubleArray {
        check(dtype == Dtype.FLOAT64) { "tensor: dtype is $dtype, not FLOAT64" }
        val out = DoubleArray(length)
        buffer().asDoubleBuffer().get(out)
        return out
    }

    public fun asInt64(): LongArray {
        check(dtype == Dtype.INT64) { "tensor: dtype is $dtype, not INT64" }
        val out = LongArray(length)
        buffer().asLongBuffer().get(out)
        return out
    }

    public fun asUInt8(): ByteArray {
        check(dtype == Dtype.UINT8) { "tensor: dtype is $dtype, not UINT8" }
        return data
    }

    /**
     * Decodes a `<U<n>` array to strings.
     *
     * NumPy stores these as fixed-width UTF-32LE, NUL-padded to `n` code points. Slicing the bytes
     * naively yields empty strings, which is how a label array turns into nine blanks that then match
     * nothing — a failure that reads like a broken model rather than a broken decoder.
     */
    public fun asUnicode(): Array<String> {
        check(dtype == Dtype.UNICODE) { "tensor: dtype is $dtype, not UNICODE" }
        val codePoints = itemSize / 4
        val buffer = buffer()
        return Array(length) { i ->
            val sb = StringBuilder(codePoints)
            for (c in 0 until codePoints) {
                val cp = buffer.getInt((i * codePoints + c) * 4)
                if (cp == 0) {
                    break // NUL padding: the string ends here, the field does not
                }
                // appendCodePoint, not append(Char): a code point above U+FFFF needs a surrogate
                // pair, and Kotlin's Char is a UTF-16 unit rather than a code point.
                sb.appendCodePoint(cp)
            }
            sb.toString()
        }
    }

    private fun buffer(): ByteBuffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)

    override fun toString(): String = "NdArray${describe(shape)} $dtype"

    public companion object {
        /** Total element count. A zero-length shape is a SCALAR: one element. */
        public fun count(shape: IntArray): Int {
            var n = 1
            for (d in shape) {
                require(d >= 0) { "tensor: negative dimension in ${describe(shape)}" }
                n *= d
            }
            return n
        }

        public fun describe(shape: IntArray): String = "(${shape.joinToString(", ")})"

        public fun fromFloat32(values: FloatArray, vararg shape: Int): NdArray {
            val bytes = ByteArray(values.size * 4)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(values)
            return NdArray(bytes, shape, Dtype.FLOAT32, 4)
        }

        public fun fromFloat64(values: DoubleArray, vararg shape: Int): NdArray {
            val bytes = ByteArray(values.size * 8)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer().put(values)
            return NdArray(bytes, shape, Dtype.FLOAT64, 8)
        }

        public fun fromUInt8(values: ByteArray, vararg shape: Int): NdArray =
            NdArray(values, shape, Dtype.UINT8, 1)

        public fun fromInt64(values: LongArray, vararg shape: Int): NdArray {
            val bytes = ByteArray(values.size * 8)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asLongBuffer().put(values)
            return NdArray(bytes, shape, Dtype.INT64, 8)
        }
    }
}
