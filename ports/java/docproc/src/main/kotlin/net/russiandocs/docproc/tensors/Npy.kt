package net.russiandocs.docproc.tensors

import java.io.EOFException
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipInputStream

/**
 * Reader and writer for the `.npy` subset defined in `conformance/spec/npy-subset.md`.
 *
 * Every port needs this regardless of the harness, because `DocTypeAngles` ships its centroids as
 * `centers.npz` — a zip of three `.npy` members. The harness reuses it rather than adding a second
 * serialisation format.
 *
 * **The header is a Python literal, not JSON.** It uses single quotes, `True`/`False`, and a trailing
 * comma in one-element tuples (`'shape': (3,)`). A JSON parser appears to work on the common cases and
 * then fails on exactly those, which is why this parses the literal directly.
 */
public object Npy {

    private val MAGIC = byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(),
        'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte())

    public fun load(path: String): NdArray = File(path).inputStream().use { read(it, path) }

    public fun parse(blob: ByteArray, origin: String = "<memory>"): NdArray =
        blob.inputStream().use { read(it, origin) }

    public fun read(stream: InputStream, origin: String): NdArray {
        val magic = readExactly(stream, 6, origin)
        if (!magic.contentEquals(MAGIC)) {
            throw IllegalArgumentException("npy: $origin: not a .npy file")
        }

        val version = readExactly(stream, 2, origin)
        if (version[0] != 1.toByte() || version[1] != 0.toByte()) {
            // 2.0 and 3.0 differ only in header length and encoding, and nothing in this project
            // writes them. Refusing beats half-supporting: a 2.0 file read as 1.0 gives a
            // plausible-looking shape from the wrong bytes.
            throw IllegalArgumentException(
                "npy: $origin: version ${version[0]}.${version[1]}, only 1.0 is supported",
            )
        }

        val lengthBytes = readExactly(stream, 2, origin)
        val headerLen = (lengthBytes[0].toInt() and 0xff) or ((lengthBytes[1].toInt() and 0xff) shl 8)
        val header = String(readExactly(stream, headerLen, origin), Charsets.US_ASCII)

        val (dtype, itemSize) = parseDescr(field(header, "descr", origin), origin)
        if (field(header, "fortran_order", origin) != "False") {
            throw IllegalArgumentException("npy: $origin: fortran_order must be False")
        }
        val shape = parseShape(field(header, "shape", origin), origin)

        val expected = NdArray.count(shape).toLong() * itemSize
        val data = readExactly(stream, expected.toInt(), origin)
        return NdArray(data, shape, dtype, itemSize)
    }

    public fun save(path: String, array: NdArray) {
        val file = File(path)
        file.parentFile?.mkdirs()
        file.outputStream().use { write(it, array) }
    }

    public fun write(stream: OutputStream, array: NdArray) {
        val descr = descrOf(array)
        // The trailing comma on a one-element tuple is required by the format, not decoration: `(3)`
        // is the integer 3 in Python, `(3,)` is a tuple. NumPy writes the comma, and a reader that
        // expects it rejects a file written without one.
        val shape = when (array.shape.size) {
            0 -> "()"
            1 -> "(${array.shape[0]},)"
            else -> "(${array.shape.joinToString(", ")},)"
        }

        var header = "{'descr': '$descr', 'fortran_order': False, 'shape': $shape, }"

        // NumPy pads the header so magic + version + length + header is a multiple of 64, aligning the
        // payload. Readers that ignore the padding still work, but writing it keeps the files
        // byte-comparable with the reference's own output.
        val prefix = MAGIC.size + 2 + 2
        val padded = ((prefix + header.length + 1 + 63) / 64) * 64
        header = header.padEnd(padded - prefix - 1) + "\n"

        stream.write(MAGIC)
        stream.write(1)
        stream.write(0)
        stream.write(header.length and 0xff)
        stream.write((header.length shr 8) and 0xff)
        stream.write(header.toByteArray(Charsets.US_ASCII))
        stream.write(array.data)
    }

    /**
     * Reads every member of a `.npz`, which is a plain zip of `.npy` files.
     *
     * `DocTypeAngles` needs this and nothing else does: its `resources/centers.npz` carries `labels`,
     * `centers` and `max_distance`. The keys are the member names with `.npy` stripped.
     *
     * **Stored (uncompressed) and deflated members both occur**, so this goes through a real zip
     * reader rather than seeking by offset. `np.savez` writes stored; `np.savez_compressed` deflates.
     */
    public fun loadNpz(path: String): Map<String, NdArray> {
        val out = LinkedHashMap<String, NdArray>()
        ZipInputStream(File(path).inputStream()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                if (entry.isDirectory) {
                    continue
                }
                // Read the member fully first: the .npy reader needs to consume exact byte counts, and
                // a ZipInputStream reports available() unreliably for a deflated entry.
                val bytes = zip.readBytes()
                out[entry.name.removeSuffix(".npy")] = parse(bytes, "$path!${entry.name}")
            }
        }
        return out
    }

    private fun descrOf(a: NdArray): String = when (a.dtype) {
        Dtype.FLOAT32 -> "<f4"
        Dtype.FLOAT64 -> "<f8"
        Dtype.UINT8 -> "|u1"
        Dtype.INT64 -> "<i8"
        Dtype.UNICODE -> "<U${a.itemSize / 4}"
    }

    private fun parseDescr(descr: String, origin: String): Pair<Dtype, Int> = when {
        descr == "<f4" || descr == "=f4" || descr == "f4" -> Dtype.FLOAT32 to 4
        descr == "<f8" || descr == "=f8" || descr == "f8" -> Dtype.FLOAT64 to 8
        descr == "|u1" || descr == "u1" || descr == "B" -> Dtype.UINT8 to 1
        descr == "<i8" || descr == "=i8" || descr == "i8" -> Dtype.INT64 to 8
        descr.startsWith("<U") -> Dtype.UNICODE to 4 * descr.substring(2).toInt()
        // Big-endian is refused rather than byte-swapped: nothing here produces it, and a silent swap
        // would hide a genuinely wrong file.
        else -> throw IllegalArgumentException("npy: $origin: unsupported dtype '$descr'")
    }

    /** Pulls one value out of the Python-literal header by key. */
    private fun field(header: String, key: String, origin: String): String {
        val needle = "'$key':"
        val at = header.indexOf(needle)
        if (at < 0) {
            throw IllegalArgumentException("npy: $origin: header has no '$key'")
        }
        var start = at + needle.length
        while (start < header.length && header[start] == ' ') {
            start++
        }

        if (header[start] == '\'') {
            val end = header.indexOf('\'', start + 1)
            return header.substring(start + 1, end)
        }
        if (header[start] == '(') {
            val end = header.indexOf(')', start)
            return header.substring(start, end + 1)
        }
        var stop = start
        while (stop < header.length && header[stop] !in charArrayOf(',', '}', ' ')) {
            stop++
        }
        return header.substring(start, stop)
    }

    private fun parseShape(tuple: String, origin: String): IntArray {
        val inner = tuple.trim('(', ')').trim()
        if (inner.isEmpty()) {
            return IntArray(0) // '()' is a scalar: one element
        }
        return inner.split(',')
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .map {
                it.toIntOrNull()
                    ?: throw IllegalArgumentException("npy: $origin: bad shape '$tuple'")
            }
            .toIntArray()
    }

    private fun readExactly(stream: InputStream, count: Int, origin: String): ByteArray {
        val buffer = ByteArray(count)
        var read = 0
        while (read < count) {
            val n = stream.read(buffer, read, count - read)
            if (n <= 0) {
                throw EOFException("npy: $origin: truncated — wanted $count bytes, got $read")
            }
            read += n
        }
        return buffer
    }

    /** Little-endian, always: the format's `<` prefix and everything this project writes. */
    internal fun buffer(bytes: ByteArray): ByteBuffer =
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
}
