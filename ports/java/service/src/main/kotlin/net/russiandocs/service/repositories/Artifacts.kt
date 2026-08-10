package net.russiandocs.service.repositories

import java.io.File
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Interpolation
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.service.store.DocumentStore
import net.russiandocs.service.store.FileStore

/**
 * Binary artifacts: the uploaded original, the rendered canvas, the thumbnail.
 *
 * **THIS LAYER STAYS ON THE FILESYSTEM EVEN AFTER A SQL MIGRATION.** Multi-megabyte PNGs do not belong in
 * a database — in a real deployment this file grows an S3 implementation, not a BLOB column. That is why
 * it is separate from [Documents] rather than folded into it.
 */
public object Artifacts {

    /**
     * The formats accepted, keyed by the bytes that actually identify them.
     *
     * SNIFFED rather than trusting the client's Content-Type, which is attacker-controlled and routinely
     * wrong even when it is not.
     */
    private val MAGIC: List<Triple<ByteArray, String, String>> = listOf(
        Triple(byteArrayOf(0xff.toByte(), 0xd8.toByte(), 0xff.toByte()), ".jpg", "image/jpeg"),
        Triple(
            byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(),
                0x0d, 0x0a, 0x1a, 0x0a),
            ".png", "image/png",
        ),
        Triple("BM".toByteArray(Charsets.ISO_8859_1), ".bmp", "image/bmp"),
        Triple(byteArrayOf('I'.code.toByte(), 'I'.code.toByte(), '*'.code.toByte(), 0), ".tif",
            "image/tiff"),
        Triple(byteArrayOf('M'.code.toByte(), 'M'.code.toByte(), 0, '*'.code.toByte()), ".tif",
            "image/tiff"),
    )

    /**
     * The extension and media type for a supported image, or `null`.
     *
     * WEBP needs a two-part check — 'RIFF' at 0 and 'WEBP' at 8 — which is why it is not in the table.
     */
    public fun sniffImage(data: ByteArray, length: Int = data.size): Pair<String, String>? {
        for ((prefix, ext, media) in MAGIC) {
            if (startsWith(data, length, prefix)) {
                return ext to media
            }
        }
        if (length >= 12 &&
            startsWith(data, length, "RIFF".toByteArray(Charsets.ISO_8859_1)) &&
            regionEquals(data, 8, "WEBP".toByteArray(Charsets.ISO_8859_1))
        ) {
            return ".webp" to "image/webp"
        }
        return null
    }

    /** Detected separately so the error can say WHY. Users will try PDFs. */
    public fun isPdf(data: ByteArray, length: Int = data.size): Boolean =
        startsWith(data, length, "%PDF".toByteArray(Charsets.ISO_8859_1))

    private fun startsWith(data: ByteArray, length: Int, prefix: ByteArray): Boolean =
        length >= prefix.size && regionEquals(data, 0, prefix)

    private fun regionEquals(data: ByteArray, offset: Int, want: ByteArray): Boolean {
        if (offset + want.size > data.size) {
            return false
        }
        for (i in want.indices) {
            if (data[offset + i] != want[i]) {
                return false
            }
        }
        return true
    }

    /** The artifact directory, created. */
    public fun docDir(db: DocumentStore, id: Int): String {
        val dir = db.docDir(id)
        File(dir).mkdirs()
        return dir
    }

    /**
     * Stores the upload byte-for-byte under a FIXED name.
     *
     * The client's filename is kept on the record for display only and never touches the filesystem — so
     * it cannot be a path-traversal vector no matter what it contains.
     */
    public fun saveOriginal(db: DocumentStore, id: Int, data: ByteArray, ext: String): String {
        val path = File(docDir(db, id), "original$ext")
        FileStore.atomicWriteBytes(path, data)
        return path.path
    }

    /**
     * The upload's width and height, or `null` if it cannot be decoded.
     *
     * Done SYNCHRONOUSLY at upload time so an undecodable file becomes an immediate, actionable 422
     * instead of a mysterious failed job minutes later.
     *
     * [Io.decodeSize] rather than a full decode: the colour conversion a full decode owes the pipeline is a
     * second pass over the image, and nothing here reads a pixel. In the Go port that was measurable, not
     * theoretical — ~72 ms per upload against ~22 ms.
     */
    public fun decodeDimensions(data: ByteArray): Pair<Int, Int>? = Io.decodeSize(data)

    /**
     * Writes the corrected canvas as PNG and returns its dimensions.
     *
     * The canvas is RGB and the encoder expects BGR. Skipping the conversion swaps red and blue in every
     * displayed document — and the result looks plausible enough on a passport that it can ship unnoticed.
     * Hence the explicit conversion inside [Io.writePngFromRgb] and the regression test asserting a
     * known-red pixel stays red.
     */
    public fun saveCanvas(db: DocumentStore, id: Int, rgb: Image): Triple<String, Int, Int> {
        val path = File(docDir(db, id), "canvas.png")
        Io.writePngFromRgb(path.path, rgb)
        return Triple(path.path, rgb.width, rgb.height)
    }

    /**
     * Writes a small JPEG for the list page.
     *
     * Without it the log page pulls full canvases for every visible row on each three-second poll —
     * megabytes per refresh for images rendered at 56 px wide.
     */
    public fun saveThumbnail(db: DocumentStore, id: Int, rgb: Image, width: Int): String {
        val dir = docDir(db, id)
        val w = if (width <= 0) 96 else width
        val h = maxOf(1, (rgb.height * w + rgb.width / 2) / rgb.width)
        Io.resize(rgb, w, h, Interpolation.AREA).use { small ->
            val path = File(dir, "thumb.jpg")
            Io.writeJpegFromRgb(path.path, small, 80)
            return path.path
        }
    }

    /** The path and media type for "original", "canvas" or "thumb". */
    public fun openArtifact(db: DocumentStore, id: Int, kind: String): Pair<String, String>? {
        val dir = File(db.docDir(id))
        return when (kind) {
            "canvas" -> {
                // PNG for anything this service rendered; JPEG for the pre-computed seed fixtures, which
                // trade exactness for a committable repository footprint.
                for ((name, media) in listOf(
                    "canvas.png" to "image/png", "canvas.jpg" to "image/jpeg")) {
                    val candidate = File(dir, name)
                    if (candidate.isFile) {
                        return candidate.path to media
                    }
                }
                null
            }

            "thumb" -> {
                val thumb = File(dir, "thumb.jpg")
                if (thumb.isFile) {
                    thumb.path to "image/jpeg"
                } else {
                    // Falls back to the full canvas rather than 404ing: a missing thumbnail is a
                    // performance problem, not a missing document.
                    openArtifact(db, id, "canvas")
                }
            }

            "original" -> {
                val matches = dir.listFiles { f -> f.name.startsWith("original.") }
                    ?.sortedBy { it.name } ?: return null
                for (candidate in matches) {
                    if (candidate.name.endsWith(".tmp")) {
                        continue
                    }
                    val head = ByteArray(16)
                    val read = try {
                        candidate.inputStream().use { it.read(head, 0, head.size) }
                    } catch (e: java.io.IOException) {
                        continue
                    }
                    val sniffed = sniffImage(head, maxOf(read, 0))
                    return candidate.path to (sniffed?.second ?: "application/octet-stream")
                }
                null
            }

            else -> null
        }
    }
}
