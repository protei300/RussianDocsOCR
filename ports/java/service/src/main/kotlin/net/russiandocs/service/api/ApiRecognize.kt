package net.russiandocs.service.api

import jakarta.servlet.http.HttpServletRequest
import java.net.URLDecoder
import java.util.Arrays
import java.util.Base64
import java.util.concurrent.ExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.docproc.imaging.Image
import net.russiandocs.docproc.imaging.Io
import net.russiandocs.docproc.viewmodel.Payload
import net.russiandocs.service.errors.ErrorKind
import net.russiandocs.service.errors.ServiceException
import net.russiandocs.service.ml.InMemoryResult
import net.russiandocs.service.repositories.Artifacts
import net.russiandocs.service.settings.SettingsSchema
import net.russiandocs.service.worker.SearchText
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity

/**
 * Recognition in memory: one image in, one answer out, nothing kept. `POST /recognize`, `GET /recognitions`.
 *
 * The body is read, decoded, recognised inside the request, answered, and every buffer this code owns is
 * zeroed. No record, no file, no artifact, no queue; one log line and one journal line with the outcome and the
 * time — nothing about the document. Available in both processing modes; in `PROCESSING_MODE=memory` it is the
 * only way a document gets in. Port of `service/api/recognize.py`; the contract is ports/MEMORY.md §3–§5.
 *
 * What "nothing kept" takes on Spring and Tomcat, because each item was a real way to keep something:
 *
 * - **The body never goes through the multipart machinery.** `MultipartFile`, `@RequestPart` and the multipart
 *   resolver hand the request to Tomcat's part parser, which writes every part past `file-size-threshold` (0 by
 *   default in Spring Boot — i.e. EVERY part) to a temporary file. This handler reads
 *   `HttpServletRequest.getInputStream()` into one array it owns, and splits a multipart body in memory;
 *   `UploadOnlyMultipartResolver` makes sure the resolver never parses anything but the stored upload, and
 *   `request.getParameter` — which on Tomcat ALSO parses a multipart body — is never called here (the query
 *   string is parsed by hand).
 * - **Buffers are zeroed** with `Arrays.fill`, native Mats with `setTo(0)` before `release()`: the body, the
 *   image part, the decoder's input and BGR pixels (`Io.decodeRgbWiping`), the RGB pixels, the PNG. The pipeline's own images are zeroed
 *   inside the lease (`InMemory.recogniseOn`).
 * - **Strings cannot be wiped** on the JVM: the recognised text and the JSON answer are released, not
 *   overwritten. The README says so rather than promising more.
 * - **Nothing about the document reaches a log**: no filename, type, size, field, library line or exception
 *   message — the type NAME of an unexpected exception only.
 * - **The answer is not cacheable**: `Cache-Control: no-store`.
 */

/** Multipart framing on top of the image limit — boundaries and part headers are a few hundred bytes. */
private const val FRAMING_BYTES = 64 * 1024

/** The 404 body of every document route in memory mode — ports/MEMORY.md §2, word for word. */
internal const val MEMORY_MODE_DETAIL: String =
    "Document storage is disabled (PROCESSING_MODE=memory): documents are not stored and cannot be listed or " +
        "opened. Recognise with POST /api/v1/recognize."

/** A refusal with its status, detail, journal outcome and headers. No stack trace: it is control flow. */
private class Refused(
    val status: Int,
    val detail: String,
    val outcome: String = "rejected",
    val headers: Map<String, String> = emptyMap(),
) : RuntimeException(detail, null, false, false)

private fun wipe(buffer: ByteArray?) {
    if (buffer != null) {
        Arrays.fill(buffer, 0.toByte())
    }
}

private fun ApiServer.observe(buffer: ByteArray): ByteArray {
    bufferObserver?.invoke(buffer)
    return buffer
}

/**
 * The whole body into ONE array this code owns, refusing more than [cap] bytes.
 *
 * With a Content-Length the array is allocated once at that size. Without one (chunked) it grows by doubling,
 * and every outgrown array is zeroed before it is dropped — `ByteArrayOutputStream` would leave each of them
 * holding a prefix of the image.
 */
private fun ApiServer.readCappedBody(request: HttpServletRequest, cap: Long): ByteArray {
    val declared = request.contentLengthLong
    if (declared > cap) {
        throw Refused(413, "Image exceeds the upload limit")
    }
    val input = request.inputStream
    var buffer = observe(ByteArray(if (declared >= 0) declared.toInt() else 64 * 1024))
    var n = 0
    while (true) {
        if (n == buffer.size) {
            // Tomcat never delivers more than the declared length.
            if (declared >= 0 && n.toLong() == declared) {
                break
            }
            if (n.toLong() > cap) {
                wipe(buffer)
                throw Refused(413, "Image exceeds the upload limit")
            }
            val grown = observe(ByteArray(minOf(maxOf(n.toLong() * 2, 64L * 1024), cap + 1).toInt()))
            System.arraycopy(buffer, 0, grown, 0, n)
            wipe(buffer)
            buffer = grown
        }
        val read = input.read(buffer, n, buffer.size - n)
        if (read < 0) {
            break
        }
        n += read
    }
    if (n.toLong() > cap) {
        wipe(buffer)
        throw Refused(413, "Image exceeds the upload limit")
    }
    if (n == buffer.size) {
        return buffer
    }
    val exact = observe(buffer.copyOf(n))
    wipe(buffer)
    return exact
}

/** The `boundary` parameter of a multipart Content-Type, as bytes; null when there is none. */
private fun boundaryOf(contentType: String): ByteArray? {
    for (part in contentType.split(';').drop(1)) {
        val eq = part.indexOf('=')
        if (eq < 0) {
            continue
        }
        if (part.substring(0, eq).trim().equals("boundary", ignoreCase = true)) {
            val value = part.substring(eq + 1).trim().trim('"')
            if (value.isNotEmpty()) {
                return value.toByteArray(Charsets.ISO_8859_1)
            }
        }
    }
    return null
}

private fun indexOf(haystack: ByteArray, needle: ByteArray, from: Int): Int {
    if (needle.isEmpty()) {
        return from
    }
    var i = maxOf(0, from)
    val last = haystack.size - needle.size
    while (i <= last) {
        var j = 0
        while (j < needle.size && haystack[i + j] == needle[j]) {
            j++
        }
        if (j == needle.size) {
            return i
        }
        i++
    }
    return -1
}

/** Case-insensitive ASCII search in `haystack[from, to)` — on the bytes, so no header text becomes a String. */
private fun containsIgnoreCase(haystack: ByteArray, from: Int, to: Int, needle: String): Boolean {
    val n = needle.lowercase().toByteArray(Charsets.ISO_8859_1)
    var i = from
    while (i + n.size <= to) {
        var j = 0
        while (j < n.size) {
            var b = haystack[i + j].toInt()
            if (b in 'A'.code..'Z'.code) {
                b += 32
            }
            if (b != n[j].toInt()) {
                break
            }
            j++
        }
        if (j == n.size) {
            return true
        }
        i++
    }
    return false
}

private val CRLF = "\r\n".toByteArray(Charsets.ISO_8859_1)
private val CRLF_CRLF = "\r\n\r\n".toByteArray(Charsets.ISO_8859_1)

/**
 * The first file part of a multipart body — a part with a `filename`, or a part named `file` — copied into its
 * own array. The part headers are inspected as bytes and never decoded: **the filename is never read into
 * anything that outlives the parse**. The caller zeroes the body afterwards; the copy is zeroed later too.
 */
private fun imageFromMultipart(body: ByteArray, boundary: ByteArray): ByteArray {
    val delimiter = "--".toByteArray(Charsets.ISO_8859_1) + boundary
    val closing = CRLF + delimiter
    var pos = indexOf(body, delimiter, 0)
    while (pos != -1) {
        val start = pos + delimiter.size
        if (start + 2 <= body.size && body[start] == '-'.code.toByte() && body[start + 1] == '-'.code.toByte()) {
            break
        }
        val headerEnd = indexOf(body, CRLF_CRLF, start)
        if (headerEnd == -1) {
            break
        }
        val dataStart = headerEnd + 4
        val next = indexOf(body, closing, dataStart)
        if (next == -1) {
            break
        }
        if (containsIgnoreCase(body, start, headerEnd, "filename=") ||
            containsIgnoreCase(body, start, headerEnd, "name=\"file\"")
        ) {
            return body.copyOfRange(dataStart, next)
        }
        pos = next + 2
    }
    throw Refused(400, "No image in the multipart body — send it as the 'file' part")
}

/**
 * `?images=` parsed from the raw query string — NOT through `request.getParameter`, which on Tomcat parses a
 * multipart body (to disk) to find form fields. Pydantic's boolean spellings; anything else is its 422.
 */
private fun imagesFlag(queryString: String?): Boolean {
    var raw: String? = null
    for (pair in (queryString ?: "").split('&')) {
        val eq = pair.indexOf('=')
        val key = URLDecoder.decode(if (eq < 0) pair else pair.substring(0, eq), Charsets.UTF_8)
        if (key == "images") {
            raw = if (eq < 0) "" else URLDecoder.decode(pair.substring(eq + 1), Charsets.UTF_8)
        }
    }
    val value = raw ?: return false
    return when (value.trim().lowercase()) {
        "1", "true", "t", "yes", "y", "on" -> true
        "0", "false", "f", "no", "n", "off" -> false
        else -> throw ParamException(ParamErrorItem(
            type = "bool_parsing",
            loc = listOf("query", "images"),
            msg = "Input should be a valid boolean, unable to interpret input",
            input = value,
        ))
    }
}

/**
 * Decode, recognise, wipe. Runs on the recognition executor; every pixel buffer it creates is zeroed here, in
 * `finally`, even when recognition fails — and so is [image], the encoded bytes.
 */
private fun ApiServer.recogniseSync(
    image: ByteArray,
    docconf: Double,
    imgSize: Int,
    withCanvas: Boolean,
    leaseTimeoutMs: Long,
): InMemoryResult {
    var rgb: Image? = null
    try {
        rgb = Io.decodeRgbWiping(image)
            ?: throw Refused(422, "The image could not be decoded — it may be corrupt")
        return memoryRecogniser.recognise(rgb, docconf, imgSize, withCanvas, leaseTimeoutMs)
    } finally {
        rgb?.zeroAndClose()
        wipe(image)
    }
}

private fun arrayOrEmpty(node: JsonElement?): JsonElement = node as? JsonArray ?: JsonArray(emptyList())
private fun objectOrEmpty(node: JsonElement?): JsonObject = node as? JsonObject ?: JsonObject(emptyMap())

/**
 * The answer: the stored document's detail without anything that points at a stored thing. Built from the same
 * serialised view model the worker stores ([SearchText.toElement]), so the JSON is identical to a stored
 * document's — DocConf lifted out of `quality` into `doc_conf`, exactly as `Documents.saveResult` does.
 */
private fun responseBody(payload: Payload, png: ByteArray?, ms: Int): JsonObject {
    val tree = SearchText.toElement(payload) as JsonObject
    val canvas = LinkedHashMap<String, JsonElement>(objectOrEmpty(tree["canvas"]))
    if (png != null) {
        canvas["image_png_base64"] = JsonPrimitive(Base64.getEncoder().encodeToString(png))
    }
    val quality = objectOrEmpty(tree["quality"])
    return JsonObject(linkedMapOf(
        "stored" to JsonPrimitive(false),
        "doc_type" to (tree["doc_type"] ?: JsonNull),
        "doc_type_base" to (tree["doc_type_base"] ?: JsonNull),
        "doc_type_era" to (tree["doc_type_era"] ?: JsonNull),
        "recognised" to (tree["recognised"] ?: JsonNull),
        "doc_conf" to (quality["DocConf"] ?: JsonNull),
        "device" to (tree["device"] ?: JsonNull),
        "canvas" to JsonObject(canvas),
        "coord_space" to (tree["coord_space"] ?: JsonNull),
        "coord_space_note" to (tree["coord_space_note"] ?: JsonNull),
        "boxes" to arrayOrEmpty(tree["boxes"]),
        "fields" to arrayOrEmpty(tree["fields"]),
        "ocr" to objectOrEmpty(tree["ocr"]),
        "quality" to JsonObject(quality.filterKeys { it != "DocConf" }),
        "address" to (tree["address"] ?: JsonNull),
        "timings" to objectOrEmpty(tree["timings"]),
        "processing_ms" to JsonPrimitive(ms),
    ))
}

private fun elapsedMs(startedNanos: Long): Int = ((System.nanoTime() - startedNanos) / 1_000_000).toInt()

/** The journal line and THE log line of an in-memory recognition: outcome, status, time. Nothing else. */
private fun ApiServer.finish(outcome: String, status: Int, startedNanos: Long) {
    val ms = elapsedMs(startedNanos)
    recognitions.record(outcome, status, ms)
    log.info("[API] recognize: outcome=$outcome status=$status ms=$ms")
}

/** `POST /recognize`. Called inside the guard, so an unauthenticated call is neither journalled nor logged. */
internal fun ApiServer.recognize(request: HttpServletRequest): ResponseEntity<*> {
    val started = System.nanoTime()
    // Before the journalled part, like FastAPI's own query validation: a malformed flag is a 422 about the URL.
    val withCanvas = imagesFlag(request.queryString)
    val limit = cfg.maxUploadBytes
    var body: ByteArray? = null
    var image: ByteArray? = null
    var png: ByteArray? = null
    try {
        body = readCappedBody(request, limit + FRAMING_BYTES)
        val contentType = request.contentType ?: ""
        if (contentType.lowercase().startsWith("multipart/form-data")) {
            val boundary = boundaryOf(contentType) ?: throw Refused(400, "Malformed multipart body")
            image = observe(imageFromMultipart(body, boundary))
            wipe(body)
        } else {
            image = body
            body = null
        }
        if (image.isEmpty()) {
            throw Refused(400, "Empty upload")
        }
        if (image.size.toLong() > limit) {
            throw Refused(413, "File exceeds the ${cfg.maxUploadMb} MB limit")
        }
        if (Artifacts.isPdf(image)) {
            throw Refused(415, "PDF is not supported — send a JPEG, PNG, WEBP, BMP or TIFF image")
        }
        if (Artifacts.sniffImage(image) == null) {
            throw Refused(415, "Unsupported file type — expected an image")
        }

        val docconf = SettingsSchema.typedFloat("docconf", settings.settingValue(db, "docconf"), cfg.docconf)
        val imgSize = SettingsSchema.typedInt("img_size", settings.settingValue(db, "img_size"), cfg.imgSize)
        val jobTimeoutSec = SettingsSchema.typedInt(
            "job_timeout_sec", settings.settingValue(db, "job_timeout_sec"), cfg.jobTimeoutSec)
        val waitSec = cfg.recognizeWaitSec

        val bytes = image
        val work = recognizeExecutor.submit<InMemoryResult> {
            recogniseSync(bytes, docconf, imgSize, withCanvas, waitSec * 1000L)
        }
        val result = try {
            work.get(jobTimeoutSec.toLong() + waitSec, TimeUnit.SECONDS)
        } catch (e: TimeoutException) {
            // The work cannot be cancelled (native code ignores interrupts); it finishes on its own and wipes
            // its pixels itself. What it would hand back — the PNG — is zeroed by this follower when it does.
            recognizeExecutor.submit {
                runCatching { work.get() }.getOrNull()?.png?.let { wipe(it) }
            }
            throw Refused(504, "Recognition took too long", "error")
        } catch (e: ExecutionException) {
            throw e.cause ?: e
        }
        image = null // wiped by recogniseSync
        png = result.png?.let { observe(it) }

        val ms = elapsedMs(started)
        val content = ApiServer.json.encodeToString(JsonElement.serializer(), responseBody(result.payload, png, ms))
        finish(if (result.payload.recognised) "recognised" else "not_recognised", 200, started)
        return ResponseEntity.ok()
            .header(HttpHeaders.CACHE_CONTROL, "no-store")
            .contentType(MediaType.APPLICATION_JSON)
            .body(content)
    } catch (e: Throwable) {
        val refused = when {
            e is Refused -> e
            e is ServiceException && e.kind == ErrorKind.PIPELINE_BUSY ->
                Refused(503, "All recognition pipelines are busy — retry shortly", "busy", mapOf("Retry-After" to "5"))
            e is ServiceException && e.kind == ErrorKind.RUNTIME_NOT_READY ->
                Refused(503, "The recognition runtime is not ready yet", "busy", mapOf("Retry-After" to "10"))
            else -> null
        }
        if (refused != null) {
            finish(refused.outcome, refused.status, started)
            val builder = ResponseEntity.status(refused.status).contentType(MediaType.APPLICATION_JSON)
            refused.headers.forEach { (name, value) -> builder.header(name, value) }
            return builder.body(ApiServer.json.encodeToString(JsonElement.serializer(),
                ApiErrors.detail(refused.detail)))
        }
        finish("error", 500, started)
        // The TYPE only: a message carries paths and values.
        log.warn("[API] recognize failed: ${e.javaClass.simpleName}")
        return jsonResponse(500, ApiErrors.detail("Recognition failed"))
    } finally {
        wipe(body)
        wipe(image)
        wipe(png)
    }
}

/**
 * `GET /recognitions`: the journal — when, how it ended, how long. Nothing else exists to show; the lines cannot
 * be opened.
 */
internal fun ApiServer.listRecognitions(request: HttpServletRequest): ResponseEntity<*> {
    val limit = QueryParams.int(request.getParameter("limit"), "limit", 100, 1, 1000)
    val offset = QueryParams.int(request.getParameter("offset"), "offset", 0, 0, 0)
    val (page, total) = recognitions.recent(limit, offset)
    return ok(JsonObject(linkedMapOf(
        "items" to JsonArray(page.map { it.public() }),
        "total" to JsonPrimitive(total),
        "counts" to JsonObject(recognitions.counts().mapValues { (_, v) -> JsonPrimitive(v) }),
    )))
}
