package net.russiandocs.service.api

import jakarta.servlet.http.HttpServletRequest
import java.io.File
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.service.errors.ServiceException
import net.russiandocs.service.model.Document
import net.russiandocs.service.model.DocumentStatus
import net.russiandocs.service.model.Timestamps
import net.russiandocs.service.repositories.Artifacts
import net.russiandocs.service.repositories.Documents
import net.russiandocs.service.store.DocumentQuery
import org.springframework.core.io.FileSystemResource
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity
import org.springframework.web.multipart.MultipartFile

/**
 * The document resource: upload, browse, inspect, re-run, delete.
 *
 * Serialisation is hand-written [row] / [detail] functions rather than derived from the record type, which is
 * the reference's convention and keeps the wire format visible in ONE place instead of spread across
 * annotations on a type that also has to satisfy the store.
 *
 * Written as extension functions because Kotlin has no partial classes: the split into router / documents /
 * misc is preserved, and the members these reach are `internal` on [ApiServer] — DEVIATIONS J-13.
 */

private const val MAX_FILENAME_LEN = 200

/**
 * Keeps a DISPLAY NAME only — it never touches the filesystem.
 *
 * Stored artifacts always use a fixed name, so even a hostile filename cannot escape the document directory.
 * This is purely so the UI shows something sensible and bounded; it is NOT the path-traversal defence, and
 * treating it as one would be a mistake, because the real defence is that the name is never used as a path
 * at all.
 */
internal fun safeFilename(raw: String): String {
    var name = raw.replace('\\', '/')
    val slash = name.lastIndexOf('/')
    if (slash >= 0) {
        name = name.substring(slash + 1)
    }
    name = name.trim()

    val builder = StringBuilder(name.length)
    for (c in name) {
        if (c.isISOControl() || c in "<>:\"|?*") {
            continue
        }
        builder.append(c)
    }
    var output = builder.toString()
    if (output.isEmpty()) {
        output = "upload"
    }
    // Truncated by CODE POINTS, not UTF-16 units: a name cut mid-surrogate renders as a replacement
    // character, and these names are routinely Cyrillic here.
    val points = output.codePointCount(0, output.length)
    return if (points > MAX_FILENAME_LEN) {
        output.substring(0, output.offsetByCodePoints(0, MAX_FILENAME_LEN))
    } else {
        output
    }
}

/** One line of the document log. */
internal fun ApiServer.row(record: Document): JsonObject {
    val (docBase, era) = record.docType?.let { splitDocType(it) } ?: (null to null)
    return JsonObject(linkedMapOf(
        "id" to JsonPrimitive(record.id),
        "filename" to JsonPrimitive(record.filename),
        "size_bytes" to JsonPrimitive(record.sizeBytes),
        "status" to JsonPrimitive(record.status),
        "doc_type" to str(record.docType),
        "doc_type_base" to str(docBase),
        "doc_type_era" to str(era),
        "recognised" to JsonPrimitive(record.recognised),
        "doc_conf" to num(record.docConf),
        "quality" to JsonObject(record.quality),
        "field_count" to JsonPrimitive(record.fieldCount),
        "device" to str(record.device),
        "processing_ms" to num(record.processingMs),
        "error" to str(record.error),
        "error_code" to str(record.errorCode),
        "retry_count" to JsonPrimitive(record.retryCount),
        "has_canvas" to JsonPrimitive(record.hasCanvas),
        "created_at" to str(Timestamps.format(record.createdAt)),
        "started_at" to str(Timestamps.format(record.startedAt)),
        "finished_at" to str(Timestamps.format(record.finishedAt)),
    ))
}

private fun str(value: String?): JsonElement = value?.let { JsonPrimitive(it) } ?: JsonNull
private fun num(value: Double?): JsonElement = value?.let { JsonPrimitive(it) } ?: JsonNull
private fun num(value: Int?): JsonElement = value?.let { JsonPrimitive(it) } ?: JsonNull

private fun splitDocType(label: String): Pair<String?, String?> {
    val i = label.lastIndexOf('_')
    if (i < 0) {
        return label.ifEmpty { null } to null
    }
    return label.substring(0, i).ifEmpty { null } to label.substring(i + 1).ifEmpty { null }
}

/**
 * The row plus the stored view model flattened into it.
 *
 * The stored result already has the client-facing shape — boxes, fields, canvas dimensions, coordinate-space
 * notes — so this adds URLs and the original's dimensions and otherwise passes it through. Re-deriving any
 * of it here would create a second definition of the wire format.
 */
internal fun ApiServer.detail(record: Document): JsonObject {
    val payload = LinkedHashMap(row(record).toMap())
    val result = record.result as? JsonObject ?: JsonObject(emptyMap())

    val canvas = LinkedHashMap<String, JsonElement>()
    (result["canvas"] as? JsonObject)?.forEach { (key, value) -> canvas[key] = value }
    canvas["url"] = JsonPrimitive("${ApiServer.PREFIX}/documents/${record.id}/image/canvas")

    payload["canvas"] = JsonObject(canvas)
    payload["original"] = JsonObject(linkedMapOf(
        "url" to JsonPrimitive("${ApiServer.PREFIX}/documents/${record.id}/image/original"),
        "width" to num(record.originalW),
        "height" to num(record.originalH),
        "content_type" to JsonPrimitive(record.contentType),
    ))
    payload["coord_space"] = result["coord_space"] ?: JsonNull
    payload["coord_space_note"] = result["coord_space_note"] ?: JsonNull
    payload["boxes"] = orEmptyArray(result["boxes"])
    payload["fields"] = orEmptyArray(result["fields"])
    payload["ocr"] = orEmptyObject(result["ocr"])
    payload["quality"] = orEmptyObject(result["quality"])
    payload["timings"] = orEmptyObject(result["timings"])
    payload["address"] = result["address"] ?: JsonNull
    return JsonObject(payload)
}

/**
 * Keeps a missing key from becoming a JSON null where the client expects a container.
 *
 * The SPA iterates `boxes` and `fields` unconditionally, so a null there is a runtime error in the browser
 * rather than an empty table.
 */
private fun orEmptyArray(node: JsonElement?): JsonElement = node as? JsonArray ?: JsonArray(emptyList())

private fun orEmptyObject(node: JsonElement?): JsonElement =
    node as? JsonObject ?: JsonObject(emptyMap())

/**
 * Accepts one image and queues it. **202 with the FULL LIST ROW**, so the SPA can insert the row without a
 * second request.
 *
 * Everything cheap is checked HERE, so a bad upload fails immediately with an actionable message instead of
 * becoming a mysterious failed job a minute later.
 */
internal fun ApiServer.upload(file: MultipartFile?): ResponseEntity<*> {
    if (file == null) {
        throw ServiceException.badRequest("no 'file' part in the upload")
    }

    val data = file.bytes
    if (data.size.toLong() > cfg.maxUploadBytes) {
        return jsonResponse(413, ApiErrors.detail("File exceeds the ${cfg.maxUploadMb} MB limit"))
    }
    if (data.isEmpty()) {
        throw ServiceException.badRequest("Empty upload")
    }

    if (Artifacts.isPdf(data)) {
        // Called out separately because people WILL try it, and "unsupported image type" does not tell
        // them what to do about it.
        return jsonResponse(415, ApiErrors.detail(
            "PDF is not supported — upload a JPEG, PNG, WEBP, BMP or TIFF image"))
    }
    // Sniffed from MAGIC BYTES, not the client's Content-Type, which is attacker-controlled and wrong often
    // enough to be useless.
    val sniffed = Artifacts.sniffImage(data)
        ?: return jsonResponse(415, ApiErrors.detail("Unsupported file type — expected an image"))
    val size = Artifacts.decodeDimensions(data)
        ?: throw ServiceException.unreadable("The image could not be decoded — it may be corrupt")

    val filename = safeFilename(file.originalFilename ?: "")

    // **BYTES FIRST, ROW SECOND.** The record is what makes the document visible to the worker, so writing
    // it before the file leaves a window in which the drain loop can claim a document whose original does
    // not exist yet — reporting a perfectly good upload as failed. See Documents.reserveId.
    val id = Documents.reserveId(db)
    Artifacts.saveOriginal(db, id, data, sniffed.first)

    var record = Document.new(id, filename, sniffed.second, data.size.toLong(), sniffed.first)
        .copy(
            originalW = size.first,
            originalH = size.second,
            searchText = filename.lowercase(),
        )
    record = Documents.create(db, record)

    worker.notifyNewWork()
    log.info("[API] queued document ${record.id} ($filename, ${data.size} bytes)")

    val output = LinkedHashMap(row(record).toMap())
    output["queue_position"] = num(Documents.queuePosition(db, record.id))
    return jsonResponse(202, JsonObject(output))
}

/** Serves one page of the document log. */
internal fun ApiServer.list(request: HttpServletRequest): ResponseEntity<*> {
    // The filter parameter is named `status` on the wire. Keeping that name is a client dependency, not a
    // preference.
    val statusFilter = QueryParams.str(request.getParameter("status"))
    if (statusFilter.isNotEmpty() && statusFilter !in DocumentStatus.VALID) {
        throw ServiceException.badRequest("Invalid status")
    }
    var sortDir = QueryParams.str(request.getParameter("sort_dir"))
    if (sortDir != "asc" && sortDir != "desc") {
        sortDir = "desc"
    }
    // Bounds copied from the reference's own declarations (service/api/documents.py:173-174): page is ge=1
    // with NO upper bound, page_size is ge=1 le=100. Out of range is a 422, not a clamp — see QueryParams.
    val page = QueryParams.int(request.getParameter("page"), "page", 1, 1, 0)
    val pageSize = QueryParams.int(request.getParameter("page_size"), "page_size", 20, 1, 100)

    val (rows, total) = Documents.getAll(db, DocumentQuery(
        status = statusFilter,
        docType = QueryParams.str(request.getParameter("doc_type")),
        search = QueryParams.str(request.getParameter("search")),
        dateFrom = QueryParams.str(request.getParameter("date_from")),
        dateTo = QueryParams.str(request.getParameter("date_to")),
        page = page,
        pageSize = pageSize,
        sortBy = QueryParams.str(request.getParameter("sort_by")),
        sortDir = sortDir,
    ))

    return ok(JsonObject(linkedMapOf(
        "items" to JsonArray(rows.map { row(it) }),
        "total" to JsonPrimitive(total),
        "page" to JsonPrimitive(page),
        "page_size" to JsonPrimitive(pageSize),
        "stats" to ApiServer.json.encodeToJsonElement(
            net.russiandocs.service.store.StoreStats.serializer(), Documents.stats(db)),
    )))
}

internal fun ApiServer.getDocument(id: Int): ResponseEntity<*> {
    val record = Documents.getById(db, id) ?: throw ServiceException.notFound("Document not found")
    return ok(detail(record))
}

/**
 * Live progress, a queue position, or a terminal state.
 *
 * **200 with a JSON null when there is nothing to report — never 404.** The polling client would otherwise
 * raise an error toast every two seconds for a document that finished perfectly well.
 */
internal fun ApiServer.documentProgress(id: Int): ResponseEntity<*> {
    val record = Documents.getById(db, id) ?: throw ServiceException.notFound("Document not found")

    worker.documentProgress(id)?.let { live ->
        return ok(ApiServer.json.encodeToJsonElement(
            net.russiandocs.service.worker.Progress.serializer(), live))
    }

    return when (record.status) {
        DocumentStatus.QUEUED -> {
            val position = Documents.queuePosition(db, id) ?: 0
            ok(JsonObject(linkedMapOf(
                "step" to JsonPrimitive("queued"),
                "label" to JsonPrimitive("Queued (#${position + 1})"),
                "pct" to JsonPrimitive(0),
                // The estimate is "everything ahead of me at the current average", which is honest about
                // being a guess and tracks reality because the average is an EMA of real completions.
                "eta_sec" to JsonPrimitive(round1(position * worker.averageDurationSec())),
                "queue_position" to JsonPrimitive(position),
            )))
        }

        DocumentStatus.DONE, DocumentStatus.FAILED -> ok(JsonObject(linkedMapOf(
            "step" to JsonPrimitive(record.status),
            "label" to JsonPrimitive(record.status.replaceFirstChar { it.uppercaseChar() }),
            "pct" to JsonPrimitive(if (record.status == DocumentStatus.DONE) 100 else 0),
            "eta_sec" to JsonNull,
            "queue_position" to JsonNull,
        )))

        // A JSON null body, deliberately. See the function note.
        else -> okNull()
    }
}

/**
 * Serves an artifact.
 *
 * `no-cache` means REVALIDATE, not "do not store": the response still carries Last-Modified, so a repeat
 * request costs a 304 with no body. `max-age` would be wrong here — reprocess overwrites canvas.png and
 * thumb.jpg at the SAME URL, so the browser would keep showing the previous recognition's image while the
 * field table beside it was already new.
 *
 * The header is set on the builder BEFORE the body, which is the ordering the .NET port had to fix
 * separately: there, `Results.File` commits the response, so a header assigned afterwards never reaches the
 * client — invisible locally, and visible as a stale canvas after a reprocess.
 */
internal fun ApiServer.imageArtifact(id: Int, kind: String): ResponseEntity<*> {
    if (kind != "original" && kind != "canvas" && kind != "thumb") {
        throw ServiceException.notFound("Unknown image kind")
    }
    val artifact = Artifacts.openArtifact(db, id, kind)
        ?: throw ServiceException.notFound("Image not available")
    val file = File(artifact.first)
    return ResponseEntity.ok()
        .header(HttpHeaders.CACHE_CONTROL, "private, no-cache")
        .contentType(MediaType.parseMediaType(artifact.second))
        .contentLength(file.length())
        .lastModified(file.lastModified())
        .body(FileSystemResource(file))
}

internal fun ApiServer.reprocess(id: Int): ResponseEntity<*> {
    var record = Documents.getById(db, id) ?: throw ServiceException.notFound("Document not found")
    if (record.status in Documents.ACTIVE_STATUSES) {
        throw ServiceException.conflict("Document is already ${record.status}")
    }
    record = Documents.requeue(db, record)
    worker.notifyNewWork()
    return ok(row(record))
}

/** Returns 204 and an EMPTY BODY. */
internal fun ApiServer.deleteDocument(id: Int): ResponseEntity<*> {
    val record = Documents.getById(db, id) ?: throw ServiceException.notFound("Document not found")
    Documents.delete(db, record)
    return noContent()
}

/** Clears the scratch store. SESSION ONLY — not something an integration does. */
internal fun ApiServer.purge(): ResponseEntity<*> {
    var removed = 0
    for (record in db.allRecords()) {
        // An in-flight job is left alone: deleting its record while the worker holds it produces a
        // "document vanished" failure that looks like a bug.
        if (record.status == DocumentStatus.PROCESSING) {
            continue
        }
        Documents.delete(db, record)
        removed++
    }
    log.info("[API] purged $removed document(s)")
    return ok(JsonObject(mapOf("deleted" to JsonPrimitive(removed))))
}
