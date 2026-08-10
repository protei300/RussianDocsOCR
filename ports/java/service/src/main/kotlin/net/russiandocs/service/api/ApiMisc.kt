package net.russiandocs.service.api

import jakarta.servlet.http.HttpServletRequest
import java.io.File
import kotlinx.serialization.builtins.ListSerializer
import kotlinx.serialization.builtins.MapSerializer
import kotlinx.serialization.builtins.serializer
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonPrimitive
import net.russiandocs.service.auth.Tokens
import net.russiandocs.service.errors.ServiceException
import net.russiandocs.service.logging.LogEntry
import net.russiandocs.service.logging.LogRing
import net.russiandocs.service.repositories.ApiKeys
import net.russiandocs.service.repositories.Documents
import net.russiandocs.service.settings.SettingDef
import net.russiandocs.service.settings.SettingsSchema
import org.springframework.core.io.FileSystemResource
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity

/** Auth, API keys, settings, logs, status, health and the SPA. */

// --- auth ---------------------------------------------------------------

/**
 * Exchanges the PIN for a session JWT.
 *
 * The failure message deliberately does not distinguish "wrong PIN" from "malformed request": there is
 * nothing useful for a legitimate user in the difference, and there is something useful in it for somebody
 * guessing.
 */
internal fun ApiServer.pinLogin(request: HttpServletRequest): ResponseEntity<*> {
    val body = bodyObject(request)
        ?: return jsonResponse(400, ApiErrors.detail("expected {\"pin\": \"...\"}"))
    val pin = runCatching { body["pin"]?.jsonPrimitive?.content }.getOrNull() ?: ""
    if (!Tokens.verifyPin(authConfig, pin)) {
        // Logged, because repeated failures are the only signal available without rate limiting — see the
        // note in Tokens about what a PIN is and is not.
        log.warn("[API] PIN login rejected")
        return jsonResponse(401, ApiErrors.detail("Incorrect PIN"))
    }

    return ok(JsonObject(linkedMapOf(
        "access_token" to JsonPrimitive(Tokens.createAccessToken(authConfig, "operator")),
        "token_type" to JsonPrimitive("bearer"),
        "user" to JsonObject(linkedMapOf(
            "name" to JsonPrimitive(Identity.SESSION.name),
            "role" to JsonPrimitive(Identity.SESSION.role),
        )),
    )))
}

// --- api keys -----------------------------------------------------------

/**
 * Rendered verbatim by the keys page.
 *
 * Surfaced so the UI can WARN rather than letting a restart quietly delete a key somebody pasted into a
 * config somewhere. The text is copied from the reference so every implementation says the same thing.
 */
private const val EPHEMERAL_KEY_NOTE =
    "Keys created here live in ephemeral storage and are lost when the service restarts. " +
        "The default key comes from the environment and always exists."

internal fun ApiServer.listKeys(): ResponseEntity<*> = ok(JsonObject(linkedMapOf(
    "items" to JsonArray(ApiKeys.public(db, authConfig).map { anyMapToJson(it) }),
    // api-keys/Index.vue reads `res.note` and renders it as a banner. Omitting it left an empty warning div
    // on the page.
    "note" to JsonPrimitive(EPHEMERAL_KEY_NOTE),
)))

/** Mints a key and returns the PLAINTEXT exactly once. */
internal fun ApiServer.createKey(request: HttpServletRequest): ResponseEntity<*> {
    // A missing body is fine — the label is optional and defaults. Erroring here would make "create a key"
    // require a payload for no reason.
    val label = runCatching {
        bodyObject(request)?.get("label")?.jsonPrimitive?.content
    }.getOrNull() ?: ""

    val (record, plaintext) = ApiKeys.create(db, label)
    log.info("[API] api key created: id=${record.id} label=${record.label}")

    val output = LinkedHashMap(anyMapToJson(record.public()).toMap())
    // The ONLY response that ever carries the full key. After this it exists nowhere but the caller's hands
    // and a sha256 in the store.
    output["key"] = JsonPrimitive(plaintext)
    output["warning"] = JsonPrimitive("Copy this key now — it will not be shown again.")
    return jsonResponse(201, JsonObject(output))
}

/**
 * Refuses to delete the default key.
 *
 * 409, not 403: the request is well-formed and the caller is allowed to delete keys — it is the STATE that
 * forbids this one. Deleting it would also be silently undone by the next restart, since it is derived from
 * the environment rather than stored.
 */
internal fun ApiServer.deleteKey(id: Int): ResponseEntity<*> {
    if (id == ApiKeys.DEFAULT_KEY_ID) {
        throw ServiceException.conflict(
            "The default key comes from the environment and cannot be deleted")
    }
    if (!ApiKeys.delete(db, id)) {
        throw ServiceException.notFound("Key not found")
    }
    return noContent()
}

/** Converts the repositories' `Map<String, Any?>` projections into JSON without a reflective mapper. */
private fun anyMapToJson(map: Map<String, Any?>): JsonObject = JsonObject(
    map.mapValues { (_, value) ->
        when (value) {
            null -> JsonNull
            is String -> JsonPrimitive(value)
            is Boolean -> JsonPrimitive(value)
            is Int -> JsonPrimitive(value)
            is Long -> JsonPrimitive(value)
            is Double -> JsonPrimitive(value)
            else -> JsonPrimitive(value.toString())
        }
    },
)

// --- settings -----------------------------------------------------------

private val SCHEMA_JSON: JsonElement by lazy {
    ApiServer.json.encodeToJsonElement(ListSerializer(SettingDef.serializer()), SettingsSchema.ALL)
}

private fun valuesJson(values: Map<String, String>): JsonElement = ApiServer.json.encodeToJsonElement(
    MapSerializer(String.serializer(), String.serializer()), values)

internal fun ApiServer.getSettings(): ResponseEntity<*> = ok(JsonObject(linkedMapOf(
    "schema" to SCHEMA_JSON,
    "values" to valuesJson(settings.allSettings(db)),
)))

/**
 * Validates and stores, reporting which changes need a restart.
 *
 * `restart_required` is not decoration: `compute_device` and `ocr_mode` are baked into the pipeline at
 * construction, so a UI that reported "saved" and left the runtime alone would be lying about something an
 * operator can verify on the status page.
 */
internal fun ApiServer.putSettings(request: HttpServletRequest): ResponseEntity<*> {
    // **The body is WRAPPED: {"values": {...}}.** settings/Index.vue posts `Api.put('/settings', { values })`,
    // and the reference declares a SettingsUpdate model with a single `values` field. An earlier version of
    // the Go port parsed the object FLAT, which was the worst possible failure: `values` is not a schema key,
    // so the whitelist dropped it, nothing was stored, and the page reported success. Exactly the "reports
    // saved while discarding the value" outcome the settings layer is written to avoid.
    val values = bodyObject(request)?.get("values") as? JsonObject
        ?: throw ServiceException.badRequest("expected {\"values\": {...}}")

    val incoming = LinkedHashMap<String, Any?>()
    for ((key, value) in values) {
        // coerce takes the raw scalar; a JsonElement's own toString would wrap a string in quotes and turn
        // "cpu" into "\"cpu\"", which then fails the choice check for a value that was fine.
        incoming[key] = scalar(value)
    }

    val (stored, restart) = settings.bulkUpdate(db, incoming)
    if (restart.isNotEmpty()) {
        log.info("[API] settings changed, restart required: ${restart.joinToString(", ")}")
    }
    return ok(JsonObject(linkedMapOf(
        "values" to valuesJson(stored),
        // The schema travels back with the values, matching the reference, so a client that only ever calls
        // PUT still has everything it needs to render the form.
        "schema" to SCHEMA_JSON,
        // An empty ARRAY, not null: the page assigns it straight to a list it iterates.
        "restart_required" to JsonArray(restart.map { JsonPrimitive(it) }),
    )))
}

private fun scalar(value: JsonElement): Any? {
    val primitive = value as? kotlinx.serialization.json.JsonPrimitive ?: return value.toString()
    if (primitive is JsonNull) {
        return null
    }
    if (primitive.isString) {
        return primitive.content
    }
    return primitive.booleanOrNull ?: primitive.doubleOrNull ?: primitive.content
}

// --- logs ---------------------------------------------------------------

/**
 * Serves the ring buffer.
 *
 * **The response key is `entries` and the count parameter is `n`.** Both are fixed by the shared frontend:
 * logs/Index.vue sends `{ n: 400 }` and reads `res.entries`. An earlier version of the Go port returned
 * `{"items": …}` and accepted `limit`, which produced a valid response the page could not read — an EMPTY
 * logs page with a 200 and no error anywhere. The same class of mistake as the status block: when the UI is
 * shared, the UI owns the wire format.
 *
 * `count` is sent alongside because the reference sends it. The page derives its "N lines" label from the
 * array length, so nothing reads it today — but a client that trusted the documented shape would.
 */
internal fun ApiServer.logs(request: HttpServletRequest): ResponseEntity<*> {
    // Bounds copied from the reference (1..2000), not from the buffer capacity: asking for more than the
    // buffer holds is not an error, it just returns everything there is.
    val n = QueryParams.int(request.getParameter("n"), "n", 200, 1, 2000)
    val entries = LogRing.recent(
        n,
        QueryParams.str(request.getParameter("level")),
        QueryParams.str(request.getParameter("search")),
    )
    return ok(JsonObject(linkedMapOf(
        "count" to JsonPrimitive(entries.size),
        "entries" to ApiServer.json.encodeToJsonElement(
            ListSerializer(LogEntry.serializer()), entries),
    )))
}

// --- status -------------------------------------------------------------

/**
 * Reports what the service is actually doing.
 *
 * **The field names are fixed by the SHARED FRONTEND.** `web/` is reused unchanged by every port, so
 * status/Index.vue is the contract — it reads `server.cpu_pct`, `gpu.vram_used_gb`,
 * `service.data_is_ephemeral` and the rest BY NAME. An earlier version of the Go handler returned a thinner
 * block, and the status page rendered completely empty.
 *
 * `device` and `ocr_device` come through SEPARATELY on purpose: with GPU detectors the OCR engines still run
 * on CPU, and a page that just says "GPU active" invites a bug report the first time somebody watches
 * nvidia-smi during recognition.
 */
internal fun ApiServer.status(): ResponseEntity<*> {
    val stats = Documents.stats(db)
    val gpu = SysInfo.readGpu()
    return ok(JsonObject(linkedMapOf(
        "server" to ApiServer.json.encodeToJsonElement(
            ServerStats.serializer(), SysInfo.readServer()),
        // null when there is no GPU, no driver, or a CPU-only container. The status page then shows the
        // compute block alone, which is the part that answers whether the GPU is being used at all.
        "gpu" to (gpu?.let { ApiServer.json.encodeToJsonElement(GpuStats.serializer(), it) }
            ?: JsonNull),
        "compute" to ApiServer.json.encodeToJsonElement(
            net.russiandocs.service.ml.RuntimeInfo.serializer(), runtime.info()),
        "service" to JsonObject(linkedMapOf(
            "uptime_sec" to JsonPrimitive(uptimeSec()),
            "version" to JsonPrimitive(cfg.gitCommit),
            "documents_queued" to JsonPrimitive(stats.queued),
            "documents_processing" to JsonPrimitive(stats.processing),
            "documents_done" to JsonPrimitive(stats.done),
            "documents_failed" to JsonPrimitive(stats.failed),
            "documents_total" to JsonPrimitive(stats.total),
            "recognised" to JsonPrimitive(stats.recognised),
            "avg_processing_ms" to (stats.avgProcessingMs?.let { JsonPrimitive(it) } ?: JsonNull),
            "data_dir_mb" to JsonPrimitive(round1(db.diskUsageBytes() / 1e6)),
            // The SPA reads this from `service`, not from `storage`. The Python service puts it only under
            // `storage`, so its own status page always renders "Retained" — a real defect on that side,
            // recorded in the progress log rather than reproduced here.
            "data_is_ephemeral" to JsonPrimitive(db.isEphemeral),
        )),
        // Which backend is live, so an operator can tell at a glance whether what they are looking at
        // survives a restart.
        "storage" to JsonObject(linkedMapOf(
            "backend" to JsonPrimitive(db.backend),
            "ephemeral" to JsonPrimitive(db.isEphemeral),
        )),
    )))
}

/**
 * The container healthcheck. No auth, no store access, no runtime dependency.
 *
 * It reports OK while the models are still loading, deliberately: the service IS healthy then — it accepts
 * uploads and queues them. Gating health on the runtime would make Docker kill the container during the
 * fifteen seconds it needs to start.
 */
internal fun ApiServer.health(): ResponseEntity<*> = ok(JsonObject(linkedMapOf(
    "status" to JsonPrimitive("ok"),
    "runtime" to JsonPrimitive(runtime.info().state),
)))

// --- the SPA ------------------------------------------------------------

/**
 * Serves the built frontend, falling back to `index.html` for client-side routes.
 *
 * Two things here are security-relevant rather than cosmetic:
 * - the resolved path is checked to be INSIDE the web root after link resolution, so a crafted path cannot
 *   escape it. Normalising the path alone is not enough on a tree that may contain links;
 * - anything under the API prefix that reached here is a 404 in JSON, not the SPA. Serving HTML for an
 *   unknown API route makes a client's JSON parse fail with a message about '<', which is a genuinely
 *   confusing way to learn a route was misspelled.
 */
internal fun ApiServer.spa(request: HttpServletRequest): ResponseEntity<*> {
    val path = request.requestURI ?: "/"
    if (path.startsWith(ApiServer.PREFIX)) {
        return jsonResponse(404, ApiErrors.detail("Not found"))
    }
    val root = webRoot
        ?: return jsonResponse(404, ApiErrors.detail(
            "No frontend build found; run `npm run build` in web/"))

    var relative = path.trimStart('/')
    if (relative.isEmpty()) {
        relative = "index.html"
    }

    val rootFile = File(root).canonicalFile
    val candidate = File(rootFile, relative).canonicalFile
    // Outside the web root: treated as not found rather than forbidden, so a prober learns nothing about
    // the filesystem layout.
    if (candidate.path.startsWith(rootFile.path) && candidate.isFile) {
        return fileResponse(candidate, ApiServer.contentTypeFor(candidate.path), null)
    }

    // A client-side route: hand back index.html and let the SPA router resolve it.
    val index = File(rootFile, "index.html")
    if (index.isFile) {
        // no-cache on the shell only: the hashed asset files under /assets are immutable and get the
        // server's default caching, but a cached index.html pins the client to an old bundle after a deploy.
        return fileResponse(index, "text/html; charset=utf-8", "no-cache")
    }
    return jsonResponse(404, ApiErrors.detail("Not found"))
}

private fun fileResponse(file: File, contentType: String, cacheControl: String?): ResponseEntity<*> {
    var builder = ResponseEntity.ok()
        .contentType(MediaType.parseMediaType(contentType))
        .contentLength(file.length())
    if (cacheControl != null) {
        builder = builder.header(HttpHeaders.CACHE_CONTROL, cacheControl)
    }
    return builder.body(FileSystemResource(file))
}
