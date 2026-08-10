package net.russiandocs.service.api

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.service.errors.ErrorKind
import net.russiandocs.service.errors.ServiceException
import net.russiandocs.service.logging.ServiceLog
import net.russiandocs.service.settings.SettingValidationException

/** A status code and a JSON body, ready for the framework to write. */
public data class HttpError(val status: Int, val body: JsonObject)

/**
 * The HTTP error contract.
 *
 * **This file comes first, and the order is not arbitrary.** Every constraint it fixes is inherited by
 * every other handler, and each one is a real client dependency rather than a style choice:
 *
 * - the error body is `{"detail": "<string>"}` — what FastAPI produces and what the SPA's fetch wrapper
 *   reads;
 * - a missing credential is **401**, not 403. 403 means "authenticated but not allowed", and the SPA
 *   redirects to the PIN screen on 401 only;
 * - DELETE returns **204 with an empty body**. A JSON body on a 204 is a protocol error that some clients
 *   reject outright;
 * - POST /documents returns **202 with the full list row**, so the SPA can insert the row without a second
 *   request;
 * - /progress returns **200 with a JSON `null`**, never 404 — a finished document is not a missing one, and
 *   a 404 there makes the SPA drop the row;
 * - images carry `Cache-Control: private, no-cache` and are fetched with an Authorization header, never a
 *   token in the query string, because a query token lands in logs and browser history;
 * - the list filter parameter is named `status`.
 *
 * **Spring's own error handling is turned OFF** (`server.error.whitelabel.enabled=false`, and every
 * controller path funnels through [write]). Left on, a thrown exception would produce Spring's
 * `{"timestamp","status","error","path"}` body — a second, competing error shape that no client of this
 * service parses.
 *
 * Port of the FastAPI conventions in `service/api`.
 */
public object ApiErrors {

    /** The one error shape. `detail` is a STRING, not an object. */
    public fun detail(message: String): JsonObject =
        JsonObject(mapOf("detail" to JsonPrimitive(message)))

    /**
     * Maps an exception to a status and the `detail` body.
     *
     * **The mapping lives HERE and nowhere else**, so a handler never picks a status code: that is what
     * keeps 401-versus-403 and 409-versus-400 consistent across a dozen endpoints.
     */
    public fun write(error: Throwable, log: ServiceLog): HttpError {
        // A query-parameter rejection is the ONE case whose body is not `{"detail": "<string>"}`: FastAPI
        // generates it from pydantic and `detail` is a list. See QueryParams for the captured reference
        // responses and why the inconsistency is reproduced rather than smoothed over.
        if (error is ParamException) {
            return HttpError(422, JsonObject(mapOf("detail" to JsonArray(listOf(error.item.toJson())))))
        }

        return when {
            error is ServiceException && error.kind == ErrorKind.NOT_FOUND ->
                HttpError(404, detail("Not found"))

            error is ServiceException && error.kind == ErrorKind.UNAUTHORIZED ->
                // 401, NOT 403 — see the type note.
                HttpError(401, detail("Not authenticated"))

            error is ServiceException && error.kind == ErrorKind.CONFLICT ->
                HttpError(409, detail(error.message ?: "Conflict"))

            error is SettingValidationException ->
                // **400, not 422.** FastAPI's own validation errors are 422, which is why 422 looks right
                // here — but the reference raises HTTPException(400) for a rejected SETTING, and the
                // reference is the contract. The message passes through because it names the bound that was
                // violated, which is the only useful thing a settings form can show.
                HttpError(400, detail(error.message ?: "Invalid setting"))

            error is ServiceException && error.kind == ErrorKind.BAD_REQUEST ->
                HttpError(400, detail(error.message ?: "Bad request"))

            error is ServiceException && error.kind == ErrorKind.IMAGE_UNREADABLE ->
                HttpError(422, detail(error.message ?: "Unreadable image"))

            error is ServiceException &&
                (error.kind == ErrorKind.RUNTIME_NOT_READY || error.kind == ErrorKind.PIPELINE_BUSY) ->
                HttpError(503, detail(error.message ?: "Service unavailable"))

            else -> {
                // The message is NOT echoed for an unclassified error: it may carry a path or an internal
                // detail, and the log already has it in full.
                log.error("[API] unhandled error", error)
                HttpError(500, detail("Internal server error"))
            }
        }
    }
}

/** One pydantic validation entry. */
public data class ParamErrorItem(
    val type: String,
    val loc: List<String>,
    val msg: String,
    /**
     * The RAW query string, not the parsed value — pydantic echoes what it was given, which is why an
     * out-of-range 500 comes back as the string "500".
     */
    val input: String,
    /** Omitted entirely for a parse failure, which is exactly the reference's shape. */
    val ctx: Map<String, Int>? = null,
) {
    public fun toJson(): JsonObject {
        val fields = linkedMapOf<String, kotlinx.serialization.json.JsonElement>(
            "type" to JsonPrimitive(type),
            "loc" to JsonArray(loc.map { JsonPrimitive(it) }),
            "msg" to JsonPrimitive(msg),
            "input" to JsonPrimitive(input),
        )
        ctx?.let {
            fields["ctx"] = JsonObject(it.mapValues { (_, v) -> JsonPrimitive(v) })
        }
        return JsonObject(fields)
    }
}

/** A query-parameter rejection. Thrown so handlers keep the one error path they use everywhere else. */
public class ParamException(public val item: ParamErrorItem) : RuntimeException(item.msg)

/**
 * Query-parameter validation, matching the reference byte for byte.
 *
 * **This is the one place where the reference does NOT use `{"detail": "<string>"}`.** Every hand-written
 * error in `service/api` raises HTTPException with a string, but a query parameter declared as
 * `Query(1, ge=1, le=100)` is validated by FastAPI itself, and FastAPI answers with pydantic's own shape:
 * `detail` is a LIST of objects. Reproducing that means reproducing an inconsistency in the reference —
 * done deliberately, because a client parses what the server actually sends, and the reference is the
 * contract.
 *
 * Captured from the running reference rather than written from memory:
 * ```
 * GET /documents?page_size=500
 * 422 {"detail":[{"type":"less_than_equal","loc":["query","page_size"],
 *                 "msg":"Input should be less than or equal to 100",
 *                 "input":"500","ctx":{"le":100}}]}
 * GET /documents?page_size=abc
 * 422 {"detail":[{"type":"int_parsing","loc":["query","page_size"],
 *                 "msg":"Input should be a valid integer, unable to parse string as an integer",
 *                 "input":"abc"}]}
 * ```
 * Note `ctx` is absent for a parse failure and present for a bound, and that the ORDER matters: parsing is
 * checked before bounds.
 *
 * This replaced silent clamping in the Go port. The old behaviour answered 200 with 100 rows for
 * `page_size=500`, so a client got a successful response to a request the reference rejects — invisible
 * from the server side and impossible to notice without diffing the two.
 *
 * **Spring's `@RequestParam(defaultValue=…) Int` is deliberately NOT used** for these: it would answer 400
 * with Spring's own body for an unparsable value, which is neither the reference's status nor its shape.
 * Handlers take the raw string and call [int].
 */
public object QueryParams {

    /**
     * Reads a bounded integer query parameter the way the reference declares it.
     *
     * An ABSENT parameter yields the default; an EMPTY one (`?page_size=`) is a parse failure, which is
     * what the reference does — verified, not assumed. Pass [hi] <= 0 for no upper bound, matching `page`,
     * which declares ge=1 and no le.
     */
    public fun int(raw: String?, name: String, def: Int, lo: Int, hi: Int): Int {
        if (raw == null) {
            return def
        }
        val value = raw.trim().toIntOrNull() ?: throw ParamException(ParamErrorItem(
            type = "int_parsing",
            loc = listOf("query", name),
            msg = "Input should be a valid integer, unable to parse string as an integer",
            input = raw,
        ))
        // Bounds AFTER parsing, and ge before le, so the reported error is the same one the reference
        // reports when a value violates both.
        if (value < lo) {
            throw ParamException(ParamErrorItem(
                type = "greater_than_equal",
                loc = listOf("query", name),
                msg = "Input should be greater than or equal to $lo",
                input = raw,
                ctx = mapOf("ge" to lo),
            ))
        }
        if (hi > 0 && value > hi) {
            throw ParamException(ParamErrorItem(
                type = "less_than_equal",
                loc = listOf("query", name),
                msg = "Input should be less than or equal to $hi",
                input = raw,
                ctx = mapOf("le" to hi),
            ))
        }
        return value
    }

    public fun str(raw: String?): String = raw ?: ""
}
