package net.russiandocs.service.api

import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import net.russiandocs.service.auth.Tokens
import net.russiandocs.service.config.Settings
import net.russiandocs.service.errors.ServiceException
import net.russiandocs.service.logging.ServiceLog
import net.russiandocs.service.ml.PipelineRuntime
import net.russiandocs.service.repositories.SettingsRepository
import net.russiandocs.service.store.DocumentStore
import net.russiandocs.service.worker.RecognitionWorker
import org.springframework.http.HttpHeaders
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity

/**
 * The HTTP surface, minus the handler bodies.
 *
 * Explicit constructor parameters rather than component scanning: a handler's dependencies are then visible
 * in one place, and the Go and .NET ports get the same shape without a framework. The file split matches
 * theirs (`router` / `documents` / `misc`), which is what lets the three be read side by side.
 *
 * **Spring MVC carries the requests and nothing else.** Every handler takes the raw
 * [HttpServletRequest], parses its own parameters through [QueryParams], serialises its own body with
 * kotlinx-serialization, and routes every failure through [ApiErrors]. That is not distrust of the
 * framework — it is the port rule that framework magic must not leak into logic, and here it is load-bearing
 * three separate times: `@RequestParam(defaultValue=…) Int` answers a Spring-shaped 400 where the contract
 * says a pydantic-shaped 422; Jackson's default naming policy differs from the sixty wire names the shared
 * SPA reads; and `@ResponseStatus` on an exception class would move the 401-versus-403 decision away from
 * the one file that documents it.
 */
public class ApiServer(
    internal val db: DocumentStore,
    internal val runtime: PipelineRuntime,
    internal val worker: RecognitionWorker,
    internal val cfg: Settings,
    internal val settings: SettingsRepository,
    internal val webRoot: String?,
    internal val log: ServiceLog,
) {
    public companion object {
        /**
         * The API root. Versioned, because a published REST contract that cannot change shape is a
         * published REST contract that gets replaced by a second service.
         */
        public const val PREFIX: String = "/api/v1"

        /** The wire serialiser. Nulls PRESENT: the SPA reads keys by name and treats absent as undefined. */
        internal val json: Json = Json {
            encodeDefaults = true
            explicitNulls = true
        }

        /**
         * Locates a built frontend, or `null` if there is none.
         *
         * Tries `web/dist` first and then `web/`, matching the reference: dist is the production build,
         * while the bare directory is what a developer has before running the bundler. Returning `null`
         * rather than failing is deliberate — the API is fully usable without a UI, and an integration does
         * not care that npm was never run.
         */
        public fun findWebRoot(repoRoot: String?): String? {
            if (repoRoot == null) {
                return null
            }
            for (relative in listOf("web/dist", "web")) {
                val candidate = File(repoRoot, relative)
                if (File(candidate, "index.html").isFile) {
                    return candidate.path
                }
            }
            return null
        }

        /**
         * The handful of content types the SPA build actually contains.
         *
         * Explicit rather than a provider lookup: an unknown type served as `application/octet-stream` is a
         * downloaded file instead of a rendered page, and the set of extensions Vite emits is small and
         * known.
         */
        internal fun contentTypeFor(path: String): String =
            when (path.substringAfterLast('.', "").lowercase()) {
                "html" -> "text/html; charset=utf-8"
                "js", "mjs" -> "text/javascript; charset=utf-8"
                "css" -> "text/css; charset=utf-8"
                "json", "map" -> "application/json; charset=utf-8"
                "svg" -> "image/svg+xml"
                "png" -> "image/png"
                "jpg", "jpeg" -> "image/jpeg"
                "ico" -> "image/x-icon"
                "woff2" -> "font/woff2"
                "woff" -> "font/woff"
                "ttf" -> "font/ttf"
                else -> "application/octet-stream"
            }
    }

    private val startedNanos: Long = System.nanoTime()

    internal fun uptimeSec(): Int = ((System.nanoTime() - startedNanos) / 1_000_000_000L).toInt()

    internal val authConfig: Tokens.Config
        get() = Tokens.Config(
            pin = cfg.authPin,
            jwtSecret = cfg.jwtSecret,
            jwtAlgorithm = cfg.jwtAlgorithm,
            jwtExpireMinutes = cfg.jwtExpireMinutes,
            defaultApiKey = cfg.defaultApiKey,
        )

    internal val auth: Authenticator get() = Authenticator(db, authConfig)

    // -- response helpers ---------------------------------------------------

    internal fun ok(body: JsonElement): ResponseEntity<String> = jsonResponse(200, body)

    internal fun jsonResponse(status: Int, body: JsonElement): ResponseEntity<String> =
        ResponseEntity.status(status)
            .contentType(MediaType.APPLICATION_JSON)
            .body(json.encodeToString(JsonElement.serializer(), body))

    /** 204 and an EMPTY body — a JSON body on a 204 is a protocol error some clients reject outright. */
    internal fun noContent(): ResponseEntity<String> = ResponseEntity.status(204).build()

    /** 200 with a JSON `null`, which is a real answer for `/progress` — never a 404. See [ApiErrors]. */
    internal fun okNull(): ResponseEntity<String> = jsonResponse(200, JsonNull)

    /**
     * Wraps a handler with an authentication requirement and the single error path.
     *
     * A wrapper rather than a check inside each handler: the check is then IMPOSSIBLE TO FORGET at the
     * routing table, where it is also visible — which is the property FastAPI's `Depends` provides and the
     * reason the routes read as a permission list.
     *
     * The `WWW-Authenticate` header accompanies every 401, because that is what makes the status code mean
     * "you may retry with credentials" rather than "go away".
     */
    internal fun guard(
        request: HttpServletRequest,
        response: HttpServletResponse,
        require: (HttpServletRequest) -> Identity,
        handler: (Identity) -> ResponseEntity<*>,
    ): ResponseEntity<*> {
        val identity = try {
            require(request)
        } catch (e: Throwable) {
            response.setHeader(HttpHeaders.WWW_AUTHENTICATE, "Bearer")
            val error = ApiErrors.write(e, log)
            return jsonResponse(error.status, error.body)
        }
        return try {
            handler(identity)
        } catch (e: Throwable) {
            val error = ApiErrors.write(e, log)
            jsonResponse(error.status, error.body)
        }
    }

    /**
     * Parses the `{id}` path value.
     *
     * A non-numeric id is a 404, because the route does not exist for that path — not a 400, which would
     * suggest the request could be fixed.
     */
    internal fun parseId(raw: String): Int {
        val value = raw.toIntOrNull()
        if (value == null || value < 0) {
            throw ServiceException.notFound("not a document id")
        }
        return value
    }

    internal fun round1(v: Double): Double = (v * 10 + 0.5).toInt() / 10.0

    /** Reads the whole request body as text. */
    internal fun readBody(request: HttpServletRequest): String =
        request.inputStream.readBytes().toString(Charsets.UTF_8)

    /**
     * Parses a JSON object body, or `null` for anything unparsable.
     *
     * Never throws: three endpoints read a body and each decides for itself what a missing or malformed one
     * means — a login must reject it, creating a key must not.
     */
    internal fun bodyObject(request: HttpServletRequest): JsonObject? = try {
        json.parseToJsonElement(readBody(request)) as? JsonObject
    } catch (e: Exception) {
        null
    }
}
