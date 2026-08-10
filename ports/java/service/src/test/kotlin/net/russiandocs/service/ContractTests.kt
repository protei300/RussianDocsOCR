package net.russiandocs.service

import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import net.russiandocs.service.api.ApiRoutes
import net.russiandocs.service.api.ApiServer
import net.russiandocs.service.config.Settings
import net.russiandocs.service.logging.ServiceLog
import net.russiandocs.service.ml.PipelineRuntime
import net.russiandocs.service.repositories.SettingsRepository
import net.russiandocs.service.store.FileStore
import net.russiandocs.service.worker.RecognitionWorker
import org.springframework.boot.WebApplicationType
import org.springframework.boot.builder.SpringApplicationBuilder
import org.springframework.context.ConfigurableApplicationContext

/**
 * The wire contract, driven over real HTTP.
 *
 * Not unit tests: they start Tomcat on port 0 and speak to it the way the SPA and an integration do,
 * because **every constraint here is one a client depends on** and none of them is visible from inside a
 * handler — the `{"detail": …}` shape, 401-not-403, 204-with-an-empty-body, the pydantic-shaped 422 for a
 * query parameter, the SPA fallback serving hashed assets.
 *
 * **The pipeline runtime is constructed but never initialised**, deliberately: nothing here needs
 * recognition, and loading 215 MB of models would make the suite slow enough that nobody runs it. Every
 * endpoint under test answers the same whether models are loaded or not, and the ONE that does not —
 * recognition itself — is verified against `service/seed_data` over HTTP instead, which is a stronger
 * check than any mock.
 */
class ContractTests {

    private lateinit var context: ConfigurableApplicationContext
    private lateinit var dataDir: java.io.File
    private lateinit var webRoot: java.io.File
    private var port = 0

    private val client: HttpClient = HttpClient.newBuilder()
        // **No proxy.** The .NET port lost 23 tests to this: the default client honoured the machine's proxy
        // settings, sent loopback requests to Squid, and every assertion failed with "'<' is an invalid
        // start of a value" — an HTML error page parsed as JSON.
        .proxy(java.net.ProxySelector.of(null))
        .build()

    private val json = Json { ignoreUnknownKeys = true }

    private val pin = "4321"
    private val apiKey = "rdk_contract_test_key"

    @BeforeTest
    fun start() {
        dataDir = Files.createTempDirectory("rdocs-contract").toFile()
        webRoot = java.io.File(dataDir, "webroot")
        java.io.File(webRoot, "assets").mkdirs()
        java.io.File(webRoot, "index.html")
            .writeText("<!doctype html><script src=\"/assets/index-abc123.js\"></script>")
        java.io.File(webRoot, "assets/index-abc123.js").writeText("export default 1")

        val cfg = Settings(
            authPin = pin,
            jwtSecret = "test-secret",
            defaultApiKey = apiKey,
            dataDir = dataDir.path,
            dataWipeOnStart = false,
            seedSamples = -1,
        )
        val log = ServiceLog("test")
        val db = FileStore(dataDir.path, log.sink())
        val runtime = PipelineRuntime(log.sink())
        val settings = SettingsRepository(cfg, log.sink())
        val worker = RecognitionWorker(db, runtime, cfg, settings, log)
        val api = ApiServer(db, runtime, worker, cfg, settings, webRoot.path, log)

        context = SpringApplicationBuilder(Application::class.java)
            .web(WebApplicationType.SERVLET)
            .bannerMode(org.springframework.boot.Banner.Mode.OFF)
            .properties(mapOf("server.port" to 0))
            .initializers({ ctx -> ctx.beanFactory.registerSingleton("apiServer", api) })
            .run()
        // `server.port=0` binds an ephemeral port; Boot publishes the real one as `local.server.port`.
        port = requireNotNull(context.environment.getProperty("local.server.port")?.toInt()) {
            "the embedded container did not publish local.server.port"
        }
    }

    @AfterTest
    fun stop() {
        context.close()
        dataDir.deleteRecursively()
    }

    // -- helpers ------------------------------------------------------------

    private fun send(
        method: String,
        path: String,
        key: String? = apiKey,
        token: String? = null,
        body: String? = null,
    ): Pair<Int, String> {
        var builder = HttpRequest.newBuilder(URI.create("http://127.0.0.1:$port$path"))
        key?.let { builder = builder.header("X-API-Key", it) }
        token?.let { builder = builder.header("Authorization", "Bearer $it") }
        builder = if (body == null) {
            builder.method(method, HttpRequest.BodyPublishers.noBody())
        } else {
            builder.header("Content-Type", "application/json")
                .method(method, HttpRequest.BodyPublishers.ofString(body))
        }
        val response = client.send(builder.build(), HttpResponse.BodyHandlers.ofString())
        return response.statusCode() to response.body()
    }

    private fun login(): String {
        val (status, body) = send("POST", "/api/v1/auth/pin-login", key = null,
            body = """{"pin":"$pin"}""")
        assertEquals(200, status, body)
        return json.parseToJsonElement(body).let {
            (it as JsonObject)["access_token"]!!.jsonPrimitive.content
        }
    }

    private fun obj(body: String): JsonObject = json.parseToJsonElement(body) as JsonObject

    // -- auth ---------------------------------------------------------------

    @Test
    fun `no credential is 401 with a string detail`() {
        val (status, body) = send("GET", "/api/v1/documents", key = null)
        assertEquals(401, status)
        // A STRING, not an object: FastAPI's HTTPException produces exactly this and the SPA reads it.
        assertTrue(obj(body)["detail"]!!.jsonPrimitive.isString, body)
    }

    @Test
    fun `a wrong PIN is 401 and a right one mints a token`() {
        val (bad, _) = send("POST", "/api/v1/auth/pin-login", key = null, body = """{"pin":"0000"}""")
        assertEquals(401, bad)
        assertTrue(login().isNotEmpty())
    }

    @Test
    fun `a malformed login body is 400, not 500`() {
        val (status, _) = send("POST", "/api/v1/auth/pin-login", key = null, body = "not json")
        assertEquals(400, status)
    }

    @Test
    fun `an api key is refused on the operator surface`() {
        // 401 rather than 403: the caller may retry with the right KIND of credential.
        val (status, _) = send("GET", "/api/v1/settings")
        assertEquals(401, status)
        val (ok, _) = send("GET", "/api/v1/settings", key = null, token = login())
        assertEquals(200, ok)
    }

    @Test
    fun `a lowercase bearer scheme is accepted`() {
        val token = login()
        val response = client.send(
            HttpRequest.newBuilder(URI.create("http://127.0.0.1:$port/api/v1/settings"))
                .header("Authorization", "bearer $token").GET().build(),
            HttpResponse.BodyHandlers.ofString(),
        )
        assertEquals(200, response.statusCode())
    }

    // -- query parameters ---------------------------------------------------

    @Test
    fun `an out-of-range page_size is a pydantic-shaped 422`() {
        val (status, body) = send("GET", "/api/v1/documents?page_size=500")
        assertEquals(422, status)
        val detail = obj(body)["detail"] as JsonArray
        val first = detail[0] as JsonObject
        assertEquals("less_than_equal", first["type"]!!.jsonPrimitive.content)
        // The RAW string, as pydantic echoes it.
        assertEquals("500", first["input"]!!.jsonPrimitive.content)
        assertTrue(first.containsKey("ctx"), body)
    }

    @Test
    fun `an unparsable page_size is 422 WITHOUT ctx`() {
        val (status, body) = send("GET", "/api/v1/documents?page_size=abc")
        assertEquals(422, status)
        val first = (obj(body)["detail"] as JsonArray)[0] as JsonObject
        assertEquals("int_parsing", first["type"]!!.jsonPrimitive.content)
        // Absent for a parse failure and present for a bound — the reference's exact shape.
        assertFalse(first.containsKey("ctx"), body)
    }

    @Test
    fun `an empty page_size is a parse failure, not a default`() {
        val (status, _) = send("GET", "/api/v1/documents?page_size=")
        assertEquals(422, status)
    }

    @Test
    fun `an invalid status filter is 400`() {
        val (status, _) = send("GET", "/api/v1/documents?status=nonsense")
        assertEquals(400, status)
    }

    // -- documents ----------------------------------------------------------

    @Test
    fun `an empty list still carries stats and paging`() {
        val (status, body) = send("GET", "/api/v1/documents")
        assertEquals(200, status)
        val payload = obj(body)
        assertEquals(0, payload["total"]!!.jsonPrimitive.content.toInt())
        assertTrue(payload.containsKey("stats"))
        assertTrue(payload["items"] is JsonArray)
    }

    @Test
    fun `an unknown document is 404 everywhere`() {
        for (path in listOf("/api/v1/documents/999", "/api/v1/documents/999/progress",
            "/api/v1/documents/999/image/canvas")) {
            val (status, _) = send("GET", path)
            assertEquals(404, status, path)
        }
    }

    @Test
    fun `a non-numeric id is 404, not 400`() {
        val (status, _) = send("GET", "/api/v1/documents/abc")
        assertEquals(404, status)
    }

    @Test
    fun `an upload without a file part is 400`() {
        val (status, _) = send("POST", "/api/v1/documents", body = "{}")
        assertEquals(400, status)
    }

    // -- api keys -----------------------------------------------------------

    @Test
    fun `the key list carries the banner note and masks the default`() {
        val (status, body) = send("GET", "/api/v1/api-keys", key = null, token = login())
        assertEquals(200, status)
        val payload = obj(body)
        // api-keys/Index.vue renders `note` as a banner; omitting it left an empty div on the page.
        assertTrue(payload["note"]!!.jsonPrimitive.content.isNotEmpty())
        val first = (payload["items"] as JsonArray)[0] as JsonObject
        assertTrue(first["is_default"]!!.jsonPrimitive.content.toBoolean())
        // A CONFIGURED default stays masked: whoever set it already has it.
        assertFalse(first.containsKey("key"), body)
    }

    @Test
    fun `creating a key returns the plaintext exactly once`() {
        val token = login()
        val (status, body) = send("POST", "/api/v1/api-keys", key = null, token = token,
            body = """{"label":"test"}""")
        assertEquals(201, status)
        val created = obj(body)
        val plaintext = created["key"]!!.jsonPrimitive.content
        assertTrue(plaintext.startsWith("rdk_"))

        val (_, listBody) = send("GET", "/api/v1/api-keys", key = null, token = token)
        val items = obj(listBody)["items"] as JsonArray
        val stored = items.map { it as JsonObject }.first {
            it["id"]!!.jsonPrimitive.content == created["id"]!!.jsonPrimitive.content
        }
        // Never again, and never the hash.
        assertFalse(stored.containsKey("key"), listBody)
        assertFalse(stored.containsKey("key_hash"), listBody)
    }

    @Test
    fun `deleting a key is 204 with an EMPTY body`() {
        val token = login()
        val (_, created) = send("POST", "/api/v1/api-keys", key = null, token = token, body = "{}")
        val id = obj(created)["id"]!!.jsonPrimitive.content
        val (status, body) = send("DELETE", "/api/v1/api-keys/$id", key = null, token = token)
        assertEquals(204, status)
        // A JSON body on a 204 is a protocol error some clients reject outright.
        assertEquals("", body)
    }

    @Test
    fun `deleting the default key is 409 with the message alone`() {
        val (status, body) = send("DELETE", "/api/v1/api-keys/0", key = null, token = login())
        assertEquals(409, status)
        val detail = obj(body)["detail"]!!.jsonPrimitive.content
        // The Go port shipped `"conflict: The default key …"` here, because wrapping folded the sentinel's
        // own name into the client-facing text.
        assertTrue(detail.startsWith("The default key"), detail)
    }

    // -- settings -----------------------------------------------------------

    @Test
    fun `settings return the schema and the values`() {
        val (status, body) = send("GET", "/api/v1/settings", key = null, token = login())
        assertEquals(200, status)
        val payload = obj(body)
        assertEquals(7, (payload["schema"] as JsonArray).size)
        assertEquals(7, (payload["values"] as JsonObject).size)
    }

    @Test
    fun `a rejected setting is 400 naming the bound`() {
        val (status, body) = send("PUT", "/api/v1/settings", key = null, token = login(),
            body = """{"values":{"docconf":"5"}}""")
        // 400, not 422: FastAPI's own validation is 422, but the reference raises HTTPException(400) for a
        // rejected SETTING, and the reference is the contract.
        assertEquals(400, status)
        assertTrue(obj(body)["detail"]!!.jsonPrimitive.content.contains("<= 1"), body)
    }

    @Test
    fun `a flat settings body is refused rather than silently dropped`() {
        // The wrapped shape is {"values": {...}}. Parsed flat, the whitelist drops everything and the page
        // reports success — the exact failure the settings layer exists to prevent.
        val (status, _) = send("PUT", "/api/v1/settings", key = null, token = login(),
            body = """{"docconf":"0.7"}""")
        assertEquals(400, status)
    }

    @Test
    fun `a baked-in change reports restart_required`() {
        val (status, body) = send("PUT", "/api/v1/settings", key = null, token = login(),
            body = """{"values":{"ocr_mode":"fast"}}""")
        assertEquals(200, status)
        val restart = obj(body)["restart_required"] as JsonArray
        assertEquals("ocr_mode", (restart[0]).jsonPrimitive.content)
    }

    @Test
    fun `an unknown setting key is dropped, not an error`() {
        val (status, _) = send("PUT", "/api/v1/settings", key = null, token = login(),
            body = """{"values":{"not_a_setting":"x"}}""")
        assertEquals(200, status)
    }

    // -- logs, status, health -----------------------------------------------

    @Test
    fun `logs use entries and n`() {
        val (status, body) = send("GET", "/api/v1/logs?n=5", key = null, token = login())
        assertEquals(200, status)
        val payload = obj(body)
        // The shared frontend sends `n` and reads `entries`; `items`/`limit` produced an empty logs page
        // with a 200 in the Go port.
        assertTrue(payload.containsKey("entries"))
        assertTrue(payload.containsKey("count"))
    }

    @Test
    fun `status carries every block the SPA reads`() {
        val (status, body) = send("GET", "/api/v1/status", key = null, token = login())
        assertEquals(200, status)
        val payload = obj(body)
        for (key in listOf("server", "compute", "service", "storage")) {
            assertTrue(payload.containsKey(key), key)
        }
        val service = payload["service"] as JsonObject
        // The SPA reads it from `service`, not from `storage`.
        assertTrue(service.containsKey("data_is_ephemeral"))
        val compute = payload["compute"] as JsonObject
        // Separate by design: with GPU detectors the OCR engines still run on CPU.
        assertTrue(compute.containsKey("device") && compute.containsKey("ocr_device"))
    }

    @Test
    fun `health needs no credential and reports the runtime state`() {
        val (status, body) = send("GET", "/health", key = null)
        assertEquals(200, status)
        // "ok" while models are still loading: the service IS healthy then, and gating on the runtime
        // would let Docker kill the container during startup.
        assertEquals("ok", obj(body)["status"]!!.jsonPrimitive.content)
        assertEquals("initializing", obj(body)["runtime"]!!.jsonPrimitive.content)
    }

    // -- the SPA ------------------------------------------------------------

    @Test
    fun `the SPA shell is served at the root`() {
        val (status, body) = send("GET", "/", key = null)
        assertEquals(200, status)
        assertTrue(body.contains("<!doctype html>"), body)
    }

    @Test
    fun `a hashed asset is served, not swallowed by the fallback`() {
        // The .NET port's fallback excluded paths containing a dot, so `/` returned 200 with HTML while
        // every asset 404'd and the page rendered blank with no server-side error.
        val (status, body) = send("GET", "/assets/index-abc123.js", key = null)
        assertEquals(200, status)
        assertTrue(body.contains("export default"), body)
    }

    @Test
    fun `a client-side route falls back to the shell`() {
        val (status, body) = send("GET", "/documents/42", key = null)
        assertEquals(200, status)
        assertTrue(body.contains("<!doctype html>"))
    }

    @Test
    fun `an unknown API route is JSON, never the SPA`() {
        val (status, body) = send("GET", "/api/v1/nope", key = null)
        assertEquals(404, status)
        // HTML here makes a client's JSON parse fail with a message about '<'.
        assertTrue(obj(body).containsKey("detail"), body)
    }

    @Test
    fun `a path escaping the web root is 404, not 403`() {
        val (status, _) = send("GET", "/../application.properties", key = null)
        // A prober learns nothing about the filesystem layout.
        assertTrue(status == 404 || status == 400, "got $status")
    }
}
