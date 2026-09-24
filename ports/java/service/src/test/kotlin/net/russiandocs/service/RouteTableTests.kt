package net.russiandocs.service

import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.util.concurrent.ConcurrentHashMap
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
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
import org.springframework.web.method.HandlerMethod
import org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerMapping

/**
 * The router against ports/AUTH.md §5's table: every route listed, nothing unlisted, each with the NAMED
 * guard.
 *
 * **Asked of the router, not of a second copy of the table.** The mappings come from Spring's
 * `RequestMappingHandlerMapping` — so a route added to [ApiRoutes] and not to [EXPECTED] fails here — and the
 * guard each one runs comes from calling it: [ApiServer.guardObserver] reports the guard's name as it runs.
 * An anonymous request cannot get past any guard, so the calls have no effect beyond the answer.
 *
 * Why the name and not merely "a guard exists": the reference's first route test asked the second question
 * and stayed green while a viewer could upload, delete and purge.
 */
class RouteTableTests {

    private companion object {
        const val P = ApiServer.PREFIX
        const val PUBLIC = "public"

        /** ports/AUTH.md §5, verbatim. */
        val EXPECTED: Set<Triple<String, String, String>> = setOf(
            Triple("GET", "/health", PUBLIC),
            Triple("GET", "$P/auth/config", PUBLIC),
            Triple("POST", "$P/auth/pin-login", PUBLIC),
            Triple("POST", "$P/auth/login", PUBLIC),
            Triple("GET", "$P/auth/me", "require_session_allow_password_change"),
            Triple("POST", "$P/auth/change-password", "require_session_allow_password_change"),
            Triple("GET", "$P/documents", "require_api_or_viewer"),
            Triple("GET", "$P/documents/{id}", "require_api_or_viewer"),
            Triple("GET", "$P/documents/{id}/progress", "require_api_or_viewer"),
            Triple("GET", "$P/documents/{id}/image/{kind}", "require_api_or_viewer"),
            Triple("POST", "$P/documents", "require_api_or_operator"),
            Triple("POST", "$P/documents/{id}/reprocess", "require_api_or_operator"),
            Triple("DELETE", "$P/documents/{id}", "require_api_or_operator"),
            Triple("POST", "$P/documents/purge", "require_admin"),
            Triple("GET", "$P/status", "require_viewer"),
            Triple("GET", "$P/api-keys", "require_admin"),
            Triple("POST", "$P/api-keys", "require_admin"),
            Triple("DELETE", "$P/api-keys/{id}", "require_admin"),
            Triple("GET", "$P/settings", "require_admin"),
            Triple("PUT", "$P/settings", "require_admin"),
            Triple("GET", "$P/logs", "require_admin"),
            Triple("GET", "$P/users", "require_admin"),
            Triple("POST", "$P/users", "require_admin"),
            Triple("PATCH", "$P/users/{id}", "require_admin"),
            Triple("POST", "$P/users/{id}/password", "require_admin"),
            Triple("DELETE", "$P/users/{id}", "require_admin"),
            Triple("GET", "$P/users/audit/entries", "require_admin"),
        )

        /** Not part of the API surface: the frontend's catch-all, which answers JSON 404 under the prefix. */
        val SPA = "GET" to "/**"
    }

    private lateinit var context: ConfigurableApplicationContext
    private lateinit var api: ApiServer
    private lateinit var dataDir: java.io.File
    private var port = 0
    private val client = HttpClient.newBuilder().proxy(java.net.ProxySelector.of(null)).build()

    @BeforeTest
    fun start() {
        dataDir = Files.createTempDirectory("rdocs-routes").toFile()
        val cfg = Settings(jwtSecret = "route-test-secret", dataDir = dataDir.path, dataWipeOnStart = false,
            seedSamples = -1)
        val log = ServiceLog("test")
        val db = FileStore(dataDir.path, log.sink())
        val runtime = PipelineRuntime(log.sink())
        val settings = SettingsRepository(cfg, log.sink())
        api = ApiServer(db, runtime, RecognitionWorker(db, runtime, cfg, settings, log), cfg, settings, null, log)
        context = SpringApplicationBuilder(Application::class.java)
            .web(WebApplicationType.SERVLET)
            .bannerMode(org.springframework.boot.Banner.Mode.OFF)
            .properties(mapOf("server.port" to 0))
            .initializers({ ctx -> ctx.beanFactory.registerSingleton("apiServer", api) })
            .run()
        port = context.environment.getProperty("local.server.port")!!.toInt()
    }

    @AfterTest
    fun stop() {
        context.close()
        dataDir.deleteRecursively()
    }

    /** Every (method, pattern) the router serves from [ApiRoutes]. */
    private fun mappings(): Set<Pair<String, String>> {
        val mapping = context.getBean("requestMappingHandlerMapping", RequestMappingHandlerMapping::class.java)
        return mapping.handlerMethods.flatMap { (info, method: HandlerMethod) ->
            if (method.beanType != ApiRoutes::class.java) {
                emptyList()
            } else {
                // A mapping with no method would answer EVERY method — itself a route the table lacks.
                val methods = info.methodsCondition.methods.map { it.name }.ifEmpty { listOf("ANY") }
                methods.flatMap { m -> info.patternValues.map { m to it } }
            }
        }.toSet()
    }

    @Test
    fun `the router serves exactly the AUTH md table`() {
        val served = mappings()
        val expected = EXPECTED.map { it.first to it.second }.toSet() + SPA
        assertEquals(emptySet(), served - expected, "routes the table does not list")
        assertEquals(emptySet(), expected - served, "routes the table lists that the router does not serve")
    }

    @Test
    fun `each route runs the guard the table names`() {
        val seen = ConcurrentHashMap<Pair<String, String>, String>()
        api.guardObserver = { method, pattern, guard -> seen[method to pattern] = guard }
        try {
            for ((method, pattern, _) in EXPECTED) {
                val path = pattern.replace("{id}", "1").replace("{kind}", "canvas")
                val request = HttpRequest.newBuilder(URI.create("http://127.0.0.1:$port$path"))
                    .header("Content-Type", "application/json")
                    .method(method, HttpRequest.BodyPublishers.ofString("{}"))
                    .build()
                val status = client.send(request, HttpResponse.BodyHandlers.discarding()).statusCode()
                val guard = seen[method to pattern]
                if (guard != null) {
                    // Anonymous: every guard must stop it, and before the body is looked at.
                    assertEquals(401, status, "$method $pattern let an anonymous caller through")
                }
            }
        } finally {
            api.guardObserver = null
        }
        val actual = EXPECTED.associate { (method, pattern, _) ->
            (method to pattern) to (seen[method to pattern] ?: PUBLIC)
        }
        val wanted = EXPECTED.associate { (method, pattern, guard) -> (method to pattern) to guard }
        for (key in wanted.keys) {
            assertEquals(wanted[key], actual[key], "${key.first} ${key.second}")
        }
    }
}
