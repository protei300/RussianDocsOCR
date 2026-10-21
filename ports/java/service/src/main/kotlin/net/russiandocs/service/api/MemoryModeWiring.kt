package net.russiandocs.service.api

import jakarta.servlet.Filter
import jakarta.servlet.FilterChain
import jakarta.servlet.ServletRequest
import jakarta.servlet.ServletResponse
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import kotlinx.serialization.json.JsonElement
import org.springframework.boot.web.servlet.FilterRegistrationBean
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import org.springframework.web.multipart.MultipartResolver
import org.springframework.web.multipart.support.StandardServletMultipartResolver
import org.springframework.web.servlet.DispatcherServlet

/**
 * The two pieces of framework wiring in-memory recognition needs, both of which must act BEFORE Spring MVC
 * touches the body. Found by component scan (the package of [ApiRoutes]), so the service and its tests get the
 * same wiring; both read the mode from [ApiServer.processing].
 */
@Configuration(proxyBeanMethods = false)
public open class MemoryModeWiring {

    /**
     * Replaces Spring Boot's multipart resolver (its auto-configuration backs off when this bean exists).
     *
     * The DispatcherServlet asks the resolver about EVERY request, and for a multipart one the standard resolver
     * calls Tomcat's `getParts()`, which writes each part past `file-size-threshold` — 0 bytes by default — to a
     * temporary file. A passport photo POSTed to `/recognize` would be on disk before the handler ran.
     */
    @Bean(name = [DispatcherServlet.MULTIPART_RESOLVER_BEAN_NAME])
    public open fun multipartResolver(api: ApiServer): MultipartResolver = UploadOnlyMultipartResolver(api)

    /** Ahead of everything but CORS, so the 404 is written before any other code sees the request. */
    @Bean
    public open fun noDocumentRoutesInMemoryMode(api: ApiServer): FilterRegistrationBean<NoDocumentRoutesFilter> =
        FilterRegistrationBean(NoDocumentRoutesFilter(api)).apply {
            order = Int.MIN_VALUE + 1
            addUrlPatterns("/*")
        }
}

/** The request path as the container decoded and normalised it — what the router matches on. */
internal fun routedPath(request: HttpServletRequest): String = (request.servletPath ?: "") + (request.pathInfo ?: "")

/**
 * Multipart parsing for exactly one route: the stored upload, `POST /documents`, and only while documents are
 * stored. Every other request — `/recognize` above all — is "not multipart" to Spring, so nothing parses its
 * body but the handler that reads it into memory.
 */
public class UploadOnlyMultipartResolver(private val api: ApiServer) : StandardServletMultipartResolver() {
    override fun isMultipart(request: HttpServletRequest): Boolean =
        api.processing.storing &&
            routedPath(request) == "${ApiServer.PREFIX}/documents" &&
            super.isMultipart(request)
}

/**
 * In `PROCESSING_MODE=memory` every `/documents` route answers 404 — **before the body is read and before
 * auth**. A check in the handlers would come too late: the upload route's body is parsed (and spooled) by the
 * framework before any handler code runs. Answering here, without calling the chain, never reads the body. The
 * cost — an anonymous caller learns that storage is off — is not a secret worth a file on disk.
 *
 * Both the raw URI and the decoded, normalised path are checked, so `/api/v1/%64ocuments` cannot route past it.
 * Port of `_no_document_routes_in_memory_mode` in `service/main.py`.
 */
public class NoDocumentRoutesFilter(private val api: ApiServer) : Filter {
    override fun doFilter(request: ServletRequest, response: ServletResponse, chain: FilterChain) {
        val http = request as? HttpServletRequest
        if (http != null && !api.processing.storing) {
            val prefix = "${ApiServer.PREFIX}/documents"
            if ((http.requestURI ?: "").startsWith(prefix) || routedPath(http).startsWith(prefix)) {
                val out = response as HttpServletResponse
                out.status = 404
                out.contentType = "application/json"
                out.characterEncoding = "UTF-8"
                // The body was never read, so the connection cannot be reused: say so, and send the answer NOW.
                // Tomcat drains an unread body before finishing a response, and gives up (dropping the
                // connection, the unsent answer with it) when the body is larger than max-swallow-size.
                out.setHeader("Connection", "close")
                out.writer.write(ApiServer.json.encodeToString(JsonElement.serializer(),
                    ApiErrors.detail(MEMORY_MODE_DETAIL)))
                out.flushBuffer()
                return
            }
        }
        chain.doFilter(request, response)
    }
}
