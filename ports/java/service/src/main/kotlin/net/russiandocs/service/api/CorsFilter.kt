package net.russiandocs.service.api

import jakarta.servlet.Filter
import jakarta.servlet.FilterChain
import jakarta.servlet.ServletRequest
import jakarta.servlet.ServletResponse
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.HttpHeaders

/**
 * Cross-origin access for the configured origins, and for nobody else.
 *
 * **Hand-written rather than `@CrossOrigin` or `CorsConfigurationSource`**, for the reason every other
 * framework hook here is refused: exact-origin matching is fifteen lines, and Spring's CORS support sits
 * inside the `DispatcherServlet` handler mapping — which this service bypasses for the SPA catch-all, so
 * the two would disagree about which paths are covered. The Go and .NET ports write the same fifteen
 * lines, and all four now behave identically.
 *
 * Three rules, and each one is the classic CORS mistake in a service that authenticates every route that
 * matters:
 *
 * - **The origin is ECHOED only when it is on the list**, never reflected blindly. A reflected origin
 *   plus credentials is an open door.
 * - **`Access-Control-Allow-Credentials` is NOT sent**, because this API does not authenticate with
 *   cookies: a caller sends `X-API-Key` or `Authorization`, both of which work under a plain
 *   `Access-Control-Allow-Origin`. Sending it would widen the grant for nothing.
 * - **`Vary: Origin`** accompanies the header, or a shared cache serves one origin's allowed response to
 *   another origin.
 *
 * With no `CORS_ALLOWED_ORIGINS` set the filter does nothing at all, which is the right default: the
 * bundled SPA is served by this same service, so the browser makes same-origin requests and CORS is not
 * involved. The list exists for the integration case — a separate site, or the demo page opened from
 * another port.
 *
 * Port of the CORS middleware in `service/main.py`.
 */
public class CorsFilter(private val allowedOrigins: List<String>) : Filter {

    override fun doFilter(request: ServletRequest, response: ServletResponse, chain: FilterChain) {
        val http = request as HttpServletRequest
        val out = response as HttpServletResponse
        val origin = http.getHeader(HttpHeaders.ORIGIN)

        if (origin != null && origin in allowedOrigins) {
            out.setHeader(HttpHeaders.ACCESS_CONTROL_ALLOW_ORIGIN, origin)
            out.setHeader(HttpHeaders.VARY, HttpHeaders.ORIGIN)
            // X-API-Key is named explicitly: a browser preflights any request carrying a header that is
            // not on the CORS-safelisted set, and an integration's every call carries this one. Omitting
            // it makes the preflight fail with no message the calling code can see.
            out.setHeader(HttpHeaders.ACCESS_CONTROL_ALLOW_HEADERS,
                "Authorization, Content-Type, X-API-Key")
            out.setHeader(HttpHeaders.ACCESS_CONTROL_ALLOW_METHODS,
                "GET, POST, PUT, DELETE, OPTIONS")
            out.setHeader(HttpHeaders.ACCESS_CONTROL_MAX_AGE, "600")
        }

        if (http.method.equals("OPTIONS", ignoreCase = true)) {
            // The preflight is answered HERE and never reaches a handler: there is no OPTIONS mapping,
            // so Spring would answer 405 — which a browser reports only as a blocked request, sending
            // the reader to look for a bug in their own fetch call.
            out.status = HttpServletResponse.SC_NO_CONTENT
            return
        }

        chain.doFilter(request, response)
    }
}
