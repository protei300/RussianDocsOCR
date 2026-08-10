package net.russiandocs.service.api

import jakarta.servlet.http.HttpServletRequest
import net.russiandocs.service.auth.Tokens
import net.russiandocs.service.errors.ServiceException
import net.russiandocs.service.repositories.ApiKeys
import net.russiandocs.service.store.DocumentStore

/**
 * Who is calling.
 *
 * There are no user accounts: the PIN authenticates "whoever is at the console", nothing finer.
 */
public data class Identity(
    val kind: String,
    val name: String,
    val role: String,
    val keyId: Int = 0,
) {
    public companion object {
        /** The single operator identity. */
        public val SESSION: Identity = Identity("session", "Operator", "admin")
    }
}

/**
 * Authentication comes in three levels, because two kinds of caller share one API.
 *
 * ```
 * requireSession        Browser only, backed by the PIN-issued JWT. Guards anything that manages the
 *                       SERVICE — API keys, settings, logs — because those are operator concerns and an
 *                       integration has no business touching them.
 * requireApiOrSession   Either a valid X-API-Key or a valid session JWT. Guards the WORKING endpoints, so
 *                       the same routes serve the bundled UI and third-party integrations without
 *                       duplicating them.
 * optional              Never rejects. For endpoints that vary by caller but must stay reachable.
 * ```
 *
 * Why not one scheme for both: a four-digit PIN is a human affordance and a poor service credential —
 * shared, guessable, and it would have to be embedded in every integration. An API key is the opposite.
 * Conflating them forces one of the two into the wrong shape.
 *
 * **Spring Security is deliberately absent.** These three functions are the whole authorisation model, and
 * a filter chain would move the decision out of the handler that depends on it — the exact "framework magic
 * leaking into logic" the port rules forbid, and unportable to Go besides.
 */
public class Authenticator(
    private val db: DocumentStore,
    private val authConfig: Tokens.Config,
) {

    /**
     * Extracts the token from an Authorization header.
     *
     * Case-insensitive on the scheme, because clients disagree about "Bearer" versus "bearer" and rejecting
     * one of them is a support ticket, not a security measure.
     */
    private fun bearerToken(request: HttpServletRequest): String {
        val header = request.getHeader("Authorization") ?: return ""
        return if (header.length >= 7 && header.regionMatches(0, "bearer ", 0, 7, ignoreCase = true)) {
            header.substring(7).trim()
        } else {
            ""
        }
    }

    /**
     * Identifies a caller on a best-effort basis, returning `null` for anonymous.
     *
     * The session is checked FIRST because it is cheap — an HMAC over the token — while the API key path
     * hashes and then scans every stored key.
     */
    public fun optional(request: HttpServletRequest): Identity? {
        val token = bearerToken(request)
        if (token.isNotEmpty() && Tokens.decodeAccessToken(authConfig, token) != null) {
            return Identity.SESSION
        }

        val presented = request.getHeader("X-API-Key") ?: ""
        if (presented.isNotEmpty()) {
            val key = ApiKeys.verify(db, authConfig, presented)
            if (key != null) {
                ApiKeys.touch(db, key)
                return Identity("api_key", key.label, "service", key.id)
            }
        }
        return null
    }

    /** Admits browser sessions only. */
    public fun requireSession(request: HttpServletRequest): Identity {
        val token = bearerToken(request)
        if (token.isEmpty() || Tokens.decodeAccessToken(authConfig, token) == null) {
            throw ServiceException.unauthorized("Sign in with the PIN to use this endpoint")
        }
        return Identity.SESSION
    }

    /** Admits either kind of caller. */
    public fun requireApiOrSession(request: HttpServletRequest): Identity =
        optional(request)
            ?: throw ServiceException.unauthorized(
                "Provide an API key in X-API-Key, or sign in with the PIN")
}
