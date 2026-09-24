package net.russiandocs.service.api

import jakarta.servlet.http.HttpServletRequest
import net.russiandocs.service.auth.AuthMode
import net.russiandocs.service.auth.Tokens
import net.russiandocs.service.model.Roles
import net.russiandocs.service.repositories.ApiKeys
import net.russiandocs.service.repositories.Users
import net.russiandocs.service.store.DocumentStore

/**
 * Who is calling.
 *
 * Three shapes (ports/AUTH.md §5): the PIN operator `{session, Operator, admin}`; a named account, whose
 * fields come FROM THE STORE, not from the token; and an API key `{api_key, <label>, service}`. The
 * account-only fields are nullable so `/auth/me` can answer `null` for them in PIN mode rather than inventing
 * values.
 */
public data class Identity(
    val kind: String,
    val name: String,
    val role: String,
    val keyId: Int = 0,
    val userId: Int? = null,
    val username: String? = null,
    val mustChangePassword: Boolean? = null,
) {
    public companion object {
        public const val SESSION_KIND: String = "session"
        public const val API_KEY_KIND: String = "api_key"

        /**
         * The PIN identity. There are no accounts in that mode — the PIN authenticates "whoever is at the
         * console", nothing finer — so it holds the top role and every role guard admits it. That is what
         * keeps PIN deployments behaving exactly as they did before accounts existed.
         */
        public val SESSION: Identity = Identity(SESSION_KIND, "Operator", Roles.ADMIN)
    }

    /** The name the audit log records: the account's username, or `pin` when there is no account. */
    public val actor: String
        get() = when (kind) {
            API_KEY_KIND -> "api_key:$name"
            else -> username ?: "pin"
        }
}

/**
 * One named authorisation rule, as the route table spells it.
 *
 * **A value with a NAME rather than a bare function reference**, because the route test has to check which
 * guard each route declares, not merely that it declares one. The Python reference's first version passed a
 * "has a guard" test while a viewer could delete every document; the names here are the reference's
 * dependency names (`require_admin`, `require_api_or_operator`, …), so ports/AUTH.md §5's table and this
 * router can be compared string for string.
 */
public class Guard(public val name: String, private val check: (HttpServletRequest) -> Identity) {
    public fun admit(request: HttpServletRequest): Identity = check(request)
    override fun toString(): String = name
}

/**
 * The single gate. **One class decides**, because the alternative — each endpoint checking for itself —
 * grows a hole the day someone adds a route and forgets a line, and that hole is silent.
 *
 * | Guard | Admits |
 * |---|---|
 * | [requireSessionAllowPasswordChange] | any session, INCLUDING one that owes a password change |
 * | [requireViewer] / [requireOperator] / [requireAdmin] | a session, no pending change, role ≥ min |
 * | [requireApiOrViewer] / [requireApiOrOperator] | an API key at any level, or a session as above |
 *
 * **Fail-safe by shape.** An account flagged `must_change_password` gets a real token, but only the
 * permissive guard admits it, and it is used by exactly two routes (`/auth/me`, `/auth/change-password`).
 * A route added later with an ordinary guard refuses the restricted session instead of serving it. Naming the
 * permissive guard after what it permits is the point: it cannot be used by accident.
 *
 * **The session is re-checked against the store on every request.** A JWT is valid for eight hours, so a
 * disabled account, a changed password or a demoted role would otherwise keep their authority for the rest
 * of that window. Each request loads the user and compares `token_version`; anything that changes authority
 * bumps it and every issued token dies at once. The cost is one map lookup.
 *
 * **Spring Security is deliberately absent** — J-12. These guards are the whole authorisation model, and a
 * filter chain would move the decision away from the routing table that shows it.
 *
 * Port of `service/api/deps.py`.
 */
public class Authenticator(
    private val db: DocumentStore,
    private val authConfig: () -> Tokens.Config,
    private val mode: AuthMode.Resolved,
) {
    public companion object {
        /**
         * The 403 that means "change your password first". A machine-readable CODE the UI routes on — the
         * message text is prose and will be reworded; this must not be.
         */
        public const val PASSWORD_CHANGE_REQUIRED: String = "password_change_required"

        private const val SIGN_IN = "Sign in to use this endpoint"
        private const val KEY_OR_SIGN_IN = "Provide an API key in X-API-Key, or sign in"
    }

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

    /** Decodes a bearer token into a session identity, or `null` if it is not usable. ports/AUTH.md §5. */
    public fun session(request: HttpServletRequest): Identity? {
        val token = bearerToken(request)
        if (token.isEmpty()) {
            return null
        }
        val claims = Tokens.decodeAccessToken(authConfig(), token) ?: return null

        if (!mode.usersEnabled) {
            // PIN mode accepts only PIN tokens. A token minted while the service ran with named accounts
            // carries a uid, and it must be REFUSED, not merely have its uid ignored: every PIN session is
            // the administrator, so ignoring the uid would promote a viewer's still-valid token to full
            // control the moment the service is switched back to PIN. The reference's earlier version did
            // exactly that while its comment claimed the opposite.
            if (claims.carriesUid) {
                return null
            }
            return Identity.SESSION
        }

        // A PIN-era token (no uid) is worthless in users mode.
        val uid = claims.uid ?: return null
        if (uid < Int.MIN_VALUE || uid > Int.MAX_VALUE) {
            return null
        }
        val user = Users.get(db, uid.toInt()) ?: return null
        if (!user.isActive) {
            return null
        }
        if (claims.tv == null || claims.tv != user.tokenVersion.toLong()) {
            // Password, role or active flag changed since this token was issued — the whole reason
            // token_version exists.
            return null
        }
        return Identity(
            kind = Identity.SESSION_KIND,
            name = user.displayName.ifEmpty { user.username },
            role = user.role,
            userId = user.id,
            username = user.username,
            mustChangePassword = user.mustChangePassword,
        )
    }

    /**
     * Best-effort identification, `null` for anonymous. The session is checked FIRST: it is cheap (an HMAC
     * and a map lookup) while the API-key path hashes and then scans every stored key.
     */
    public fun optional(request: HttpServletRequest): Identity? {
        session(request)?.let { return it }

        val presented = request.getHeader("X-API-Key") ?: ""
        if (presented.isNotEmpty()) {
            val cfg = authConfig()
            val key = ApiKeys.verify(db, cfg, presented)
            if (key != null) {
                ApiKeys.touch(db, key)
                return Identity(Identity.API_KEY_KIND, key.label, "service", key.id)
            }
        }
        return null
    }

    private fun rejectIfPasswordChangePending(identity: Identity) {
        if (identity.mustChangePassword == true) {
            throw ApiException.forbidden(PASSWORD_CHANGE_REQUIRED)
        }
    }

    private fun requireRoleOf(identity: Identity, minimum: String) {
        if (!Roles.atLeast(identity.role, minimum)) {
            throw ApiException.forbidden("This action requires the $minimum role")
        }
    }

    /** A session, including a restricted one. Used by exactly two routes. */
    public val requireSessionAllowPasswordChange: Guard = Guard("require_session_allow_password_change") {
        session(it) ?: throw ApiException.unauthorized(SIGN_IN)
    }

    private fun requireRole(minimum: String): Guard = Guard("require_$minimum") { request ->
        val identity = session(request) ?: throw ApiException.unauthorized(SIGN_IN)
        rejectIfPasswordChangePending(identity)
        requireRoleOf(identity, minimum)
        identity
    }

    /**
     * An API key, or a session whose role is at least [minimum]. An API key is admitted at ANY level because
     * its scope is the document API and nothing else: users, keys, settings and logs all use [requireRole]
     * and therefore refuse API keys outright, with a 401 — the caller may retry with the right KIND of
     * credential.
     */
    private fun requireApiOrRole(minimum: String): Guard = Guard("require_api_or_$minimum") { request ->
        val identity = optional(request) ?: throw ApiException.unauthorized(KEY_OR_SIGN_IN)
        if (identity.kind != Identity.API_KEY_KIND) {
            rejectIfPasswordChangePending(identity)
            requireRoleOf(identity, minimum)
        }
        identity
    }

    public val requireViewer: Guard = requireRole(Roles.VIEWER)
    public val requireOperator: Guard = requireRole(Roles.OPERATOR)
    public val requireAdmin: Guard = requireRole(Roles.ADMIN)
    public val requireApiOrViewer: Guard = requireApiOrRole(Roles.VIEWER)
    public val requireApiOrOperator: Guard = requireApiOrRole(Roles.OPERATOR)
}
