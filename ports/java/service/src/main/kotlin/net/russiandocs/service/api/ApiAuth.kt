package net.russiandocs.service.api

import jakarta.servlet.http.HttpServletRequest
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import net.russiandocs.service.auth.AuthMode
import net.russiandocs.service.auth.Passwords
import net.russiandocs.service.auth.Tokens
import net.russiandocs.service.config.Settings
import net.russiandocs.service.model.User
import net.russiandocs.service.repositories.Audit
import net.russiandocs.service.repositories.UserError
import net.russiandocs.service.repositories.Users
import org.springframework.http.ResponseEntity

/**
 * Sign-in for the browser UI: a shared PIN, or a named account — whichever [AuthMode] says is live.
 *
 * `GET /auth/config` is the only endpoint here that answers before anyone has authenticated, and it is what
 * lets ONE frontend serve both modes: the login page asks what to render rather than guessing.
 *
 * Security notes on this file specifically:
 * - Failed sign-ins are **throttled** ([net.russiandocs.service.auth.LoginThrottle]) and **audited**. The
 *   submitted PIN or password is never logged: rejected credentials are often a typo away from the real one.
 * - The reply to a bad username, a bad password and a disabled account is identical, including its timing
 *   (`Users.authenticate` verifies against a decoy hash). Distinguishing them hands over a list of accounts.
 * - A sign-in for an account that owes a password change returns a **restricted** token: real, but refused
 *   by every guard except the one on `/auth/me` and `/auth/change-password`. The response says so with
 *   `must_change_password`, so the UI routes to the change form instead of bouncing off a 403.
 *
 * Port of `service/api/auth.py`.
 */

// --- request bodies -----------------------------------------------------
// Pydantic's job in the reference, done by hand for the reason J-12 gives. Unparsable JSON is a 400 (the
// port's long-standing answer, and a test pins it); a field that is missing, of the wrong type or outside
// its length bounds is a 422, which is FastAPI's status. AUTH.md §7 allows either; `detail` is a string.

/** The body as a JSON object, or a 400. Called INSIDE a guard's block, never before it. */
internal fun ApiServer.requireBody(request: HttpServletRequest): JsonObject =
    bodyObject(request) ?: throw ApiException.badRequest("Expected a JSON object body")

/** A string field of 1..[max] characters (code points), or a 422 naming it. */
internal fun requiredString(body: JsonObject, key: String, max: Int): String {
    val value = optionalString(body, key, max) ?: throw ApiException.unprocessable("$key is required")
    if (value.isEmpty()) {
        throw ApiException.unprocessable("$key must not be empty")
    }
    return value
}

/** A string field that may be absent or `null` (→ null). Present but not a string, or too long → 422. */
internal fun optionalString(body: JsonObject, key: String, max: Int = Int.MAX_VALUE): String? {
    val element = body[key]
    if (element == null || element is JsonNull) {
        return null
    }
    val primitive = element as? JsonPrimitive
    if (primitive == null || !primitive.isString) {
        throw ApiException.unprocessable("$key must be a string")
    }
    val value = primitive.content
    if (value.codePointCount(0, value.length) > max) {
        throw ApiException.unprocessable("$key must be at most $max characters")
    }
    return value
}

/** A boolean field that may be absent or `null` (→ null). */
internal fun optionalBool(body: JsonObject, key: String): Boolean? {
    val element = body[key]
    if (element == null || element is JsonNull) {
        return null
    }
    val primitive = element as? JsonPrimitive
    if (primitive == null || primitive.isString) {
        throw ApiException.unprocessable("$key must be a boolean")
    }
    return primitive.booleanOrNull ?: throw ApiException.unprocessable("$key must be a boolean")
}

internal fun rulesJson(): JsonArray = JsonArray(Passwords.rulesForUi().map { rule ->
    JsonObject(rule.mapValues { (_, v) -> JsonPrimitive(v) })
})

internal fun publicUser(user: User): JsonObject = anyMapToJson(user.public())

private fun nullable(value: String?): JsonElement = value?.let { JsonPrimitive(it) } ?: JsonNull

// --- endpoints ----------------------------------------------------------

/**
 * The seeded credentials, **only while publishing them is harmless** — two conditions, both necessary.
 *
 * *The configured password is the built-in demo one.* The first Python version returned the configured
 * password unconditionally, so an operator who set `ADMIN_PASSWORD` to a real secret had it printed on the
 * login page for every anonymous visitor. A value from the environment is a secret by default; only the
 * documented demo value is not.
 *
 * *The seeded account still owes its password change.* After that "admin/1234" is false, and advertising a
 * credential that does not work is noise at best and a hint about the account name at worst.
 */
private fun ApiServer.demoCredentials(): JsonObject? {
    if (cfg.adminPassword != Settings.DEFAULT_ADMIN_PASSWORD) {
        return null
    }
    val seeded = Users.find(db, cfg.adminUsername) ?: return null
    if (!seeded.mustChangePassword) {
        return null
    }
    return JsonObject(linkedMapOf(
        "username" to JsonPrimitive(seeded.username),
        "password" to JsonPrimitive(Settings.DEFAULT_ADMIN_PASSWORD),
    ))
}

/** What the login page needs before anyone has authenticated — and nothing it does not. */
internal fun ApiServer.authConfigEndpoint(): ResponseEntity<*> {
    val payload = linkedMapOf<String, JsonElement>(
        "mode" to JsonPrimitive(mode.mode),
        "pin_required" to JsonPrimitive(mode.mode == AuthMode.PIN),
        // Named so the UI can hide user management without inferring it from the mode string; the two are
        // the same today and need not stay that way.
        "users_enabled" to JsonPrimitive(mode.usersEnabled),
        "downgrade_reason" to nullable(mode.downgradeReason),
    )
    if (mode.usersEnabled) {
        payload["password_rules"] = rulesJson()
        demoCredentials()?.let { payload["demo_credentials"] = it }
    }
    return ok(JsonObject(payload))
}

/** Exchanges the PIN for an operator token. Throttled on the identity `pin`, and audited. */
internal fun ApiServer.pinLogin(request: HttpServletRequest): ResponseEntity<*> {
    if (mode.usersEnabled) {
        // Not 401: the credential is not wrong, the endpoint is not in service.
        throw ApiException.conflict(
            "This service is configured for named accounts; sign in with a username and password")
    }
    val body = requireBody(request)
    val pin = requiredString(body, "pin", 32)

    val client = clientAddress(request)
    val blocked = throttle.blockedFor("pin", client)
    if (blocked > 0) {
        throw ApiException.tooManyAttempts(blocked)
    }

    if (!Tokens.verifyPin(authConfig, pin)) {
        throttle.noteFailure("pin", client)
        Audit.record(db, action = "login_failed", actor = "pin")
        // Without the attempted value: writing rejected PINs to disk would be its own small leak.
        log.warn("[API] rejected PIN sign-in attempt")
        // No WWW-Authenticate here, as in the reference: this is a wrong credential on the sign-in
        // endpoint itself, not a guard asking for one.
        throw ApiException(401, "Wrong PIN")
    }

    throttle.clear("pin", client)
    Audit.record(db, action = "login", actor = "pin")
    val token = Tokens.createAccessToken(authConfig, Tokens.Claims(
        sub = "operator", name = Identity.SESSION.name, role = Identity.SESSION.role))
    return ok(JsonObject(linkedMapOf(
        "access_token" to JsonPrimitive(token),
        "token_type" to JsonPrimitive("bearer"),
        "user" to JsonObject(linkedMapOf(
            "name" to JsonPrimitive(Identity.SESSION.name),
            "role" to JsonPrimitive(Identity.SESSION.role),
        )),
    )))
}

/** Signs in with a named account. */
internal fun ApiServer.login(request: HttpServletRequest): ResponseEntity<*> {
    if (!mode.usersEnabled) {
        throw ApiException.conflict("This service is configured for PIN sign-in")
    }
    val body = requireBody(request)
    val username = requiredString(body, "username", 64)
    val password = requiredString(body, "password", 256)

    val client = clientAddress(request)
    val blocked = throttle.blockedFor(username, client)
    if (blocked > 0) {
        throw ApiException.tooManyAttempts(blocked)
    }

    val user = Users.authenticate(db, username, password)
    if (user == null) {
        throttle.noteFailure(username, client)
        // The username IS recorded — it is what makes the log useful when someone walks a list of accounts.
        // The password never is, and neither is the client address: the audit log holds nothing personal,
        // and the throttle keeps the address in memory only.
        Audit.record(db, action = "login_failed", actor = username.trim())
        log.warn("[API] rejected sign-in for ${Users.repr(username)}")
        throw ApiException(401, "Wrong username or password")
    }

    throttle.clear(username, client)
    Audit.record(db, action = "login", actor = user.username)
    return ok(JsonObject(linkedMapOf(
        "access_token" to JsonPrimitive(tokenFor(user)),
        "token_type" to JsonPrimitive("bearer"),
        "user" to publicUser(user),
        "must_change_password" to JsonPrimitive(user.mustChangePassword),
    )))
}

private fun ApiServer.tokenFor(user: User): String = Tokens.createAccessToken(authConfig, Tokens.Claims(
    sub = user.username,
    name = user.displayName.ifEmpty { user.username },
    role = user.role,
    uid = user.id.toLong(),
    // What kills stale sessions: see Authenticator.session.
    tv = user.tokenVersion.toLong(),
))

/** Who the current token belongs to. Admits a restricted session, so the change form can say who it is. */
internal fun ApiServer.whoami(identity: Identity): ResponseEntity<*> = ok(JsonObject(linkedMapOf(
    "mode" to JsonPrimitive(mode.mode),
    "user" to JsonObject(linkedMapOf(
        "username" to nullable(identity.username),
        "name" to JsonPrimitive(identity.name),
        "role" to JsonPrimitive(identity.role),
        "user_id" to (identity.userId?.let { JsonPrimitive(it) } ?: JsonNull),
        "must_change_password" to (identity.mustChangePassword?.let { JsonPrimitive(it) } ?: JsonNull),
    )),
)))

/**
 * Changes one's own password, ending every session of the account — including this one.
 *
 * The current password is required although the caller is authenticated: a token left open on an unattended
 * machine must not be enough to take the account over permanently. **No new token is returned**: the version
 * bump has just invalidated this one, and handing back a fresh one would quietly defeat the point of asking
 * the user to sign in again with the password they have just chosen.
 */
internal fun ApiServer.changePassword(request: HttpServletRequest, identity: Identity): ResponseEntity<*> {
    if (!mode.usersEnabled) {
        throw ApiException.conflict("There are no accounts in PIN mode")
    }
    val body = requireBody(request)
    val current = requiredString(body, "current_password", 256)
    val next = requiredString(body, "new_password", 256)

    val user = identity.userId?.let { Users.get(db, it) }
        ?: throw ApiException(401, "Session no longer valid")
    try {
        Users.changePassword(db, user, next, currentPassword = current)
    } catch (e: UserError) {
        throw ApiException.badRequest(e.message ?: "Bad request")
    }
    Audit.record(db, action = "password.change", actor = user.username, targetType = "user",
        targetId = user.id)
    return ok(JsonObject(linkedMapOf(
        "status" to JsonPrimitive("ok"),
        "reauthenticate" to JsonPrimitive(true),
    )))
}
