package net.russiandocs.service.api

import jakarta.servlet.http.HttpServletRequest
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import net.russiandocs.service.model.AuditEntry
import net.russiandocs.service.model.Roles
import net.russiandocs.service.model.Timestamps
import net.russiandocs.service.model.User
import net.russiandocs.service.repositories.Audit
import net.russiandocs.service.repositories.UserError
import net.russiandocs.service.repositories.Users
import org.springframework.http.ResponseEntity

/**
 * User management — administrators only, and only with `AUTH_MODE=users`.
 *
 * **In PIN mode every route here answers 404, not an empty list** (checked AFTER the guard). "There are no
 * users" and "users are not a concept in this configuration" are different answers, and a UI that cannot tell
 * them apart shows an empty management page nobody can make work. The frontend asks `/auth/config` and hides
 * the section; the 404 is the backstop for anyone calling the API directly.
 *
 * Every mutation is audited, with the acting administrator as the actor and the account as the target — an
 * id and a username, never a password.
 *
 * Port of `service/api/users.py`.
 */

private fun ApiServer.requireUsersMode() {
    if (!mode.usersEnabled) {
        throw ApiException.notFound("User accounts are disabled (AUTH_MODE=pin)")
    }
}

private fun ApiServer.load(id: String): User {
    // A non-numeric id names no user, which is the same answer as an unknown one.
    val value = id.toIntOrNull() ?: throw ApiException.notFound("No such user")
    return Users.get(db, value) ?: throw ApiException.notFound("No such user")
}

/**
 * UserError messages are written to be shown: "Cannot demote the last active administrator" is the whole
 * explanation, and a generic 400 would leave the operator guessing which rule they hit.
 */
private inline fun <T> rules(block: () -> T): T = try {
    block()
} catch (e: UserError) {
    throw ApiException.badRequest(e.message ?: "Bad request")
}

private fun ApiServer.recordUser(action: String, admin: Identity, id: Int, detail: String) {
    Audit.record(db, action = action, actor = admin.username ?: "?", targetType = "user", targetId = id,
        detail = detail)
}

internal fun ApiServer.listUsers(): ResponseEntity<*> {
    requireUsersMode()
    return ok(JsonObject(linkedMapOf(
        "items" to JsonArray(Users.all(db).map { publicUser(it) }),
        "roles" to JsonArray(Roles.ALL.map { JsonPrimitive(it) }),
        "password_rules" to rulesJson(),
    )))
}

internal fun ApiServer.createUser(request: HttpServletRequest, admin: Identity): ResponseEntity<*> {
    requireUsersMode()
    val body = requireBody(request)
    val username = requiredString(body, "username", 64)
    val password = requiredString(body, "password", 256)
    val role = optionalString(body, "role") ?: Roles.VIEWER
    val displayName = optionalString(body, "display_name", 128) ?: ""

    val user = rules {
        Users.create(db, username, password, role, displayName, mustChangePassword = true)
    }
    recordUser("user.create", admin, user.id, "${user.username} as ${user.role}")
    return jsonResponse(201, publicUser(user))
}

internal fun ApiServer.updateUser(request: HttpServletRequest, admin: Identity, id: String): ResponseEntity<*> {
    requireUsersMode()
    val body = requireBody(request)
    val role = optionalString(body, "role")
    val displayName = optionalString(body, "display_name", 128)
    val isActive = optionalBool(body, "is_active")

    val before = load(id)
    val after = rules {
        Users.update(db, before, role = role, displayName = displayName, isActive = isActive)
    }
    val changes = buildList {
        if (before.role != after.role) {
            add("role ${before.role}->${after.role}")
        }
        if (before.isActive != after.isActive) {
            add(if (after.isActive) "activated" else "deactivated")
        }
    }
    recordUser("user.update", admin, after.id,
        "${after.username}: ${changes.joinToString(", ").ifEmpty { "profile" }}")
    return ok(publicUser(after))
}

/** Sets someone else's password. They must change it at their next sign-in. */
internal fun ApiServer.resetPassword(request: HttpServletRequest, admin: Identity, id: String): ResponseEntity<*> {
    requireUsersMode()
    val body = requireBody(request)
    val password = requiredString(body, "new_password", 256)

    val user = load(id)
    val updated = rules { Users.resetPassword(db, user, password) }
    recordUser("user.password_reset", admin, updated.id, updated.username)
    return ok(publicUser(updated))
}

internal fun ApiServer.deleteUser(admin: Identity, id: String): ResponseEntity<*> {
    requireUsersMode()
    val user = load(id)
    rules { Users.delete(db, user, actingUserId = admin.userId) }
    recordUser("user.delete", admin, user.id, user.username)
    return noContent()
}

/**
 * The action log. Administrator only — it names who did what.
 *
 * Available in **both** modes: in PIN mode the actor is the literal `pin`, and knowing that something
 * happened at a given moment is still worth more than nothing. Only account management is mode-gated.
 */
internal fun ApiServer.listAudit(request: HttpServletRequest): ResponseEntity<*> {
    // Parsed like the reference's `limit: int = 200` — an unparsable value is pydantic's 422 — and then
    // CLAMPED to 1..1000 rather than rejected, which is what the reference does with it.
    val limit = QueryParams.int(request.getParameter("limit"), "limit", 200, Int.MIN_VALUE, 0)
        .coerceIn(1, 1000)
    val entries = Audit.recent(db, limit,
        QueryParams.str(request.getParameter("action")),
        QueryParams.str(request.getParameter("actor")))
    return ok(JsonObject(linkedMapOf(
        "items" to JsonArray(entries.map { auditJson(it) }),
        "count" to JsonPrimitive(entries.size),
    )))
}

private fun auditJson(entry: AuditEntry): JsonObject = JsonObject(linkedMapOf(
    "id" to JsonPrimitive(entry.id),
    "action" to JsonPrimitive(entry.action),
    "actor" to JsonPrimitive(entry.actor),
    "target_type" to JsonPrimitive(entry.targetType),
    "target_id" to JsonPrimitive(entry.targetId),
    "detail" to JsonPrimitive(entry.detail),
    "at" to JsonPrimitive(Timestamps.format(entry.at) ?: ""),
))
