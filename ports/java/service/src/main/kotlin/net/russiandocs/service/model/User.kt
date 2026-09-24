package net.russiandocs.service.model

import java.time.Instant
import kotlinx.serialization.KSerializer
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder

/**
 * The role ladder. Order is meaning: `viewer < operator < admin`, and a permission check is a comparison
 * against this list rather than a table of (role, endpoint) pairs — a table grows a hole the day someone adds
 * an endpoint and forgets a row, and that hole is silent.
 */
public object Roles {
    public const val VIEWER: String = "viewer"
    public const val OPERATOR: String = "operator"
    public const val ADMIN: String = "admin"
    public val ALL: List<String> = listOf(VIEWER, OPERATOR, ADMIN)

    /** True when [role] is [required] or higher. **An unknown role satisfies nothing** — deny, never allow. */
    public fun atLeast(role: String?, required: String): Boolean {
        val have = ALL.indexOf(role)
        val need = ALL.indexOf(required)
        return have >= 0 && need >= 0 && have >= need
    }
}

/**
 * [Instant] through [Timestamps], for NON-null values. Declared on nullable properties, the serialization
 * plugin wraps it in `.nullable` itself — which is what makes a JSON `null` for `last_login_at` read back as
 * null instead of throwing on `decodeString`.
 */
public object UtcInstantSerializer : KSerializer<Instant> {
    override val descriptor: SerialDescriptor =
        PrimitiveSerialDescriptor("UtcInstant", PrimitiveKind.STRING)

    override fun serialize(encoder: Encoder, value: Instant): Unit =
        encoder.encodeString(Timestamps.format(value)!!)

    override fun deserialize(decoder: Decoder): Instant = Instant.parse(decoder.decodeString())
}

/**
 * A named account for the website, used when `AUTH_MODE=users`.
 *
 * **The JSON names are the `users.json` format shared by all four services** (ports/AUTH.md §9), so every
 * one is written by hand and in the reference's field order.
 *
 * **`var`s on purpose, unlike [Document].** The repository's rules are written as "re-read inside the lock,
 * then change the fresh copy", which is the reference's shape and reads naturally with mutation. The price is
 * that the store must hand out COPIES and store copies — and it does; the store test mutates what it got and
 * checks the index did not move. An immutable type would make that test impossible to fail, which is not the
 * same as making the rule hold for a future SQL backend.
 *
 * `tokenVersion` is the part worth reading twice: it is embedded in every issued JWT and compared on every
 * request, so bumping it — password, role, active flag — kills every token already handed out for that
 * account, on every device. Without it a disabled administrator keeps working access for eight hours.
 */
@Serializable
public data class User(
    @SerialName("id") var id: Int = 0,
    @SerialName("username") var username: String = "",
    @SerialName("role") var role: String = Roles.VIEWER,
    /** Argon2id PHC string. **Never** leaves the service — see [public]. */
    @SerialName("password_hash") var passwordHash: String = "",
    @SerialName("display_name") var displayName: String = "",
    @SerialName("is_active") var isActive: Boolean = true,
    @SerialName("must_change_password") var mustChangePassword: Boolean = false,
    @SerialName("token_version") var tokenVersion: Int = 1,
    @SerialName("created_at") @Serializable(with = UtcInstantSerializer::class)
    var createdAt: Instant? = null,
    @SerialName("last_login_at") @Serializable(with = UtcInstantSerializer::class)
    var lastLoginAt: Instant? = null,
) {
    /** What the UI may see — never the hash, never the token version. */
    public fun public(): Map<String, Any?> = linkedMapOf(
        "id" to id,
        "username" to username,
        "display_name" to displayName.ifEmpty { username },
        "role" to role,
        "is_active" to isActive,
        "must_change_password" to mustChangePassword,
        "created_at" to Timestamps.format(createdAt),
        "last_login_at" to Timestamps.format(lastLoginAt),
    )
}

/**
 * One recorded action: who did what, to which object, when.
 *
 * **No personal data goes in here — a hard rule.** Documents are erased at every restart; this log
 * deliberately is not. A filename such as `Ivanov_passport.jpg`, a recognised value, or a client address
 * would quietly carry personal data across the very erasure the temporary store promises. So the target is an
 * ID, and `detail` holds things like a role name or a username.
 *
 * Not tamper-proof, and the documentation says so: anyone with write access to the data directory can edit
 * the file.
 */
@Serializable
public data class AuditEntry(
    @SerialName("id") var id: Int = 0,
    /** `login`, `login_failed`, `password.change`, `user.create`, … */
    @SerialName("action") var action: String = "",
    /** A username, `pin` for the shared PIN session, or `anonymous`. */
    @SerialName("actor") var actor: String = "",
    @SerialName("target_type") var targetType: String = "",
    /** An id as a STRING — never a filename. */
    @SerialName("target_id") var targetId: String = "",
    @SerialName("detail") var detail: String = "",
    @SerialName("at") @Serializable(with = UtcInstantSerializer::class)
    var at: Instant? = null,
)
