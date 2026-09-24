package net.russiandocs.service.auth

/**
 * Which authentication mode is in force — decided here, once, and nowhere else.
 *
 * Nothing else reads `Settings.authMode`: the whole point is that ONE function turns an unusable
 * configuration into a usable one, so a typo cannot enable named accounts in one handler and leave another
 * on the PIN.
 *
 * **Resolution never throws and never refuses to start.** An existing deployment that pulls this version
 * must keep working exactly as before, and "as before" is the PIN. So every way of getting it wrong — unset,
 * a typo, users mode on a backend that does not implement accounts — resolves to `pin` with a reason
 * attached. The reason is RETURNED rather than only logged so `/auth/config` can show it: a silent downgrade
 * from named accounts to a shared four-digit PIN is precisely what nobody notices from the outside.
 *
 * The reference has a fourth row — argon2-cffi missing. Here Argon2 is a compile-time dependency and cannot
 * be missing, so the row does not exist (ports/AUTH.md §1).
 *
 * Port of `resolve_auth_mode` in `service/core/auth.py`.
 */
public object AuthMode {
    public const val PIN: String = "pin"
    public const val USERS: String = "users"
    public val ALL: List<String> = listOf(PIN, USERS)

    /** The effective mode and, when it differs from what was asked for, why. */
    public data class Resolved(val mode: String, val downgradeReason: String?) {
        val usersEnabled: Boolean get() = mode == USERS
    }

    /**
     * @param raw the configured `AUTH_MODE`, as read from the environment
     * @param backend the store's backend name; named accounts exist for `files` only
     */
    public fun resolve(raw: String?, backend: String): Resolved {
        val value = (raw ?: "").trim().lowercase()
        if (value.isEmpty()) {
            return Resolved(PIN, null)          // unset is not a mistake, it is the default
        }
        if (value !in ALL) {
            // Python repr quoting, so the reason reads identically in all four services' logs.
            return Resolved(PIN, "AUTH_MODE='$value' is not one of ${ALL.joinToString(", ")} — " +
                "falling back to PIN authentication")
        }
        if (value == USERS && backend != "files") {
            return Resolved(PIN, "AUTH_MODE=users is implemented for the temporary file store only; the " +
                "database backend's user methods are stubs you are expected to implement (see docs/auth.md) " +
                "— falling back to PIN authentication")
        }
        return Resolved(value, null)
    }
}
