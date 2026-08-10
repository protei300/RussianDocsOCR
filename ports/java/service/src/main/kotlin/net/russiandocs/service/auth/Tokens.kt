package net.russiandocs.service.auth

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/**
 * Two authentication paths, for two different callers.
 *
 * - **The website** signs in with a PIN and gets a short-lived JWT. One shared operator identity; there
 *   are no user accounts.
 * - **Machine callers** send an API key in `X-API-Key`. Keys are managed from the UI at runtime, plus one
 *   bootstrap key from the environment.
 *
 * Why the split: a PIN is a human affordance and a terrible service credential — four digits, shared, and
 * it would have to be embedded in every integration. An API key is the opposite. Endpoints both kinds of
 * caller use accept either.
 *
 * Security notes, honestly:
 * - Comparison is constant-time. For the PIN that is mostly symbolic against a four-digit space: there is
 *   no rate limiting or lockout here, and a PIN is not a defence against an attacker who can reach the
 *   port. It keeps honest people out of the browser UI; the NETWORK BOUNDARY is the real control.
 * - Only key HASHES are stored. A leaked data directory must not yield working credentials.
 *
 * Port of `service/core/auth.py`. **The JWT is hand-rolled rather than taken from a dependency** — HS256
 * with two base64url segments and an HMAC is about forty lines, and the JVM ships every primitive it
 * needs in `javax.crypto`. Spring Security would bring an authentication model this service does not have
 * (no users, no roles, no sessions) and would hide the two rules that actually matter, below. The Go and
 * .NET ports made the same choice, so all three files read alike.
 */
public object Tokens {

    /**
     * Makes keys greppable in logs and recognisable when pasted somewhere they should not be — the same
     * reason GitHub uses `ghp_`.
     */
    public const val KEY_PREFIX: String = "rdk_"

    /** `rdk_` plus six characters: enough to tell keys apart. */
    public const val KEY_PREFIX_DISPLAY_LEN: Int = 10

    /** What auth needs from the environment tier. */
    public data class Config(
        val pin: String = "",
        val jwtSecret: String = "",
        val jwtAlgorithm: String = "HS256",
        val jwtExpireMinutes: Int = 480,
        val defaultApiKey: String = "",
    )

    /** The JWT payload. Only what is actually used. */
    @Serializable
    public data class Claims(
        @SerialName("sub") val sub: String = "",
        @SerialName("exp") val exp: Long = 0,
    )

    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val random = SecureRandom()

    /** Signs a JWT valid for the configured window. */
    public fun createAccessToken(cfg: Config, subject: String): String {
        if (cfg.jwtAlgorithm.isNotEmpty() && cfg.jwtAlgorithm != "HS256") {
            // Refused rather than silently downgraded: a caller who configured RS256 and got HS256 would
            // believe they had asymmetric signing.
            throw IllegalStateException(
                "auth: unsupported JWT algorithm \"${cfg.jwtAlgorithm}\" (only HS256)")
        }
        val header = """{"alg":"HS256","typ":"JWT"}""".toByteArray(Charsets.UTF_8)
        val claims = json.encodeToString(
            Claims.serializer(),
            Claims(
                sub = subject,
                exp = System.currentTimeMillis() / 1000 + cfg.jwtExpireMinutes.toLong() * 60,
            ),
        ).toByteArray(Charsets.UTF_8)
        val signing = b64(header) + "." + b64(claims)
        return signing + "." + b64(sign(signing, cfg.jwtSecret))
    }

    /**
     * Returns the claims, or `null` for anything invalid or expired.
     *
     * **The signature is verified BEFORE the claims are parsed**, and with a constant-time compare.
     * Parsing first would mean acting on attacker-controlled JSON; a plain equality test on the MAC leaks
     * how much of it matched.
     */
    public fun decodeAccessToken(cfg: Config, token: String): Claims? {
        val parts = token.split('.')
        if (parts.size != 3) {
            return null
        }
        val signing = parts[0] + "." + parts[1]
        val want = sign(signing, cfg.jwtSecret)
        val got = unb64(parts[2]) ?: return null
        if (!MessageDigest.isEqual(want, got)) {
            return null
        }

        val raw = unb64(parts[1]) ?: return null
        val claims = try {
            json.decodeFromString(Claims.serializer(), String(raw, Charsets.UTF_8))
        } catch (e: Exception) {
            return null
        }
        if (claims.exp != 0L && System.currentTimeMillis() / 1000 >= claims.exp) {
            return null
        }
        return claims
    }

    /**
     * `MessageDigest.isEqual` is the JDK's constant-time comparison — documented as such since Java 7,
     * and the reason no hand-written loop is needed here.
     */
    private fun sign(signing: String, secret: String): ByteArray {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(secret.toByteArray(Charsets.UTF_8), "HmacSHA256"))
        return mac.doFinal(signing.toByteArray(Charsets.UTF_8))
    }

    /** base64url without padding, which is what the JWT format requires. */
    private fun b64(data: ByteArray): String =
        Base64.getUrlEncoder().withoutPadding().encodeToString(data)

    private fun unb64(value: String): ByteArray? = try {
        // The decoder tolerates a missing pad; a malformed segment throws and reads as "invalid token".
        Base64.getUrlDecoder().decode(value)
    } catch (e: IllegalArgumentException) {
        null
    }

    /**
     * Compares in constant time. See the type note on what that is and is not worth for a four-digit
     * secret.
     */
    public fun verifyPin(cfg: Config, candidate: String): Boolean = MessageDigest.isEqual(
        candidate.toByteArray(Charsets.UTF_8), cfg.pin.toByteArray(Charsets.UTF_8))

    /** Mints a fresh key, shown to the user exactly once. */
    public fun generateApiKey(): String {
        val bytes = ByteArray(32)
        random.nextBytes(bytes)
        return KEY_PREFIX + Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
    }

    public fun hashApiKey(key: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(key.toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it) }
    }

    public fun prefix(key: String): String =
        if (key.length < KEY_PREFIX_DISPLAY_LEN) key else key.substring(0, KEY_PREFIX_DISPLAY_LEN)

    // --- the bootstrap key --------------------------------------------------
    //
    // Resolved once per process. Two cases:
    //
    //   DEFAULT_API_KEY set    -> use it. Stable across restarts, so integrations keep working. Treated
    //                             as a secret the operator already holds, so the UI shows it masked.
    //   DEFAULT_API_KEY unset  -> generate a random one and log it. Nobody could know it otherwise, so
    //                             the UI DOES reveal it in full. That is the deliberate trade, and it
    //                             only happens when no explicit key was configured.
    //
    // The alternative — a constant fallback in the source — would give every unconfigured deployment the
    // same publicly-known key. That is worse than either branch here.

    private val defaultGate = Any()
    private var defaultKey: String? = null
    private var defaultGenerated = false

    /** Returns the bootstrap key and whether it was generated. Idempotent; safe to call from anywhere. */
    public fun resolveDefaultKey(cfg: Config): Pair<String, Boolean> = synchronized(defaultGate) {
        if (defaultKey == null) {
            val configured = cfg.defaultApiKey.trim()
            if (configured.isNotEmpty()) {
                defaultKey = configured
                defaultGenerated = false
            } else {
                defaultKey = generateApiKey()
                defaultGenerated = true
            }
        }
        defaultKey!! to defaultGenerated
    }

    /** Test seam: forgets the resolved bootstrap key. */
    internal fun resetDefaultKeyForTests(): Unit = synchronized(defaultGate) {
        defaultKey = null
        defaultGenerated = false
    }
}
