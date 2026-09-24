package net.russiandocs.service.auth

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
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
 * - Comparison is constant-time. For the PIN that is mostly symbolic against a four-digit space; what
 *   actually limits guessing is the failed-login throttle (`LoginThrottle`), and the NETWORK BOUNDARY
 *   remains the real control.
 * - Only key HASHES are stored. A leaked data directory must not yield working credentials.
 *
 * With `AUTH_MODE=users` the same token carries a named account (`uid`, `tv`) instead of the shared
 * operator; which shape a request may present is decided by the gate in `api/Identity.kt`, not here.
 *
 * Port of `service/core/auth.py`. **The JWT is hand-rolled rather than taken from a dependency** — HS256
 * with two base64url segments and an HMAC is about forty lines, and the JVM ships every primitive it
 * needs in `javax.crypto`. Spring Security would bring its own authentication model and would hide the
 * three rules that actually matter, below. The Go and .NET ports made the same choice, so all three files
 * read alike.
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

    /**
     * The JWT payload, in both of its shapes (ports/AUTH.md §4).
     *
     * A PIN token is `{sub: "operator", name: "Operator", role: "admin", exp}`; an account token adds `uid`
     * and `tv`. **`uid` and `tv` are nullable so ABSENT and ZERO stay different** — a non-null default of 0
     * would make "no uid" and "uid 0" indistinguishable. [carriesUid] records presence separately again,
     * because a `uid` that is present but not an integer must still count as "carries a uid" in PIN mode
     * (refused) while being unusable in users mode (also refused).
     */
    @Serializable
    public data class Claims(
        @SerialName("sub") val sub: String = "",
        @SerialName("name") val name: String? = null,
        @SerialName("role") val role: String? = null,
        @SerialName("uid") val uid: Long? = null,
        @SerialName("tv") val tv: Long? = null,
        @SerialName("exp") val exp: Long = 0,
        @kotlinx.serialization.Transient val carriesUid: Boolean = uid != null,
    )

    /**
     * The published default secret. It is in this repository — so with it anyone can mint a token, and in
     * users mode that is a full takeover: the administrator is uid 1 and `token_version` starts at 1, so a
     * forged `{"uid": 1, "tv": 1}` is a guess, not an attack.
     */
    public const val DEFAULT_JWT_SECRET: String = "changeme-in-production"

    // explicitNulls = false on the way OUT, so a PIN token has no `uid`/`tv` keys at all rather than
    // `"uid": null`, which Python's `"uid" in claims` — and every port's gate — reads as present.
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true; explicitNulls = false }
    private val random = SecureRandom()

    private val secretGate = Any()
    private var processSecret: String? = null

    /** True when [signingSecret] is the random per-process one — logged at startup. */
    public fun secretIsEphemeral(cfg: Config): Boolean {
        val configured = cfg.jwtSecret.trim()
        return configured.isEmpty() || configured == DEFAULT_JWT_SECRET
    }

    /**
     * The signing secret actually in use.
     *
     * **A known secret is not a secret**, so the published default is never used to sign: when the
     * configured value is empty or still the default, 48 random bytes are generated ONCE per process. The
     * only cost is that sessions do not survive a restart — and on this service nothing does: the store is
     * wiped at every start, so a session outliving it would point at an account that no longer exists.
     */
    public fun signingSecret(cfg: Config): String {
        if (!secretIsEphemeral(cfg)) {
            return cfg.jwtSecret.trim()
        }
        synchronized(secretGate) {
            val existing = processSecret
            if (existing != null) {
                return existing
            }
            val bytes = ByteArray(48)
            random.nextBytes(bytes)
            val fresh = Base64.getUrlEncoder().withoutPadding().encodeToString(bytes)
            processSecret = fresh
            return fresh
        }
    }

    /** Signs a JWT valid for the configured window. The `exp` of [claims] is replaced. */
    public fun createAccessToken(cfg: Config, claims: Claims): String {
        if (cfg.jwtAlgorithm.isNotEmpty() && cfg.jwtAlgorithm != "HS256") {
            // Refused rather than silently downgraded: a caller who configured RS256 and got HS256 would
            // believe they had asymmetric signing.
            throw IllegalStateException(
                "auth: unsupported JWT algorithm \"${cfg.jwtAlgorithm}\" (only HS256)")
        }
        val header = """{"alg":"HS256","typ":"JWT"}""".toByteArray(Charsets.UTF_8)
        val body = json.encodeToString(
            Claims.serializer(),
            claims.copy(exp = System.currentTimeMillis() / 1000 + cfg.jwtExpireMinutes.toLong() * 60),
        ).toByteArray(Charsets.UTF_8)
        val signing = b64(header) + "." + b64(body)
        return signing + "." + b64(sign(signing, signingSecret(cfg)))
    }

    /**
     * Returns the claims, or `null` for anything invalid or expired.
     *
     * Three checks, in this order, and the order is the point:
     * 1. **The algorithm is pinned.** A header that does not say exactly `HS256` is refused before anything
     *    else is read — never negotiated, so `alg: none`, `HS512`, or an asymmetric algorithm keyed with our
     *    own secret gets nowhere.
     * 2. **The signature is verified BEFORE the claims are parsed**, with a constant-time compare. Parsing
     *    first would mean acting on attacker-controlled JSON; a plain equality test on the MAC leaks how much
     *    of it matched.
     * 3. Expiry.
     */
    public fun decodeAccessToken(cfg: Config, token: String): Claims? {
        val parts = token.split('.')
        if (parts.size != 3) {
            return null
        }
        val header = unb64(parts[0])?.let { parseObject(it) } ?: return null
        val alg = header["alg"] as? JsonPrimitive
        if (alg == null || !alg.isString || alg.content != "HS256") {
            return null
        }

        val signing = parts[0] + "." + parts[1]
        val want = sign(signing, signingSecret(cfg))
        val got = unb64(parts[2]) ?: return null
        if (!MessageDigest.isEqual(want, got)) {
            return null
        }

        val payload = unb64(parts[1])?.let { parseObject(it) } ?: return null
        val claims = try {
            // uid/tv are read by hand: the decoder would throw on a non-integer uid and drop the whole token,
            // losing the "it carried a uid" fact that PIN mode refuses on.
            val rest = JsonObject(payload.filterKeys { it != "uid" && it != "tv" })
            json.decodeFromJsonElement(Claims.serializer(), rest).copy(
                uid = integer(payload["uid"]),
                tv = integer(payload["tv"]),
                carriesUid = payload.containsKey("uid"),
            )
        } catch (e: Exception) {
            return null
        }
        if (claims.exp != 0L && System.currentTimeMillis() / 1000 >= claims.exp) {
            return null
        }
        return claims
    }

    private fun parseObject(raw: ByteArray): JsonObject? = try {
        json.parseToJsonElement(String(raw, Charsets.UTF_8)) as? JsonObject
    } catch (e: Exception) {
        null
    }

    /** A JSON integer, or null for absent, null, a string, a float or anything else. */
    private fun integer(element: JsonElement?): Long? {
        val primitive = element as? JsonPrimitive ?: return null
        if (primitive is JsonNull || primitive.isString) {
            return null
        }
        return primitive.content.toLongOrNull()
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
