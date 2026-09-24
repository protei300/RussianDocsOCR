package net.russiandocs.service.auth

import java.security.MessageDigest
import java.security.SecureRandom
import java.util.Base64
import java.util.concurrent.Semaphore
import net.russiandocs.service.logging.ServiceLog
import org.bouncycastle.crypto.generators.Argon2BytesGenerator
import org.bouncycastle.crypto.params.Argon2Parameters

/**
 * Password hashing — deliberately slow, unlike the API-key hashing in [Tokens].
 *
 * **Why this is not SHA-256, which [Tokens.hashApiKey] uses.** An API key is 32 random bytes: nothing to
 * guess, so a fast hash is right. A password is a short string a human chose, and a fast hash over a
 * low-entropy input is exactly what an attacker wants — a consumer GPU tries billions of SHA-256 candidates a
 * second against a stolen file. The defence is an algorithm that is INTENTIONALLY expensive in time and
 * memory, with a per-password salt so one cracking run cannot amortise across accounts. Argon2id is
 * memory-hard and OWASP's first recommendation.
 *
 * **Why a PHC string and not (salt, hash) fields.** The data directory is implementation-neutral: the Python,
 * Go and .NET services read the same `users.json`. `$argon2id$v=19$m=65536,t=3,p=4$<salt>$<digest>` carries
 * its own algorithm and parameters, so any of the four can verify what another wrote without agreeing on a
 * private layout. ports/AUTH.md §3 is the contract; the two interop vectors there were produced by
 * argon2-cffi and are verified in this port's unit tests.
 *
 * **BouncyCastle, not `argon2-jvm`** — the latter is LGPL and wraps a native library, which would add a
 * third entry to this port's Windows DLL-loading story (J-01, J-16). BouncyCastle is pure Java. It does not
 * speak PHC, so the encoder and parser below are this file's own, and they are strict: a string that is not
 * exactly the argon2-cffi shape is refused, never guessed at.
 *
 * Port of `service/core/passwords.py`.
 */
public object Passwords {

    // OWASP baseline. Changing these is safe: the parameters travel inside every stored hash, verification
    // reads them from there, and [needsRehash] reports which hashes predate a change.
    public const val MEMORY_KIB: Int = 65536
    public const val ITERATIONS: Int = 3
    public const val PARALLELISM: Int = 4
    public const val HASH_LEN: Int = 32
    public const val SALT_LEN: Int = 16

    /** Argon2 1.3, which PHC spells `v=19`. The only version any of the four services writes or reads. */
    private const val VERSION: Int = 19

    /**
     * **At most this many Argon2 computations at once, service-wide.**
     *
     * Memory-hard hashing is a denial-of-service lever pointed at yourself: every hash costs 64 MiB, Tomcat
     * runs up to 200 request threads, and sign-in is reachable without a token. A flood of logins with a
     * different username each time walks straight past the per-account lockout, and two hundred concurrent
     * verifications would be 12.5 GiB. Four permits cap the peak at 256 MiB; the rest wait their turn.
     * Around hash AND verify — both allocate the same matrix.
     */
    public const val HASH_CONCURRENCY: Int = 4
    private val slots = Semaphore(HASH_CONCURRENCY, true)

    public const val MIN_PASSWORD_LENGTH: Int = 8

    /** One composition rule. The pattern is served to the UI verbatim and must mean the same there. */
    public data class Rule(val code: String, val label: String, val pattern: String) {
        internal val regex: Regex = Regex(pattern)
    }

    /**
     * The composition rules, as data, in order.
     *
     * Written to mean the same thing in `java.util.regex`, Python's `re` and a browser's `RegExp` — no
     * lookbehind, no `\p{…}` — so the server hands THIS list to the password page and the page ticks rules
     * off as you type without a second copy in TypeScript. Two copies of a rule drift, and the drift shows up
     * as a form that accepts a password the server then rejects.
     *
     * Cyrillic is in the letter classes on purpose: a Russian deployment that rejected "Пароль1" as having no
     * letters would be a bug, not a policy. `length` is checked by CODE POINTS — see [unmetRules].
     */
    public val RULES: List<Rule> = listOf(
        Rule("length", "at least $MIN_PASSWORD_LENGTH characters", ".{$MIN_PASSWORD_LENGTH,}"),
        Rule("digit", "at least one digit", "[0-9]"),
        Rule("letter", "at least one letter", "[a-zA-Zа-яёА-ЯЁ]"),
        Rule("upper", "at least one capital letter", "[A-ZА-ЯЁ]"),
    )

    private val log = ServiceLog("auth")
    private val random = SecureRandom()
    private val encoder: Base64.Encoder = Base64.getEncoder().withoutPadding()

    /** A stored hash that is not a PHC string this service can verify. Its NAME is what gets logged. */
    public class InvalidHashException(message: String) : IllegalArgumentException(message)

    /** The parsed form of a PHC string. */
    internal data class Phc(
        val memoryKib: Int,
        val iterations: Int,
        val parallelism: Int,
        val salt: ByteArray,
        val digest: ByteArray,
    )

    /** Hashes a password for storage and returns the self-describing PHC string. */
    public fun hash(plain: String): String {
        val salt = ByteArray(SALT_LEN).also { random.nextBytes(it) }
        val digest = derive(plain, salt, MEMORY_KIB, ITERATIONS, PARALLELISM, HASH_LEN)
        return "\$argon2id\$v=$VERSION\$m=$MEMORY_KIB,t=$ITERATIONS,p=$PARALLELISM\$" +
            encoder.encodeToString(salt) + "\$" + encoder.encodeToString(digest)
    }

    /**
     * Checks a password against a stored hash. **False for a wrong password AND for anything malformed.**
     *
     * The distinction is deliberately not exposed: an endpoint that answered differently for "wrong password"
     * and "corrupt record" would let an attacker enumerate which accounts exist, and a 500 on a corrupt record
     * says the same thing louder.
     *
     * **Fail closed on ANY exception.** A broad catch is usually a smell; in an authentication primitive it is
     * the right default, because both alternatives are worse — a propagated exception turns a corrupt record
     * into that 500, and a narrow list lets the next unforeseen type through. The Python reference had the
     * narrow list and it was wrong twice in five minutes. The exception's type is logged, so a real bug stays
     * visible instead of hiding behind a silent `false`.
     */
    public fun verify(storedHash: String?, candidate: String): Boolean = try {
        val phc = parse(storedHash ?: throw InvalidHashException("no hash"))
        val computed = derive(candidate, phc.salt, phc.memoryKib, phc.iterations, phc.parallelism,
            phc.digest.size)
        // The JDK's constant-time comparison: an early-exit equality on the digest leaks how much matched.
        MessageDigest.isEqual(computed, phc.digest)
    } catch (e: Exception) {
        log.warn("[AUTH] unreadable password hash (${e.javaClass.simpleName}) — treating as a failed login")
        false
    }

    /**
     * True when the stored hash was made with parameters other than the current ones.
     *
     * Checked after a SUCCESSFUL sign-in — the only moment the plaintext is available to re-hash with the
     * current cost. An unreadable hash reports true, as the reference does; it cannot have verified anyway.
     */
    public fun needsRehash(storedHash: String): Boolean = try {
        val phc = parse(storedHash)
        phc.memoryKib != MEMORY_KIB || phc.iterations != ITERATIONS || phc.parallelism != PARALLELISM ||
            phc.digest.size != HASH_LEN
    } catch (e: Exception) {
        true
    }

    /** Codes of the rules this password fails, in declaration order. */
    public fun unmetRules(candidate: String): List<String> = RULES.filter { rule ->
        if (rule.code == "length") {
            // **Code points, not UTF-16 units**, counted explicitly rather than left to the regex engine's
            // reading of `.`: four emoji are eight UTF-16 units and four characters, and "at least 8
            // characters" has to mean the same thing here as in Python's `re` and the Go port's rune count.
            // Also sidesteps `.` not matching a line terminator, which would make "abc\ndefgh" too short.
            candidate.codePointCount(0, candidate.length) < MIN_PASSWORD_LENGTH
        } else {
            !rule.regex.containsMatchIn(candidate)
        }
    }.map { it.code }

    /** A human-readable complaint, or `null` when the password is acceptable. */
    public fun validate(candidate: String): String? {
        val failed = unmetRules(candidate)
        if (failed.isEmpty()) {
            return null
        }
        val labels = RULES.associate { it.code to it.label }
        return "Password needs: " + failed.joinToString(", ") { labels.getValue(it) }
    }

    /** The rule list as the password page consumes it — one source of truth. */
    public fun rulesForUi(): List<Map<String, String>> =
        RULES.map { linkedMapOf("code" to it.code, "label" to it.label, "pattern" to it.pattern) }

    // --- internals -----------------------------------------------------------------------------------

    /**
     * Parses exactly `$argon2id$v=19$m=<m>,t=<t>,p=<p>$<salt>$<digest>`.
     *
     * **The bounds are checked BEFORE anything is allocated.** The parameters come from the stored string, so
     * a hostile or corrupted record could otherwise ask one login for `m=4194304` — four gigabytes — and the
     * semaphore would happily let four of them run. `argon2i` and `argon2d` are refused rather than verified:
     * the contract is Argon2id, and accepting a weaker variant on read is how a downgrade gets in.
     */
    internal fun parse(stored: String): Phc {
        val parts = stored.split('$')
        if (parts.size != 6 || parts[0].isNotEmpty()) {
            throw InvalidHashException("not a PHC string")
        }
        if (parts[1] != "argon2id") {
            throw InvalidHashException("unsupported variant")
        }
        if (parts[2] != "v=$VERSION") {
            throw InvalidHashException("unsupported version")
        }
        val params = HashMap<String, Int>()
        for (pair in parts[3].split(',')) {
            val eq = pair.indexOf('=')
            if (eq <= 0) {
                throw InvalidHashException("malformed parameters")
            }
            val key = pair.substring(0, eq)
            // Digits only: toIntOrNull would accept "+3" and "-1", neither of which argon2-cffi writes.
            val raw = pair.substring(eq + 1)
            if (raw.isEmpty() || raw.length > 9 || !raw.all { it in '0'..'9' } || key in params) {
                throw InvalidHashException("malformed parameters")
            }
            params[key] = raw.toInt()
        }
        if (params.keys != setOf("m", "t", "p")) {
            throw InvalidHashException("malformed parameters")
        }
        val m = params.getValue("m")
        val t = params.getValue("t")
        val p = params.getValue("p")
        if (t !in 1..10 || p !in 1..16 || m < 8 * p || m > 1_048_576) {
            throw InvalidHashException("parameters out of bounds")
        }
        // Standard alphabet: the JDK's basic decoder takes unpadded input and throws on a URL-safe or any
        // other stray character, which lands in verify's catch as a failed login.
        val salt = Base64.getDecoder().decode(parts[4])
        val digest = Base64.getDecoder().decode(parts[5])
        if (salt.size < 8 || digest.size !in 16..64) {
            throw InvalidHashException("salt or digest length out of bounds")
        }
        return Phc(m, t, p, salt, digest)
    }

    private fun derive(
        plain: String,
        salt: ByteArray,
        memoryKib: Int,
        iterations: Int,
        parallelism: Int,
        length: Int,
    ): ByteArray {
        val params = Argon2Parameters.Builder(Argon2Parameters.ARGON2_id)
            .withVersion(Argon2Parameters.ARGON2_VERSION_13)
            .withMemoryAsKB(memoryKib)
            .withIterations(iterations)
            .withParallelism(parallelism)
            .withSalt(salt)
            .build()
        val out = ByteArray(length)
        // Interruptibly would turn a shutdown into a spurious "failed login"; a login waiting on three others
        // for a few hundred milliseconds is not worth that.
        slots.acquireUninterruptibly()
        try {
            val generator = Argon2BytesGenerator()
            generator.init(params)
            // UTF-8 bytes, as argon2-cffi encodes a str: the Cyrillic vector in AUTH.md proves the two agree.
            generator.generateBytes(plain.toByteArray(Charsets.UTF_8), out)
        } finally {
            slots.release()
        }
        return out
    }
}
