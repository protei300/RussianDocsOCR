package net.russiandocs.service.auth

/**
 * Failed-login throttling, in memory, for one process.
 *
 * Honest about what it is: the service is pinned to a single process, so a map is the whole mechanism. A
 * deployment behind several instances needs shared state — Redis, or the reverse proxy's own rate limiting —
 * and docs/auth.md says so.
 *
 * **Two counters, and both are needed** (ports/AUTH.md §6):
 * - **per (identity, address)** — locking by account alone would let anyone lock a known user out by failing
 *   on purpose from anywhere;
 * - **per address across every identity**, under the key `*` (not a valid username, so it cannot collide) at
 *   three times the limit — without it the first counter is defeated by trying a different username each
 *   time, which is exactly what password spraying does. The reference's first version had only the first.
 *
 * The address is `request.remoteAddr` and nothing else. `X-Forwarded-For` is whatever the client wants it to
 * be, and a throttle keyed on it throttles nobody.
 *
 * Monotonic time ([System.nanoTime]), so a wall-clock step cannot lift or extend a lockout. The clock is a
 * constructor parameter only so the tests can move it.
 *
 * Port of the throttle in `service/core/auth.py`.
 */
public class LoginThrottle(
    private val maxAttempts: Int,
    private val windowSeconds: Int,
    private val clock: () -> Long = System::nanoTime,
) {
    public companion object {
        /** How many failures one address may make across ALL usernames, as a multiple of [maxAttempts]. */
        public const val ADDRESS_LIMIT_FACTOR: Int = 3

        private const val ANY_IDENTITY = "*"
        private const val SWEEP_ABOVE = 10_000
    }

    private val attempts = HashMap<Pair<String, String>, MutableList<Long>>()
    private val windowNanos: Long get() = windowSeconds.toLong() * 1_000_000_000L

    private fun key(identity: String, client: String): Pair<String, String> =
        identity.trim().lowercase() to client.ifEmpty { "-" }

    /** Seconds remaining in a lockout, or 0 when the caller may try again. */
    public fun blockedFor(identity: String, client: String): Int = synchronized(attempts) {
        val now = clock()
        maxOf(
            blocked(key(identity, client), maxAttempts, now),
            blocked(key(ANY_IDENTITY, client), maxAttempts * ADDRESS_LIMIT_FACTOR, now),
        )
    }

    private fun blocked(key: Pair<String, String>, limit: Int, now: Long): Int {
        val recent = attempts[key]?.filter { now - it < windowNanos } ?: return 0
        if (recent.size < limit) {
            return 0
        }
        // floor(window − (now − oldest)), never below 1: "try again in 0 s" invites an immediate retry that
        // is still refused.
        val remaining = (windowNanos - (now - recent.first())) / 1_000_000_000L
        return maxOf(1L, remaining).toInt()
    }

    /** Appends to BOTH counters. */
    public fun noteFailure(identity: String, client: String): Unit = synchronized(attempts) {
        val now = clock()
        for (k in listOf(key(identity, client), key(ANY_IDENTITY, client))) {
            val recent = attempts[k]?.filterTo(ArrayList()) { now - it < windowNanos } ?: ArrayList()
            recent.add(now)
            attempts[k] = recent
        }
        // Opportunistic sweep: without it the map grows once per distinct (identity, address) pair for the
        // life of the process.
        if (attempts.size > SWEEP_ABOVE) {
            attempts.entries.removeIf { (_, times) -> times.none { now - it < windowNanos } }
        }
    }

    /**
     * Called after a successful sign-in, so one typo does not linger.
     *
     * Clears the IDENTITY counter only, never the address-wide one: otherwise an attacker holding one valid
     * account could reset their budget by signing in between guesses at everyone else's.
     */
    public fun clear(identity: String, client: String): Unit = synchronized(attempts) {
        attempts.remove(key(identity, client))
    }

    /** Test seam: the number of keys held, to check the sweep. */
    internal fun size(): Int = synchronized(attempts) { attempts.size }
}
