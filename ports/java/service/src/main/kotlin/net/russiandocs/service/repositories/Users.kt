package net.russiandocs.service.repositories

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import net.russiandocs.service.auth.Passwords
import net.russiandocs.service.model.Roles
import net.russiandocs.service.model.Timestamps
import net.russiandocs.service.model.User
import net.russiandocs.service.store.DocumentStore

/** A rule was violated. The message is written to be shown to the caller as it is — a 400's `detail`. */
public class UserError(message: String) : RuntimeException(message)

/**
 * User accounts: the whole lifecycle, and the rules that keep it usable. Only reached with
 * `AUTH_MODE=users`.
 *
 * **The rules worth stating before the code, because each only bites in production:**
 *
 * *You cannot lock everyone out.* Deleting, deactivating or demoting the last active administrator is
 * refused. It sounds like an edge case until someone demotes themselves to viewer to try the role and
 * discovers no account can undo it.
 *
 * *That check is only a check if it is atomic with the change.* Two administrators demoting each other at the
 * same instant each see the other still active, both checks pass, and the service ends with none. So every
 * mutation runs under ONE lock and RE-READS the account inside it: check and change are one step. The first
 * Python version had the gap, and its first test for it passed with the lock removed — this port's test
 * forces the interleaving through [onAdminCounted] instead of hoping two threads collide.
 *
 * *Hash before taking the lock.* Argon2 is ~80 ms and 64 MiB; holding the write lock across it would
 * serialise administration behind every sign-in.
 *
 * *Anything that changes authority bumps `token_version`* — password, role, active flag — and so kills every
 * token already issued for the account. A display-name edit does not.
 *
 * *Usernames are ASCII.* `admin` and `аdmin` (Cyrillic а) pass a case-insensitive uniqueness check and are
 * indistinguishable on screen. Display names may be anything — they are for reading, not for trusting.
 *
 * A SQL backend turns the lock into a transaction — `SELECT … FOR UPDATE` on the rows involved, or a
 * serialisable transaction around the admin count.
 *
 * Port of `service/repositories/users.py`.
 */
public object Users {

    /**
     * One lock for every change to an account. Re-entrant because a mutation may call a helper that also
     * takes it. Sign-in does NOT hold it while verifying — see [authenticate].
     */
    private val write = ReentrantLock()

    /** Letters, digits, dot, underscore, hyphen; starting with a letter or digit; ASCII only. */
    private val usernamePattern = Regex("^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

    /**
     * A real Argon2 hash of a value nobody knows, used only to spend the same time verifying a password for a
     * username that does not exist. Computed ONCE — hashing on every failed login would itself be a timing
     * signal — and forced at startup by [warmDecoy], so the first unknown-user login is not the one that
     * pays for it.
     */
    private val timingDecoy: String by lazy { Passwords.hash("no-such-user-timing-decoy") }

    /** Test hook: runs inside the last-admin count, after it is computed. Always null in production. */
    @Volatile
    internal var onAdminCounted: (() -> Unit)? = null

    /** Test hook: runs in [authenticate] between verification and write-back. Always null in production. */
    @Volatile
    internal var onVerified: ((User) -> Unit)? = null

    public fun warmDecoy() {
        timingDecoy.length
    }

    public fun all(db: DocumentStore): List<User> = db.allUsers()

    public fun get(db: DocumentStore, id: Int): User? = db.getUser(id)

    public fun find(db: DocumentStore, username: String): User? = db.findUser(username)

    internal fun countActiveAdmins(db: DocumentStore, excluding: Int? = null): Int {
        val count = db.allUsers().count { it.role == Roles.ADMIN && it.isActive && it.id != excluding }
        onAdminCounted?.invoke()
        return count
    }

    /** "Last active admin" = this account is an active admin, and no OTHER account is. */
    private fun requireNotLastAdmin(db: DocumentStore, user: User, what: String) {
        if (user.role == Roles.ADMIN && user.isActive && countActiveAdmins(db, excluding = user.id) == 0) {
            throw UserError("Cannot $what the last active administrator")
        }
    }

    private fun normaliseUsername(username: String): String {
        val name = username.trim()
        if (name.isEmpty()) {
            throw UserError("Username is required")
        }
        if (!usernamePattern.matches(name)) {
            throw UserError("Username may contain only Latin letters, digits, '.', '_' and '-', " +
                "must start with a letter or digit, and be at most 64 characters")
        }
        return name
    }

    /** Python `repr` of a string, which is what the reference's messages quote with. */
    internal fun repr(value: String): String =
        if (value.contains('\'') && !value.contains('"')) {
            "\"$value\""
        } else {
            "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
        }

    private fun requireRole(role: String) {
        if (role !in Roles.ALL) {
            throw UserError("Unknown role ${repr(role)}")
        }
    }

    /** Re-read inside the lock. The copy the caller holds may already be stale. */
    private fun fresh(db: DocumentStore, id: Int): User =
        db.getUser(id) ?: throw UserError("No such user")

    public fun create(
        db: DocumentStore,
        username: String,
        password: String,
        role: String,
        displayName: String = "",
        mustChangePassword: Boolean = true,
    ): User {
        val name = normaliseUsername(username)
        requireRole(role)
        Passwords.validate(password)?.let { throw UserError(it) }
        val hash = Passwords.hash(password)

        write.withLock {
            // Uniqueness and id allocation are checked and used in one step, or two concurrent creations of
            // "petrov" would both succeed.
            if (db.findUser(name) != null) {
                throw UserError("User ${repr(name)} already exists")
            }
            return db.putUser(User(
                id = db.nextUserId(),
                username = name,
                role = role,
                passwordHash = hash,
                displayName = displayName.trim(),
                mustChangePassword = mustChangePassword,
                createdAt = Timestamps.now(),
            ))
        }
    }

    /**
     * Creates the bootstrap administrator when there are no users at all; `null` when there already are.
     *
     * The ONE place that bypasses the password rules, and it has to: the demo password is printed on the
     * login page so the demo can be used, and it fails every rule. `mustChangePassword` is what makes that
     * defensible — the account can do nothing else until it is changed.
     */
    public fun seedAdmin(db: DocumentStore, username: String, password: String): User? {
        val name = normaliseUsername(username)
        val hash = Passwords.hash(password)
        write.withLock {
            if (db.allUsers().isNotEmpty()) {
                return null
            }
            return db.putUser(User(
                id = db.nextUserId(),
                username = name,
                role = Roles.ADMIN,
                displayName = "Administrator",
                passwordHash = hash,
                mustChangePassword = true,
                createdAt = Timestamps.now(),
            ))
        }
    }

    /**
     * Verifies credentials. `null` for an unknown user, a wrong password, or a disabled account — ONE answer
     * for all three, so the endpoint cannot be used to enumerate accounts. An unknown user is still verified,
     * against [timingDecoy], so the response time does not tell either.
     *
     * **Verification runs OUTSIDE the lock** (it is the slow part), and the bookkeeping afterwards RE-READS
     * the account under it. Writing back the copy loaded before hashing would silently undo anything an
     * administrator changed during those ~80 ms — a demotion, say. The re-read also refuses the sign-in if
     * the account was disabled, deleted, or had its password changed meanwhile.
     */
    public fun authenticate(db: DocumentStore, username: String, password: String): User? {
        val user = db.findUser(username)
        if (user == null) {
            Passwords.verify(timingDecoy, password)
            return null
        }
        if (!Passwords.verify(user.passwordHash, password)) {
            return null
        }
        onVerified?.invoke(user)

        val rehash = if (Passwords.needsRehash(user.passwordHash)) Passwords.hash(password) else null
        write.withLock {
            val current = db.getUser(user.id)
            if (current == null || !current.isActive || current.passwordHash != user.passwordHash) {
                return null
            }
            if (rehash != null) {
                current.passwordHash = rehash
            }
            current.lastLoginAt = Timestamps.now()
            return db.putUser(current)
        }
    }

    /** Changes one's own password. Bumps the token version, ending every session including this one. */
    public fun changePassword(db: DocumentStore, user: User, newPassword: String, currentPassword: String?): User {
        Passwords.validate(newPassword)?.let { throw UserError(it) }
        val newHash = Passwords.hash(newPassword)

        write.withLock {
            val current = fresh(db, user.id)
            if (currentPassword != null && !Passwords.verify(current.passwordHash, currentPassword)) {
                throw UserError("Current password is incorrect")
            }
            if (Passwords.verify(current.passwordHash, newPassword)) {
                throw UserError("The new password must differ from the current one")
            }
            current.passwordHash = newHash
            current.mustChangePassword = false
            current.tokenVersion += 1
            return db.putUser(current)
        }
    }

    /** An administrator sets someone else's password; they must change it at their next sign-in. */
    public fun resetPassword(db: DocumentStore, user: User, newPassword: String): User {
        Passwords.validate(newPassword)?.let { throw UserError(it) }
        val newHash = Passwords.hash(newPassword)
        write.withLock {
            val current = fresh(db, user.id)
            current.passwordHash = newHash
            current.mustChangePassword = true
            current.tokenVersion += 1
            return db.putUser(current)
        }
    }

    /** Changes role, display name or active flag; `null` leaves a field as it is. */
    public fun update(
        db: DocumentStore,
        user: User,
        role: String? = null,
        displayName: String? = null,
        isActive: Boolean? = null,
    ): User {
        if (role != null) {
            requireRole(role)
        }
        write.withLock {
            val current = fresh(db, user.id)
            var authorityChanged = false

            if (role != null && role != current.role) {
                if (role != Roles.ADMIN) {
                    requireNotLastAdmin(db, current, "demote")
                }
                current.role = role
                authorityChanged = true
            }
            if (isActive != null && isActive != current.isActive) {
                if (!isActive) {
                    requireNotLastAdmin(db, current, "deactivate")
                }
                current.isActive = isActive
                authorityChanged = true
            }
            if (displayName != null) {
                current.displayName = displayName.trim()
            }
            if (authorityChanged) {
                // Skipping this on a role change is the subtle hole: a demoted administrator would keep
                // administrator rights for the rest of an eight-hour token.
                current.tokenVersion += 1
            }
            return db.putUser(current)
        }
    }

    public fun delete(db: DocumentStore, user: User, actingUserId: Int?) {
        if (actingUserId != null && user.id == actingUserId) {
            // Usability more than safety: there is no undo, and deleting the account you are signed in as is
            // never what was meant.
            throw UserError("You cannot delete your own account")
        }
        write.withLock {
            val current = fresh(db, user.id)
            requireNotLastAdmin(db, current, "delete")
            db.dropUser(current.id)
        }
    }
}
