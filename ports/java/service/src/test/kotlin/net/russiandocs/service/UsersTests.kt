package net.russiandocs.service

import java.io.File
import java.nio.file.Files
import java.util.concurrent.BrokenBarrierException
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import net.russiandocs.service.model.AuditEntry
import net.russiandocs.service.model.Roles
import net.russiandocs.service.repositories.Audit
import net.russiandocs.service.repositories.UserError
import net.russiandocs.service.repositories.Users
import net.russiandocs.service.store.FileStore

/**
 * The account rules and the store under them — ports/AUTH.md §8–§9, and what an HTTP black box cannot see.
 *
 * The two concurrency tests FORCE the interleaving through the repository's test hooks rather than hoping
 * two threads collide. The first Python version of the demotion test raced two threads at a barrier and
 * passed with the lock removed, because check and write take microseconds; a race test the race cannot fail
 * checks nothing.
 */
class UsersTests {

    private lateinit var dir: File
    private lateinit var db: FileStore

    @BeforeTest
    fun open() {
        dir = Files.createTempDirectory("rdocs-users").toFile()
        db = FileStore(dir.path) { }
    }

    @AfterTest
    fun close() {
        Users.onAdminCounted = null
        Users.onVerified = null
        dir.deleteRecursively()
    }

    @Test
    fun `concurrent demotion of two admins cannot remove both`() {
        val alpha = Users.create(db, "alpha", "Alpha1234", Roles.ADMIN)
        val bravo = Users.create(db, "bravo", "Bravo1234", Roles.ADMIN)

        // Inside the "last admin" count, after it is computed, each thread waits for the other. Without the
        // lock both threads count BEFORE either writes — both see the other still active, both demote, and
        // nobody is left. With the lock the second thread cannot reach the count, the wait times out, and
        // the first proceeds alone.
        val counted = CyclicBarrier(2)
        Users.onAdminCounted = {
            try {
                counted.await(500, TimeUnit.MILLISECONDS)
            } catch (e: TimeoutException) {
                // the lock kept the other thread out: correct
            } catch (e: BrokenBarrierException) {
                // ditto, seen by the thread that arrives after the timeout
            }
        }

        val start = CyclicBarrier(2)
        val outcomes = java.util.Collections.synchronizedList(ArrayList<String>())
        val threads = listOf(alpha, bravo).map { user ->
            Thread {
                start.await()
                try {
                    Users.update(db, user, role = Roles.VIEWER)
                    outcomes += "demoted"
                } catch (e: UserError) {
                    outcomes += "refused"
                }
            }
        }
        threads.forEach { it.start() }
        threads.forEach { it.join(10_000) }
        Users.onAdminCounted = null

        assertEquals(1, db.allUsers().count { it.role == Roles.ADMIN && it.isActive })
        assertEquals(listOf("demoted", "refused"), outcomes.sorted())
    }

    @Test
    fun `a sign-in does not undo a demotion made while it was hashing`() {
        Users.create(db, "alpha", "Alpha1234", Roles.ADMIN)
        val target = Users.create(db, "worker", "Worker1234", Roles.OPERATOR)

        var fired = false
        Users.onVerified = { user ->
            if (user.id == target.id && !fired) {
                fired = true
                Users.update(db, Users.get(db, target.id)!!, role = Roles.VIEWER)
            }
        }
        val signedIn = Users.authenticate(db, "worker", "Worker1234")
        Users.onVerified = null

        assertTrue(fired, "the race was not exercised")
        assertEquals(Roles.VIEWER, signedIn?.role)
        assertEquals(Roles.VIEWER, Users.get(db, target.id)?.role)
        assertNotNull(Users.get(db, target.id)?.lastLoginAt)
    }

    @Test
    fun `a sign-in is refused if the account was disabled while it was hashing`() {
        Users.create(db, "alpha", "Alpha1234", Roles.ADMIN)
        val target = Users.create(db, "worker", "Worker1234", Roles.OPERATOR)
        Users.onVerified = { Users.update(db, Users.get(db, target.id)!!, isActive = false) }
        assertNull(Users.authenticate(db, "worker", "Worker1234"))
    }

    @Test
    fun `store reads hand out copies, not the indexed object`() {
        val user = Users.create(db, "alpha", "Alpha1234", Roles.ADMIN)

        Users.get(db, user.id)!!.role = Roles.VIEWER          // edits nobody saved
        db.allUsers().first().isActive = false
        db.findUser("ALPHA")!!.tokenVersion = 99
        assertEquals(Roles.ADMIN, Users.get(db, user.id)?.role)
        assertTrue(Users.get(db, user.id)!!.isActive)
        assertEquals(1, Users.get(db, user.id)?.tokenVersion)

        // ...and writes store one: the caller's object stays the caller's.
        val held = Users.get(db, user.id)!!
        db.putUser(held)
        held.role = Roles.VIEWER
        assertEquals(Roles.ADMIN, Users.get(db, user.id)?.role)

        db.appendAudit(AuditEntry(action = "x")).actor = "tampered"
        assertEquals("", db.recentAudit(10, "", "").first().actor)
    }

    @Test
    fun `the last active administrator cannot be removed`() {
        val admin = Users.create(db, "alpha", "Alpha1234", Roles.ADMIN)
        val other = Users.create(db, "bravo", "Bravo1234", Roles.VIEWER)

        assertEquals("Cannot demote the last active administrator",
            assertFailsWith<UserError> { Users.update(db, admin, role = Roles.OPERATOR) }.message)
        assertEquals("Cannot deactivate the last active administrator",
            assertFailsWith<UserError> { Users.update(db, admin, isActive = false) }.message)
        assertEquals("Cannot delete the last active administrator",
            assertFailsWith<UserError> { Users.delete(db, admin, actingUserId = other.id) }.message)
        assertEquals("You cannot delete your own account",
            assertFailsWith<UserError> { Users.delete(db, other, actingUserId = other.id) }.message)
        // An inactive second admin does not count.
        Users.update(db, other, role = Roles.ADMIN)
        Users.update(db, other, isActive = false)
        assertFailsWith<UserError> { Users.update(db, admin, role = Roles.VIEWER) }
    }

    @Test
    fun `authority changes bump the token version, a rename does not`() {
        Users.create(db, "alpha", "Alpha1234", Roles.ADMIN)
        val user = Users.create(db, "bravo", "Bravo1234", Roles.VIEWER)
        assertEquals(1, user.tokenVersion)
        assertEquals(1, Users.update(db, user, displayName = "  Bravo  ").tokenVersion)
        assertEquals("Bravo", Users.get(db, user.id)?.displayName)
        assertEquals(2, Users.update(db, user, role = Roles.OPERATOR).tokenVersion)
        assertEquals(3, Users.update(db, user, isActive = false).tokenVersion)
        assertEquals(4, Users.resetPassword(db, user, "Reset-Pass9").tokenVersion)
        val changed = Users.changePassword(db, Users.get(db, user.id)!!, "Other-Pass9", "Reset-Pass9")
        assertEquals(5, changed.tokenVersion)
        assertEquals(false, changed.mustChangePassword)
    }

    @Test
    fun `names, roles and passwords are validated with the reference's messages`() {
        assertEquals("Username is required",
            assertFailsWith<UserError> { Users.create(db, "  ", "Alpha1234", Roles.VIEWER) }.message)
        // Cyrillic а: looks like "admin", is a different identity.
        assertTrue(assertFailsWith<UserError> { Users.create(db, "аdmin", "Alpha1234", Roles.VIEWER) }
            .message!!.startsWith("Username may contain only Latin letters"))
        assertEquals("Unknown role 'root'",
            assertFailsWith<UserError> { Users.create(db, "petrov", "Alpha1234", "root") }.message)
        assertEquals("Password needs: at least 8 characters",
            assertFailsWith<UserError> { Users.create(db, "petrov", "short1A", Roles.VIEWER) }.message)
        Users.create(db, "Petrov", "Alpha1234", Roles.VIEWER)
        assertEquals("User 'PETROV' already exists",
            assertFailsWith<UserError> { Users.create(db, "PETROV", "Alpha1234", Roles.VIEWER) }.message)
    }

    @Test
    fun `an unknown user and a wrong password look the same`() {
        Users.create(db, "alpha", "Alpha1234", Roles.ADMIN)
        assertNull(Users.authenticate(db, "alpha", "wrong"))
        assertNull(Users.authenticate(db, "nobody", "wrong"))
        assertNotNull(Users.authenticate(db, " ALPHA ", "Alpha1234"))
    }

    @Test
    fun `the seed bypasses the rules once and only into an empty store`() {
        val seeded = Users.seedAdmin(db, "admin", "1234")
        assertNotNull(seeded)
        assertEquals("Administrator", seeded.displayName)
        assertTrue(seeded.mustChangePassword)
        assertNull(Users.seedAdmin(db, "admin", "1234"))
    }

    @Test
    fun `users json is snake_case, ordered, UTF-8 and survives a restart`() {
        Users.create(db, "alpha", "Alpha1234", Roles.ADMIN, displayName = "Иван Петров")
        val text = File(dir, "users.json").readText(Charsets.UTF_8)
        assertTrue(text.contains("Иван Петров"), "Cyrillic must be written as-is: $text")
        val row = (Json.parseToJsonElement(text) as JsonArray)[0] as JsonObject
        assertEquals(listOf("id", "username", "role", "password_hash", "display_name", "is_active",
            "must_change_password", "token_version", "created_at", "last_login_at"), row.keys.toList())

        val reopened = FileStore(dir.path) { }
        val user = reopened.findUser("alpha")
        assertEquals("Иван Петров", user?.displayName)
        assertNull(user?.lastLoginAt)
        assertEquals(2, reopened.nextUserId())
    }

    @Test
    fun `unreadable account files start empty instead of failing`() {
        File(dir, "users.json").writeText("{ not json")
        File(dir, "audit.jsonl").writeText("garbage\n")
        val reopened = FileStore(dir.path) { }
        assertEquals(emptyList(), reopened.allUsers())
        assertEquals(emptyList(), reopened.recentAudit(10, "", ""))
    }

    @Test
    fun `the audit log is newest first, filtered, capped, and never fails the action`() {
        Audit.record(db, action = "login", actor = "Viewer1")
        Audit.record(db, action = "user.delete", actor = "admin", targetType = "user", targetId = 7)
        Audit.record(db, action = "login", actor = "admin")
        val all = db.recentAudit(10, "", "")
        assertEquals(listOf(3, 2, 1), all.map { it.id })
        assertEquals("7", all[1].targetId)
        assertEquals(listOf("user.delete"), db.recentAudit(10, "user.delete", "").map { it.action })
        assertEquals(listOf("Viewer1"), db.recentAudit(10, "", "VIEW").map { it.actor })

        repeat(FileStore.AUDIT_MAX_ENTRIES + 10) { Audit.record(db, action = "bulk") }
        assertEquals(FileStore.AUDIT_MAX_ENTRIES, File(dir, "audit.jsonl").readLines().count { it.isNotBlank() })
        assertEquals(FileStore.AUDIT_MAX_ENTRIES, FileStore(dir.path) { }.recentAudit(10_000, "", "").size)

        // An unwritable log: the entry is still returned, nothing is thrown.
        File(dir, "audit.jsonl").delete()
        File(dir, "audit.jsonl").mkdirs()
        File(dir, "audit.jsonl/occupied").writeText("x")      // non-empty, so no rename can replace it
        assertEquals("after", Audit.record(db, action = "after").action)
    }
}
