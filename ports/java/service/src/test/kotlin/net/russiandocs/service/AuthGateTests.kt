package net.russiandocs.service

import java.io.File
import java.nio.file.Files
import java.util.Base64
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue
import net.russiandocs.service.api.ApiException
import net.russiandocs.service.api.Authenticator
import net.russiandocs.service.api.Identity
import net.russiandocs.service.auth.AuthMode
import net.russiandocs.service.auth.LoginThrottle
import net.russiandocs.service.auth.Tokens
import net.russiandocs.service.model.Roles
import net.russiandocs.service.repositories.Users
import net.russiandocs.service.store.FileStore
import org.springframework.mock.web.MockHttpServletRequest

/**
 * The gate (ports/AUTH.md §4–§6) below the HTTP layer: token shapes, the PIN-mode `uid` refusal, the
 * published-secret rule, mode resolution and the throttle. The black-box contract test checks the same
 * rules over HTTP; these exist so each rule has a test that fails on its own when the rule is removed.
 */
class AuthGateTests {

    private lateinit var dir: File
    private lateinit var db: FileStore
    private val cfg = Tokens.Config(pin = "1234", jwtSecret = "unit-test-secret")

    @BeforeTest
    fun open() {
        dir = Files.createTempDirectory("rdocs-gate").toFile()
        db = FileStore(dir.path) { }
    }

    @AfterTest
    fun close() {
        dir.deleteRecursively()
    }

    private fun request(token: String? = null): MockHttpServletRequest = MockHttpServletRequest().also {
        if (token != null) {
            it.addHeader("Authorization", "Bearer $token")
        }
    }

    private fun b64(data: ByteArray): String = Base64.getUrlEncoder().withoutPadding().encodeToString(data)

    /** A token signed by hand, the way an attacker (or the contract test) would. */
    private fun forge(payload: String, secret: String, alg: String = "HS256"): String {
        val signing = b64("""{"alg":"$alg","typ":"JWT"}""".toByteArray()) + "." + b64(payload.toByteArray())
        val mac = Mac.getInstance("HmacSHA256").apply { init(SecretKeySpec(secret.toByteArray(), "HmacSHA256")) }
        return signing + "." + b64(mac.doFinal(signing.toByteArray()))
    }

    private val pinGate get() = Authenticator(db, { cfg }, AuthMode.Resolved(AuthMode.PIN, null))
    private val usersGate get() = Authenticator(db, { cfg }, AuthMode.Resolved(AuthMode.USERS, null))

    // -- tokens ---------------------------------------------------------------------------------------

    @Test
    fun `a PIN token carries no account claims, an account token carries integers`() {
        val pin = Tokens.createAccessToken(cfg, Tokens.Claims(sub = "operator", name = "Operator", role = "admin"))
        val body = String(Base64.getUrlDecoder().decode(pin.split('.')[1]))
        assertTrue("uid" !in body && "tv" !in body, body)

        val account = Tokens.decodeAccessToken(cfg,
            Tokens.createAccessToken(cfg, Tokens.Claims(sub = "petrov", role = "viewer", uid = 0, tv = 1)))
        assertNotNull(account)
        // Zero, not absent: the two must stay distinguishable.
        assertEquals(0L, account.uid)
        assertTrue(account.carriesUid)
    }

    @Test
    fun `the algorithm is pinned`() {
        val payload = """{"sub":"operator","role":"admin","exp":${System.currentTimeMillis() / 1000 + 60}}"""
        assertNotNull(Tokens.decodeAccessToken(cfg, forge(payload, "unit-test-secret")))
        assertNull(Tokens.decodeAccessToken(cfg, forge(payload, "unit-test-secret", alg = "HS512")))
        assertNull(Tokens.decodeAccessToken(cfg, forge(payload, "unit-test-secret", alg = "none")))
        assertNull(Tokens.decodeAccessToken(cfg, forge(payload, "wrong-secret")))
        assertNull(Tokens.decodeAccessToken(cfg, "not.a.token"))
    }

    @Test
    fun `the published default secret never signs`() {
        for (configured in listOf("", "  ", Tokens.DEFAULT_JWT_SECRET, " ${Tokens.DEFAULT_JWT_SECRET} ")) {
            val weak = Tokens.Config(jwtSecret = configured)
            assertTrue(Tokens.secretIsEphemeral(weak), "'$configured' must not be used")
            assertNotEquals(Tokens.DEFAULT_JWT_SECRET, Tokens.signingSecret(weak))
            assertTrue(Tokens.signingSecret(weak).length >= 64, "48 random bytes, base64url")
            // Once per process: the same secret on every call, so issued tokens keep verifying.
            assertEquals(Tokens.signingSecret(weak), Tokens.signingSecret(Tokens.Config()))

            val payload = """{"sub":"operator","role":"admin","exp":${System.currentTimeMillis() / 1000 + 60}}"""
            assertNull(Tokens.decodeAccessToken(weak, forge(payload, Tokens.DEFAULT_JWT_SECRET)))
            val real = Tokens.createAccessToken(weak, Tokens.Claims(sub = "operator"))
            assertNotNull(Tokens.decodeAccessToken(weak, real))
        }
        assertEquals("unit-test-secret", Tokens.signingSecret(cfg))
    }

    // -- the gate -------------------------------------------------------------------------------------

    @Test
    fun `PIN mode refuses a token that carries a uid`() {
        val exp = System.currentTimeMillis() / 1000 + 60
        assertEquals(Identity.SESSION, pinGate.session(request(forge(
            """{"sub":"operator","name":"Operator","role":"admin","exp":$exp}""", "unit-test-secret"))))
        // A viewer's still-valid account token must not become the administrator when the service is
        // switched back to PIN — and neither may a uid that is not even an integer.
        for (uid in listOf("2", "0", "\"2\"", "null")) {
            assertNull(pinGate.session(request(forge(
                """{"sub":"viewer1","uid":$uid,"tv":1,"role":"viewer","exp":$exp}""", "unit-test-secret"))), uid)
        }
    }

    @Test
    fun `users mode loads the account on every request`() {
        Users.create(db, "alpha", "Alpha1234", Roles.ADMIN, mustChangePassword = false)
        val viewer = Users.create(db, "viewer1", "Viewer1234", Roles.VIEWER, mustChangePassword = false)
        fun tokenFor(uid: Long?, tv: Long?) = Tokens.createAccessToken(cfg,
            Tokens.Claims(sub = "viewer1", role = "admin", uid = uid, tv = tv))

        val identity = usersGate.session(request(tokenFor(viewer.id.toLong(), 1)))
        assertNotNull(identity)
        // The role comes from the STORE, not from the token's claim of "admin".
        assertEquals(Roles.VIEWER, identity.role)
        assertEquals("viewer1", identity.username)

        assertNull(usersGate.session(request(tokenFor(null, 1))), "a PIN-era token")
        assertNull(usersGate.session(request(tokenFor(viewer.id.toLong(), 2))), "a stale token version")
        assertNull(usersGate.session(request(tokenFor(viewer.id.toLong(), null))), "no token version")
        assertNull(usersGate.session(request(tokenFor(999, 1))), "a deleted account")

        Users.update(db, viewer, role = Roles.OPERATOR)
        assertNull(usersGate.session(request(tokenFor(viewer.id.toLong(), 1))), "revoked by the role change")
        Users.update(db, Users.get(db, viewer.id)!!, isActive = false)
        assertNull(usersGate.session(request(tokenFor(viewer.id.toLong(), 3))), "a disabled account")
    }

    @Test
    fun `the guards answer with the reference's codes and texts`() {
        Users.create(db, "alpha", "Alpha1234", Roles.ADMIN, mustChangePassword = false)
        val restricted = Users.create(db, "fresh", "Fresh1234", Roles.ADMIN, mustChangePassword = true)
        val viewer = Users.create(db, "viewer1", "Viewer1234", Roles.VIEWER, mustChangePassword = false)
        fun token(uid: Int) = Tokens.createAccessToken(cfg, Tokens.Claims(uid = uid.toLong(), tv = 1))
        val gate = usersGate

        val anonymous = assertFailsWith<ApiException> { gate.requireAdmin.admit(request()) }
        assertEquals(401 to "Sign in to use this endpoint", anonymous.status to anonymous.detail)
        assertEquals("Bearer", anonymous.headers["WWW-Authenticate"])
        assertEquals("Provide an API key in X-API-Key, or sign in",
            assertFailsWith<ApiException> { gate.requireApiOrViewer.admit(request()) }.detail)

        assertEquals("password_change_required",
            assertFailsWith<ApiException> { gate.requireViewer.admit(request(token(restricted.id))) }.detail)
        assertEquals("password_change_required",
            assertFailsWith<ApiException> { gate.requireApiOrViewer.admit(request(token(restricted.id))) }.detail)
        assertEquals("fresh", gate.requireSessionAllowPasswordChange.admit(request(token(restricted.id))).username)

        val forbidden = assertFailsWith<ApiException> { gate.requireOperator.admit(request(token(viewer.id))) }
        assertEquals(403 to "This action requires the operator role", forbidden.status to forbidden.detail)
        assertEquals(Roles.VIEWER, gate.requireApiOrViewer.admit(request(token(viewer.id))).role)
    }

    // -- mode resolution ------------------------------------------------------------------------------

    @Test
    fun `mode resolution never fails`() {
        assertEquals(AuthMode.Resolved("pin", null), AuthMode.resolve(null, "files"))
        assertEquals(AuthMode.Resolved("pin", null), AuthMode.resolve("  ", "files"))
        assertEquals(AuthMode.Resolved("users", null), AuthMode.resolve(" USERS ", "files"))
        val typo = AuthMode.resolve("bogus", "files")
        assertEquals("pin", typo.mode)
        assertEquals("AUTH_MODE='bogus' is not one of pin, users — falling back to PIN authentication",
            typo.downgradeReason)
        val sql = AuthMode.resolve("users", "sql")
        assertEquals("pin", sql.mode)
        assertTrue(sql.downgradeReason!!.startsWith("AUTH_MODE=users is implemented for the temporary file"))
    }

    // -- the throttle ---------------------------------------------------------------------------------

    private var now = 0L
    private val second = 1_000_000_000L
    private fun throttle() = LoginThrottle(maxAttempts = 3, windowSeconds = 120, clock = { now })

    @Test
    fun `failures lock the account from one address, and only from it`() {
        val t = throttle()
        repeat(3) { t.noteFailure("Admin", "10.0.0.1") }
        now += 20 * second
        // floor(120 − 20) = 100, and the identity is case-folded and trimmed.
        assertEquals(100, t.blockedFor(" admin ", "10.0.0.1"))
        assertEquals(0, t.blockedFor("admin", "10.0.0.2"))
        now += 101 * second
        assertEquals(0, t.blockedFor("admin", "10.0.0.1"), "the window has passed")
    }

    @Test
    fun `rotating usernames does not escape the address counter`() {
        val t = throttle()
        repeat(8) { t.noteFailure("spray$it", "10.0.0.1") }
        assertEquals(0, t.blockedFor("spray99", "10.0.0.1"))
        t.noteFailure("spray8", "10.0.0.1")
        assertTrue(t.blockedFor("anyone", "10.0.0.1") > 0, "9 = 3 × 3 failures across names")
        assertEquals(0, t.blockedFor("anyone", "10.0.0.2"))
    }

    @Test
    fun `a success clears the account counter and never the address one`() {
        val t = throttle()
        repeat(2) { t.noteFailure("alice", "10.0.0.1") }
        t.clear("alice", "10.0.0.1")
        repeat(2) { t.noteFailure("alice", "10.0.0.1") }
        assertEquals(0, t.blockedFor("alice", "10.0.0.1"), "the earlier two were cleared")
        // ...but the address still remembers all four, plus five more reaches nine.
        repeat(5) { t.noteFailure("bob$it", "10.0.0.1") }
        assertTrue(t.blockedFor("carol", "10.0.0.1") > 0)
    }

    @Test
    fun `the retry hint is never zero and the map is swept`() {
        val t = throttle()
        repeat(3) { t.noteFailure("a", "x") }
        now += 119 * second + second / 2
        assertEquals(1, t.blockedFor("a", "x"))
        now += 200 * second
        repeat(10_001) { t.noteFailure("u$it", "addr$it") }
        assertTrue(t.size() <= 20_003)
        now += 200 * second
        t.noteFailure("last", "y")
        assertTrue(t.size() <= 3, "stale keys dropped once the map is large: ${t.size()}")
    }
}
