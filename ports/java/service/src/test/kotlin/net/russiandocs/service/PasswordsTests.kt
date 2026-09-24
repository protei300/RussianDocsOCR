package net.russiandocs.service

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue
import net.russiandocs.service.auth.Passwords

/**
 * Argon2id and the composition rules — ports/AUTH.md §3, mirrored from `tests/service/test_auth_security.py`.
 *
 * The two vectors were produced by argon2-cffi. They are the whole interop claim: a `users.json` written by
 * the Python service verifies here, and the second one proves the parameters are READ from the string —
 * it is m=19456,t=2,p=1, not the OWASP set this port writes.
 */
class PasswordsTests {

    private val vectorAscii =
        "\$argon2id\$v=19\$m=65536,t=3,p=4\$PWP+BS8J+heQ62HqF9F7Yg\$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo"
    private val vectorCyrillic =
        "\$argon2id\$v=19\$m=19456,t=2,p=1\$rppcAWOP4qJuFb6Dc52G3g\$NoDmjBcYZrj9DJvzNb421/YYxfGj1D+TxplD/tAEero"

    @Test
    fun `both interop vectors verify, and refuse a wrong password`() {
        assertTrue(Passwords.verify(vectorAscii, "Vector-Pass1"))
        assertTrue(Passwords.verify(vectorCyrillic, "Пароль-42"))
        assertFalse(Passwords.verify(vectorAscii, "Vector-Pass2"))
        assertFalse(Passwords.verify(vectorCyrillic, "Пароль-43"))
    }

    @Test
    fun `malformed hashes are false, never an exception`() {
        val salt = "PWP+BS8J+heQ62HqF9F7Yg"
        val digest = "gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo"
        val cases = listOf(
            "",
            "not a hash",
            // Right shape, wrong variant: a weaker variant is refused on read, not verified.
            "\$argon2i\$v=19\$m=65536,t=3,p=4\$$salt\$$digest",
            "\$argon2d\$v=19\$m=65536,t=3,p=4\$$salt\$$digest",
            "\$argon2id\$v=16\$m=65536,t=3,p=4\$$salt\$$digest",
            vectorAscii.dropLast(20),                                   // truncated digest
            vectorAscii.substringBeforeLast('$'),                       // no digest at all
            "\$argon2id\$v=19\$m=65536,t=3,p=4\$!!!notbase64!!!\$$digest",
            "\$argon2id\$v=19\$m=65536,t=3,p=4\$${salt.replace('+', '-')}\$$digest", // URL-safe alphabet
            "\$argon2id\$v=19\$m=65536,t=3\$$salt\$$digest",            // a parameter missing
            "\$argon2id\$v=19\$m=abc,t=3,p=4\$$salt\$$digest",
            "\$argon2id\$v=19\$m=65536,t=0,p=4\$$salt\$$digest",
            "\$argon2id\$v=19\$m=65536,t=3,p=4\$c2FsdA\$$digest",        // salt of 4 bytes
        )
        for (hash in cases) {
            assertFalse(Passwords.verify(hash, "Vector-Pass1"), hash)
        }
        assertFalse(Passwords.verify(null, "Vector-Pass1"))
    }

    @Test
    fun `an absurd memory cost is refused before anything is allocated`() {
        // 4 GiB. If the bound were checked after allocation this would either take seconds or fail with an
        // OutOfMemoryError; either way it would not come back false in a few milliseconds.
        val hostile = vectorAscii.replace("m=65536", "m=4194304")
        val started = System.nanoTime()
        assertFalse(Passwords.verify(hostile, "Vector-Pass1"))
        val ms = (System.nanoTime() - started) / 1_000_000
        assertTrue(ms < 50, "took $ms ms — the bound is not checked before hashing")
    }

    @Test
    fun `a fresh hash round-trips and carries the OWASP parameters`() {
        val hash = Passwords.hash("Str0ng-Pass")
        assertTrue(hash.startsWith("\$argon2id\$v=19\$m=65536,t=3,p=4\$"), hash)
        // Standard base64, no padding, as argon2-cffi writes it.
        val (salt, digest) = hash.split('$').takeLast(2)
        assertEquals(22 to 43, salt.length to digest.length, hash)
        assertFalse((salt + digest).contains('=') || (salt + digest).contains('-'), hash)
        assertTrue(Passwords.verify(hash, "Str0ng-Pass"))
        assertFalse(Passwords.verify(hash, "Str0ng-pass"))
        assertFalse(Passwords.needsRehash(hash))
        // Salted: the same password twice is two different strings.
        assertTrue(hash != Passwords.hash("Str0ng-Pass"))
    }

    @Test
    fun `needsRehash reports weaker parameters and unreadable hashes`() {
        assertTrue(Passwords.needsRehash(vectorCyrillic))
        assertFalse(Passwords.needsRehash(vectorAscii))
        assertTrue(Passwords.needsRehash("garbage"))
    }

    @Test
    fun `the rules count Cyrillic as letters and characters as code points`() {
        assertEquals(listOf("upper"), Passwords.unmetRules("пароль12"))
        // Seven CHARACTERS (six Cyrillic letters and a digit) fail `length` although they are 13 UTF-8
        // bytes; seven letters and a digit are eight characters and pass it. A byte count would pass both.
        assertEquals(listOf("length", "upper"), Passwords.unmetRules("пароль1"))
        assertEquals(listOf("upper"), Passwords.unmetRules("парольё1"))
        assertEquals(listOf("length"), Passwords.unmetRules("Пароль1"))
        assertEquals(emptyList(), Passwords.unmetRules("Пароль12"))
        assertEquals(listOf("digit", "upper"), Passwords.unmetRules("weakpass"))
        assertEquals(listOf("length", "digit", "letter", "upper"), Passwords.unmetRules(""))
        // Four emoji plus "Ab1" is 7 characters and 11 UTF-16 units: too short.
        assertTrue("length" in Passwords.unmetRules("😀😀😀😀Ab1"))
        assertEquals("Password needs: at least one digit, at least one capital letter",
            Passwords.validate("weakpass"))
        assertNull(Passwords.validate("Str0ng-Pass"))
    }

    @Test
    fun `the rules are served in order with the reference's patterns`() {
        val rules = Passwords.rulesForUi()
        assertEquals(listOf("length", "digit", "letter", "upper"), rules.map { it["code"] })
        assertEquals(".{8,}", rules[0]["pattern"])
        assertEquals("[a-zA-Zа-яёА-ЯЁ]", rules[2]["pattern"])
    }
}
