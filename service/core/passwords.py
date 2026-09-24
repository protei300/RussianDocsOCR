"""Password hashing — deliberately slow, unlike the API-key hashing next door.

**Why this is not `hashlib.sha256`, which `core/auth.py` uses for API keys.**
An API key is 32 random bytes: there is nothing to guess, so a fast hash is the
right tool. A password is a short string a human chose, and a fast hash over a
low-entropy input is exactly what an attacker wants — a consumer GPU tries
billions of sha256 candidates per second against a stolen file. The defence is
an algorithm that is *intentionally* expensive in time and memory, with a
per-password salt so one cracking run cannot amortise across accounts.

**Why Argon2id and not bcrypt.** Argon2id is memory-hard: the cost cannot be
bought down with parallel hardware nearly as cheaply as with bcrypt, and it is
the first recommendation in OWASP's password storage guidance. bcrypt remains a
reasonable fallback and is easier to find on the JVM; see the note below.

**Why the stored value is a PHC string and not (salt, hash) columns.**
The data directory is deliberately implementation-neutral — the Go, .NET and
Kotlin services read the same files as this one, which is how the four ports are
kept honest. A hash stored as ``$argon2id$v=19$m=65536,t=3,p=4$<salt>$<digest>``
carries its own algorithm and parameters, so any implementation can verify it
without agreeing on a private layout in advance. Splitting salt and digest into
separate fields would make this file readable only by the implementation that
wrote it, and that is the one property of the store we must not break.

Porting note for the other three services: Go has ``golang.org/x/crypto/argon2``
(BSD), .NET has ``Konscious.Security.Cryptography.Argon2`` (MIT), and on the JVM
use **BouncyCastle's** ``Argon2BytesGenerator`` rather than ``argon2-jvm`` — the
latter is LGPL, which this project has reason to avoid. All of them consume and
produce the same PHC string, so the file stays portable.

**Parameters.** The defaults below follow the OWASP baseline (64 MiB, 3
iterations, 4 lanes) and cost roughly 50-100 ms per verification on a normal
server. That is the point: it is imperceptible for one login and ruinous for a
brute-force run. They are recorded *inside* every stored hash, so raising them
later does not invalidate existing passwords — see ``needs_rehash``.
"""
from __future__ import annotations

import logging
import re
import threading

log = logging.getLogger(__name__)

#: **The import is guarded so an existing deployment keeps working.**
#: ``argon2-cffi`` is a new requirement that arrived with named accounts. A
#: service that is updated without it must still start and still serve — on the
#: PIN path, which needs no password hashing at all. Raising at import would
#: turn a missing optional dependency into a service that will not boot, and the
#: PIN mode it would have refused to run is the one that does not need the
#: library in the first place.
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
    from argon2.low_level import Type

    #: OWASP baseline. Changing these is safe: the parameters travel inside each
    #: hash, and ``needs_rehash`` reports which stored hashes predate the change.
    _HASHER = PasswordHasher(
        time_cost=3,
        memory_cost=65536,      # 64 MiB
        parallelism=4,
        hash_len=32,
        salt_len=16,
        type=Type.ID,       # Argon2id: the hybrid, resistant to GPU and side-channel attacks
    )
    AVAILABLE = True
except Exception as _import_error:                      # noqa: BLE001
    _HASHER = None
    InvalidHashError = VerificationError = VerifyMismatchError = Exception  # type: ignore[misc]
    AVAILABLE = False
    log.warning("[AUTH] argon2-cffi unavailable (%s) — named accounts are disabled, "
                "the service will use PIN authentication", _import_error)

#: **At most this many Argon2 computations at once, service-wide.**
#:
#: Memory-hard hashing is a denial-of-service lever pointed at yourself: every
#: hash costs 64 MiB, sync endpoints run on a pool of ~40 threads, and the sign-in
#: endpoint is reachable without a token. A flood of logins with a different
#: username each time walks straight past the per-account lockout, and forty
#: concurrent verifications is 2.5 GB of RAM. Four slots caps the peak at 256 MiB;
#: requests beyond that wait their turn instead of all allocating at once.
HASH_CONCURRENCY = 4
_HASH_SLOTS = threading.BoundedSemaphore(HASH_CONCURRENCY)

MIN_PASSWORD_LENGTH = 8

#: The composition rules, as data rather than a chain of ``if``s.
#:
#: The patterns are written to mean the same thing in Python's ``re`` and in a
#: browser's ``RegExp`` — no lookbehind, no ``\p{...}``, no named groups — so the
#: **server can hand this exact list to the UI** and the password page can tick
#: the rules off as you type without a second copy of them in TypeScript. Two
#: copies of a validation rule drift, and the drift shows up as a form that
#: accepts a password the server then rejects.
#:
#: Cyrillic is included in the letter classes on purpose: this is a Russian
#: deployment, and a rule that silently rejected "Пароль1" as having no letters
#: would be a bug, not a policy. The labels stay English — the UI is English-only
#: by project convention, and these strings are rendered in it.
PASSWORD_RULES: tuple[tuple[str, str, str], ...] = (
    ("length", f"at least {MIN_PASSWORD_LENGTH} characters", f".{{{MIN_PASSWORD_LENGTH},}}"),
    ("digit", "at least one digit", r"[0-9]"),
    ("letter", "at least one letter", r"[a-zA-Zа-яёА-ЯЁ]"),
    ("upper", "at least one capital letter", r"[A-ZА-ЯЁ]"),
)

#: The bootstrap password (``admin``/``1234``) fails every rule above, and that
#: is the design: it is printed on the login page, so the only thing making it
#: safe is that the account cannot do anything until it is changed. The policy
#: therefore applies to passwords a **person chooses**, never to the seeded one
#: — otherwise the service could not start with the credentials it advertises.


def hash_password(plain: str) -> str:
    """Hash a password for storage. Returns a self-describing PHC string."""
    if _HASHER is None:
        raise RuntimeError("argon2-cffi is not installed; named accounts are unavailable")
    with _HASH_SLOTS:
        return _HASHER.hash(plain)


def verify_password(stored_hash: str, candidate: str) -> bool:
    """Check a password against a stored hash.

    Returns ``False`` for a wrong password **and** for a malformed stored hash.
    The distinction is deliberately not exposed to the caller: an endpoint that
    answered differently for "wrong password" and "corrupt record" would let an
    attacker enumerate which accounts exist.
    """
    if _HASHER is None:
        return False
    try:
        with _HASH_SLOTS:
            return _HASHER.verify(stored_hash, candidate)
    except VerifyMismatchError:
        return False
    except Exception as error:                          # noqa: BLE001 - see below
        # **Fail closed on anything unexpected.** A bare `except Exception` is
        # usually a smell; in an authentication primitive it is the correct
        # default, because the two alternatives are both worse: an uncaught
        # exception turns a corrupt record into a 500 that tells an attacker the
        # account exists, and a narrow list lets the next unforeseen type
        # through as a crash.
        #
        # The list *was* narrow, and it was wrong twice in five minutes:
        # argon2-cffi encodes the stored hash as ASCII before parsing, so a
        # corrupted record with any non-ASCII byte raises UnicodeEncodeError,
        # and a missing hash (None) raises AttributeError. Neither is a
        # VerificationError. The type is logged so a real bug stays visible
        # instead of hiding behind a silent False.
        log.warning("[AUTH] unreadable password hash (%s) — treating as a failed login",
                    type(error).__name__)
        return False


def needs_rehash(stored_hash: str) -> bool:
    """True when the hash was made with weaker parameters than we now use.

    Call after a *successful* verification — that is the only moment the
    plaintext is available to re-hash with the current cost.
    """
    if _HASHER is None:
        return False
    try:
        return _HASHER.check_needs_rehash(stored_hash)
    except Exception:                                   # noqa: BLE001
        return True


def unmet_rules(candidate: str) -> list[str]:
    """Codes of the composition rules this password fails, in declaration order."""
    text = candidate or ""
    return [code for code, _label, pattern in PASSWORD_RULES if not re.search(pattern, text)]


def validate_password(candidate: str) -> str | None:
    """Returns a human-readable complaint, or ``None`` when the password is acceptable."""
    failed = unmet_rules(candidate)
    if not failed:
        return None
    labels = {code: label for code, label, _ in PASSWORD_RULES}
    return "Password needs: " + ", ".join(labels[c] for c in failed)


def rules_for_ui() -> list[dict[str, str]]:
    """The rule list as the password page consumes it — one source of truth."""
    return [{"code": code, "label": label, "pattern": pattern}
            for code, label, pattern in PASSWORD_RULES]
