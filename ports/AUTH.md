# Authentication — the contract every service implements

Normative for the Go, .NET and Kotlin services, like [`CONVENTIONS.md`](CONVENTIONS.md).
The reference is the Python service: `service/core/auth.py`, `service/core/passwords.py`,
`service/api/deps.py`, `service/api/auth.py`, `service/api/users.py`,
`service/repositories/users.py`, `service/repositories/audit.py`. **Why** each rule exists
is in [`docs/auth.md`](../docs/auth.md) (Russian); this file is **what** to build, precisely
enough that a port needs zero design decisions.

**The executable form of this file is [`conformance/auth_contract.py`](../conformance/auth_contract.py).**
It starts a service five times (PIN, PIN with a known secret, a mistyped mode, named
accounts, named accounts with a real admin password), drives it over HTTP only and checks
every rule below. The reference passes it; a port is done when it does too:

```bash
python -m conformance.auth_contract --port python    # must be green first
python -m conformance.auth_contract --port go        # after `. ports/go/env.ps1`
python -m conformance.auth_contract --port dotnet
python -m conformance.auth_contract --port java
```

It is the black box. Each port also carries its own unit tests for what a black box cannot
see (§11).

---

## 1. Two modes, chosen by configuration only

| `AUTH_MODE` | Sign-in | Identity |
|---|---|---|
| unset, `pin` | `POST /auth/pin-login` with the shared PIN | one operator, role `admin` |
| `users` | `POST /auth/login` with username + password | a named account with a role |

**Mode resolution never fails and never refuses to start.** One function returns
`(mode, downgrade_reason)`; nothing else reads the raw setting. Value is trimmed and
lower-cased first.

| raw value | effective | `downgrade_reason` |
|---|---|---|
| empty / unset | `pin` | `null` |
| not `pin`/`users` | `pin` | `AUTH_MODE='<raw>' is not one of pin, users — falling back to PIN authentication` |
| `users`, store backend is not `files` | `pin` | `AUTH_MODE=users is implemented for the temporary file store only; the database backend's user methods are stubs you are expected to implement (see docs/auth.md) — falling back to PIN authentication` |
| `users` | `users` | `null` |

(Python has a fourth row — the Argon2 library missing. In the ports Argon2 is a compile-time
dependency and cannot be missing, so the row does not exist.) The reason is logged as a
warning at startup and returned by `/auth/config`. The exact wording above is what the
reference emits; ports copy it — the contract test checks that it names the bad value and
the word `PIN`.

## 2. Configuration (environment)

| Variable | Default | |
|---|---|---|
| `AUTH_MODE` | `pin` | §1 |
| `AUTH_PIN` | `1234` | existing |
| `JWT_SECRET` | `changeme-in-production` | §4 — the default is **never used to sign** |
| `JWT_EXPIRE_MINUTES` | `480` | existing |
| `ADMIN_USERNAME` | `admin` | seeded administrator |
| `ADMIN_PASSWORD` | `1234` | its initial password; the constant `DEFAULT_ADMIN_PASSWORD = "1234"` |
| `LOGIN_MAX_ATTEMPTS` | `10` | §6 |
| `LOGIN_LOCKOUT_SECONDS` | `300` | §6 |

The mode is **not** a runtime setting and does not appear on the settings page.

## 3. Passwords

**Argon2id, PHC string, OWASP baseline:** `m=65536` (64 MiB), `t=3`, `p=4`, 32-byte
digest, 16-byte random salt, version 19. Stored as

```
$argon2id$v=19$m=65536,t=3,p=4$<salt>$<digest>
```

`<salt>` and `<digest>` are **standard base64 without padding** (alphabet `+/`, not
URL-safe) — that is what the PHC format and argon2-cffi use. A `users.json` written by any
of the four services must verify in the other three.

| | Library | Licence |
|---|---|---|
| Go | `golang.org/x/crypto/argon2` (`IDKey`) | BSD-3 |
| .NET | `Konscious.Security.Cryptography.Argon2` | MIT |
| JVM | BouncyCastle `bcprov-jdk18on`, `Argon2BytesGenerator` | MIT-style |

**Not `argon2-jvm` — it is LGPL.** None of the three parses PHC strings, so each port writes
a small encoder/parser. Verification **reads the parameters from the stored string**
(`m`, `t`, `p`, digest length = decoded digest length) — never assumes the current ones —
and compares the digest in constant time.

Rules:

- **Verify fails closed.** Any exception, malformed string, unknown variant (`argon2i`,
  `argon2d`), empty or null hash → `false`, logged as
  `[AUTH] unreadable password hash (<ExceptionType>) — treating as a failed login`.
  Never propagate: a 500 on a corrupt record tells an attacker the account exists.
  Guard the parameters before hashing (`1 ≤ t ≤ 10`, `8·p ≤ m ≤ 1 048 576`, `1 ≤ p ≤ 16`,
  digest 16–64 bytes, salt ≥ 8 bytes) so a hostile record cannot make one login allocate
  gigabytes.
- **At most 4 hashes at once, service-wide** — a counting semaphore around every
  hash *and* verify. 64 MiB each; sign-in is reachable without a token.
- `needs_rehash(hash)` = the stored `m`/`t`/`p`/digest length differ from the current ones.
  Checked after a successful sign-in; if true, re-hash with the password just verified.
- **Interop vectors** (made by argon2-cffi; every port's unit tests verify both, and reject
  both with a wrong password):

  ```
  password  Vector-Pass1
  $argon2id$v=19$m=65536,t=3,p=4$PWP+BS8J+heQ62HqF9F7Yg$gKntrbpo0K/DP8I7BSoF+jkllxaGgjqr9555lul9PXo

  password  Пароль-42          (UTF-8 bytes; different parameters — proves they are read, not assumed)
  $argon2id$v=19$m=19456,t=2,p=1$rppcAWOP4qJuFb6Dc52G3g$NoDmjBcYZrj9DJvzNb421/YYxfGj1D+TxplD/tAEero
  ```

**Composition rules**, as data, in this order. The patterns are served to the UI verbatim
and must mean the same thing in the browser's `RegExp`:

| code | label | pattern |
|---|---|---|
| `length` | `at least 8 characters` | `.{8,}` |
| `digit` | `at least one digit` | `[0-9]` |
| `letter` | `at least one letter` | `[a-zA-Zа-яёА-ЯЁ]` |
| `upper` | `at least one capital letter` | `[A-ZА-ЯЁ]` |

A password fails a rule when the pattern finds no match anywhere in it. `.{8,}` counts
**characters (code points)**, not bytes and not UTF-16 units — in Go use a rune count,
not `len`. Validation message: `Password needs: ` + the failed labels joined with `, `.
The **seeded** admin password bypasses the rules (it is `1234` by design); every password a
person chooses goes through them.

## 4. Session tokens

HS256 JWT, hand-rolled as the ports already do. **The algorithm is pinned:** a header that
does not say exactly `HS256` is refused before anything else is read.

**Signing secret.** If `JWT_SECRET` (trimmed) is empty or equals `changeme-in-production`,
the service generates a random secret (≥ 48 random bytes, base64url) **once per process**
and uses that. Startup then logs a warning:
`[BOOT] JWT_SECRET is unset or still the published default — using a random per-process secret instead; every session ends when the service restarts. Set JWT_SECRET to keep sessions across restarts.`
Replace the existing "JWT_SECRET is the built-in default" warning with this one.

Claims:

| | PIN token | account token |
|---|---|---|
| `sub` | `"operator"` | username |
| `name` | `"Operator"` | display name, or username if empty |
| `role` | `"admin"` | the account's role |
| `uid` | — **absent** | integer user id |
| `tv` | — **absent** | integer `token_version` |
| `exp` | integer seconds | integer seconds |

The claims type must accept both shapes; `uid`/`tv` must be distinguishable between
*absent* and *zero* (pointer / nullable).

## 5. The gate — one function decides

A session is resolved from the bearer token like this, **on every request**:

1. Decode and verify; any failure → no session.
2. **PIN mode:** if the claims carry `uid` → **no session** (refused, not ignored — every
   PIN session is the administrator, so ignoring `uid` would promote a viewer's live token
   the moment the service is switched back to PIN). Otherwise the operator identity
   `{kind: session, name: Operator, role: admin}`.
3. **Users mode:** `uid` absent or not an integer → no session (a PIN-era token). Load the
   user by id; missing, `is_active == false`, or `tv != token_version` → no session.
   Otherwise `{kind: session, user_id, username, name: display_name or username, role,
   must_change_password}` — **from the store, not from the token**.

An API key (`X-API-Key`) yields `{kind: api_key, name: label, role: service}` as today.
The session is checked first.

Guards, as the reference names them (the names appear in the route table below):

| Guard | Admits | Rejects with |
|---|---|---|
| `require_session_allow_password_change` | any session, **including** one that owes a password change | 401 `Sign in to use this endpoint` |
| `require_role(min)` → `require_viewer` / `require_operator` / `require_admin` | a session, no pending change, role ≥ min | 401 as above; 403 `password_change_required`; 403 `This action requires the <min> role` |
| `require_api_or_role(min)` → `require_api_or_viewer` / `require_api_or_operator` | an API key at any level, **or** a session as `require_role(min)` | 401 `Provide an API key in X-API-Key, or sign in`; then 403 as above |

Role order `viewer < operator < admin`; an unknown role satisfies nothing. Every 401 from a
guard carries `WWW-Authenticate: Bearer`. **`password_change_required` is a
machine-readable code in `detail`** — the UI routes on it; do not reword it.

Fail-safe by shape: only the permissive guard admits a restricted session, and it is used
by exactly two routes. A route added later with an ordinary guard refuses it.

### Route table

This is the whole surface; every route not listed does not exist.

| Method | Path | Guard |
|---|---|---|
| GET | `/health` | public |
| GET | `/api/v1/auth/config` | public |
| POST | `/api/v1/auth/pin-login` | public |
| POST | `/api/v1/auth/login` | public |
| GET | `/api/v1/auth/me` | `require_session_allow_password_change` |
| POST | `/api/v1/auth/change-password` | `require_session_allow_password_change` |
| GET | `/api/v1/documents` | `require_api_or_viewer` |
| GET | `/api/v1/documents/{id}` | `require_api_or_viewer` |
| GET | `/api/v1/documents/{id}/progress` | `require_api_or_viewer` |
| GET | `/api/v1/documents/{id}/image/{kind}` | `require_api_or_viewer` |
| POST | `/api/v1/documents` | `require_api_or_operator` |
| POST | `/api/v1/documents/{id}/reprocess` | `require_api_or_operator` |
| DELETE | `/api/v1/documents/{id}` | `require_api_or_operator` |
| POST | `/api/v1/documents/purge` | `require_admin` |
| GET | `/api/v1/status` | `require_viewer` |
| GET/POST | `/api/v1/api-keys` | `require_admin` |
| DELETE | `/api/v1/api-keys/{id}` | `require_admin` |
| GET/PUT | `/api/v1/settings` | `require_admin` |
| GET | `/api/v1/logs` | `require_admin` |
| GET/POST | `/api/v1/users` | `require_admin` |
| PATCH | `/api/v1/users/{id}` | `require_admin` |
| POST | `/api/v1/users/{id}/password` | `require_admin` |
| DELETE | `/api/v1/users/{id}` | `require_admin` |
| GET | `/api/v1/users/audit/entries` | `require_admin` |

**The guard runs before the body is read or validated** — a viewer's upload is a 403, not a
422 about the file. In PIN mode all of these are satisfied by the operator session, so PIN
deployments behave exactly as before.

**Each port's route test compares its router against this table** (every route listed,
nothing unlisted, each with the named guard). A test that only asks "is there *a* guard"
passed on the first Python version while a viewer could delete everything.

## 6. Failed-login throttling

In memory, one process. Two counters, both keyed with the client address
(`RemoteAddr` host / `HttpContext.Connection.RemoteIpAddress` / `request.remoteAddr` —
never `X-Forwarded-For`):

- **per (identity, address)** — identity is the trimmed, case-folded username, or the
  literal `pin` for PIN sign-in; limit `LOGIN_MAX_ATTEMPTS`;
- **per address across all identities** — key identity `*` (not a valid username, so it
  cannot collide); limit `LOGIN_MAX_ATTEMPTS × 3`.

Both count failures within the last `LOGIN_LOCKOUT_SECONDS` (monotonic clock). Before
checking credentials: if either counter is at its limit, answer **429**
`Too many attempts. Try again in <n> s` with `Retry-After: <n>`, where
`n = max(1, floor(window − (now − oldest failure in window)))`, the larger of the two.
A failure appends to **both**; a success clears **only the per-identity** counter (never the
address one). When the map exceeds 10 000 keys, drop keys with no failure in the window.

## 7. Endpoints

All errors are `{"detail": "<text>"}`. Request bodies: reject lengths beyond the limits
(pin 1–32, username 1–64, password 1–256, display name ≤ 128) with 400 or 422.

**`GET /auth/config`** — public.

```json
{"mode": "pin", "pin_required": true, "users_enabled": false, "downgrade_reason": null}
```

In users mode additionally `"password_rules": [{"code","label","pattern"}, …]` and,
**only while** `ADMIN_PASSWORD == "1234"` **and** the account named `ADMIN_USERNAME`
exists and still has `must_change_password`,
`"demo_credentials": {"username": "<its username>", "password": "1234"}`. A real
`ADMIN_PASSWORD` is never published — the first Python version did.

**`POST /auth/pin-login`** `{"pin"}` — users mode: 409
`This service is configured for named accounts; sign in with a username and password`.
Throttle on identity `pin`. Wrong: note failure, audit `login_failed` actor `pin`, 401
`Wrong PIN`. Right: clear, audit `login` actor `pin`, 200
`{"access_token", "token_type": "bearer", "user": {"name": "Operator", "role": "admin"}}`.

**`POST /auth/login`** `{"username", "password"}` — PIN mode: 409
`This service is configured for PIN sign-in`. Throttle on the username. Authenticate (§8).
Failure: note, audit `login_failed` with actor = the trimmed submitted username, 401
`Wrong username or password` — identical for unknown user, wrong password and disabled
account. Success: clear, audit `login` actor username, 200
`{"access_token", "token_type": "bearer", "user": <public user>, "must_change_password": bool}`.

**`GET /auth/me`** — `{"mode": <effective>, "user": {"username", "name", "role", "user_id",
"must_change_password"}}`; keys absent from the identity are `null` (in PIN mode:
username, user_id, must_change_password are `null`).

**`POST /auth/change-password`** `{"current_password", "new_password"}` — PIN mode: 409
`There are no accounts in PIN mode`. User gone: 401 `Session no longer valid`. Errors from
§8 → 400 with their message. Success: audit `password.change` actor username, target
`user`/id; 200 `{"status": "ok", "reauthenticate": true}`. **No new token** — the version
bump has just killed this one, and the user signs in again with the new password.

**`/users…`** — in PIN mode every route answers **404**
`User accounts are disabled (AUTH_MODE=pin)` (checked after the guard). Unknown id: 404
`No such user`. Rule violations: 400 with the message.

| Route | Body | Result | Audit (actor = acting admin's username) |
|---|---|---|---|
| `GET /users` | | `{"items": [public…] by id, "roles": ["viewer","operator","admin"], "password_rules": […]}` | — |
| `POST /users` | `username, password, role="viewer", display_name=""` | **201** public user, `must_change_password: true` | `user.create`, detail `<username> as <role>` |
| `PATCH /users/{id}` | `role?, display_name?, is_active?` (absent = unchanged) | 200 public user | `user.update`, detail `<username>: <changes>` — changes `role a->b`, `activated`/`deactivated`, joined `, `, or `profile` |
| `POST /users/{id}/password` | `new_password` | 200 public user, `must_change_password: true` | `user.password_reset`, detail username |
| `DELETE /users/{id}` | | **204** | `user.delete`, detail username |

All with `target_type: "user"`, `target_id: "<id>"`.

**`GET /users/audit/entries?limit=200&action=&actor=`** — both modes. `limit` clamped to
1…1000. `action` exact match; `actor` case-insensitive substring. Newest first.
`{"items": [entry…], "count": <len(items)>}`.

## 8. The account rules

**Public user** (the only shape that leaves the service — never the hash, never the
token version):
`{"id", "username", "display_name": display_name or username, "role", "is_active",
"must_change_password", "created_at", "last_login_at"}`, timestamps ISO-8601 UTC ending in
`Z` (the formatter the port already uses for documents), `last_login_at` may be `null`.

**Username:** trimmed; empty → `Username is required`; must match
`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` (ASCII only — `аdmin` with a Cyrillic а is a different
account that looks identical) else
`Username may contain only Latin letters, digits, '.', '_' and '-', must start with a letter or digit, and be at most 64 characters`.
Uniqueness is case-insensitive: `User '<name>' already exists`. Role not in the list:
`Unknown role '<role>'` (Python `repr` quoting — single quotes). Display name is trimmed
and may be anything.

**One write lock** (re-entrant) around every account mutation, and each mutation **re-reads
the account inside it**. Hash *before* taking the lock. The check "is this the last
admin" is only a check if it is atomic with the change.

- **create** — validate name, role, password; hash; lock; uniqueness; allocate id; store.
- **seed** — at startup in users mode, if there are **no users at all**: `ADMIN_USERNAME`,
  role `admin`, display name `Administrator`, `ADMIN_PASSWORD` hashed without the rules,
  `must_change_password: true`. A bad name or any error is logged
  (`[AUTH] could not seed the first administrator`), never fatal. Log
  `[AUTH] seeded the first administrator '<name>', password '1234' — …` **only when the
  password is the default**; otherwise `password from ADMIN_PASSWORD (not logged)`.
- **authenticate** — find by name (case-insensitive). Unknown → verify the password against
  a decoy hash computed once at startup (same cost, so timing does not reveal existence),
  return none. Verify **outside** the lock. Then lock, **re-read**; none if the account is
  gone, inactive, or its hash changed meanwhile; re-hash if `needs_rehash`; set
  `last_login_at`; store the fresh copy. Writing back the copy loaded before hashing would
  silently undo a demotion made during those ~80 ms.
- **change own password** — rules first; hash; lock; re-read; `current` must verify
  (`Current password is incorrect`); `new` must not verify against the current hash
  (`The new password must differ from the current one`); set hash,
  `must_change_password = false`, `token_version += 1`.
- **admin reset** — rules; hash; lock; re-read; set hash, `must_change_password = true`,
  `token_version += 1`.
- **update** — lock; re-read. Role change to non-admin of the last active admin →
  `Cannot demote the last active administrator`; deactivating it →
  `Cannot deactivate the last active administrator`. A role or active-flag change bumps
  `token_version`; a display-name change does not.
- **delete** — own account → `You cannot delete your own account` (checked first); lock;
  re-read; last active admin → `Cannot delete the last active administrator`.

"Last active admin" = the account is `admin` and active, and no **other** account is.

## 9. Storage

In the existing `FileStore`, next to `api_keys.json`:

- **`users.json`** — a JSON array ordered by id, written atomically (temp + rename) under the
  store lock, pretty-printed UTF-8 without ASCII escaping. Fields, exactly:
  `id, username, role, password_hash, display_name, is_active, must_change_password,
  token_version (starts at 1), created_at, last_login_at`.
- **`audit.jsonl`** — one JSON object per line, UTF-8: `id, action, actor, target_type,
  target_id (string), detail, at`. Kept in memory, capped at the **last 5000**; append a
  line per entry, rewrite the file from memory when trimming. Write it under the store lock.
  **A failed audit write is logged and swallowed** — it must never fail the action.
- **No personal data in the audit log** — no client address, no filename, no recognised
  value. The address feeds the throttle in memory only.
- Reads return **copies**; a caller mutating a returned user must not change the index.
- Unreadable `users.json` / `audit.jsonl` at startup: log and start empty, never crash.
- Add the methods to the store interface — `AllUsers, GetUser, FindUser (case-insensitive),
  NextUserID, PutUser, DropUser, AppendAudit, RecentAudit` — so a SQL backend is a body swap.

The startup wipe already exists in every port; it removes these two files with the rest.

## 10. The frontend

Nothing to change — `web/` already speaks this contract. It asks `/auth/config`, so a
service that implements it shows the account screens and one that does not shows the PIN
keypad.

## 11. What each port must test itself

The contract test sees HTTP only. The unit tests cover what it cannot, and each mirrors a
test in `tests/service/test_auth_security.py`:

- both interop vectors verify; wrong password and malformed strings (`""`, `$argon2i$…`,
  truncated, non-base64, absurd `m`) return false without throwing;
- a fresh hash round-trips and starts with `$argon2id$v=19$m=65536,t=3,p=4$`;
- password rules including Cyrillic (`пароль12` fails only `upper`; `Пароль1` — seven
  characters — fails only `length`; `Парольк1` passes), and length counted in code points
  (four emoji are four characters, not eight UTF-16 units);
- **the route table** of §5 against the actual router;
- **concurrent demotion of two admins cannot remove both** — with a forced interleaving
  (a barrier/latch inside the "last admin" count), not by hoping two threads collide; a
  plain two-thread race passes without the lock, which the first Python test proved;
- **a sign-in does not undo a demotion made while it was hashing** (demote between
  verify and write-back);
- store reads return copies;
- the throttle: per account, per address across rotated usernames, success clears only
  the account counter.

Then prove the tests: remove the lock / the re-read / the copy / the `uid` refusal /
the default-secret check, confirm the matching test fails, and put it back. A test that
passes both ways checks nothing.

## 12. Documentation per port

Add a section to the port's `DEVIATIONS.md` only for what genuinely differs (the Argon2
library row in §1, anything else found). Update the port README's service section with
`AUTH_MODE` and a pointer to `docs/auth-setup.md` — the operator guide applies unchanged.
