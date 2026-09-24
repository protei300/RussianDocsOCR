"""Black-box check of the authentication contract, against ANY of the four services.

    python -m conformance.auth_contract --port python
    python -m conformance.auth_contract --port go
    python -m conformance.auth_contract --port dotnet
    python -m conformance.auth_contract --port java
    python -m conformance.auth_contract --cmd "path/to/rdocs-service --addr :{port}"

Each run starts the service **several times** — once per scenario below — on a free
port with a fresh temporary DATA_DIR, drives it over HTTP only, and stops it. Nothing
here imports the service: the same checks grade the Python reference and every port,
which is the only way "the port behaves like the reference" becomes a measurement
rather than a claim. The reference must pass it first; see ports/AUTH.md.

Scenarios:

    pin          AUTH_MODE unset — the default, and the backward-compatible path
    pin-secret   AUTH_MODE=pin with a known JWT_SECRET, so tokens can be forged here
                 to prove which ones the service refuses
    downgrade    AUTH_MODE=bogus — must serve PIN and say why, never refuse to start
    users        AUTH_MODE=users — the whole account lifecycle, roles on every route,
                 token revocation, the last-admin rules, the audit log, the throttle
    users-admin  AUTH_MODE=users with a real ADMIN_PASSWORD — it must never be published

The Go, .NET and Kotlin presets expect their build to exist and the platform
environment to be in place (on Windows: dot-source ports/go/env.ps1 for Go; see each
port's README for the native-library variables). The runner only adds the variables
the scenario needs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import pathlib
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable

import httpx

REPO = pathlib.Path(__file__).resolve().parents[1]
API = "/api/v1"

#: The published default secret. Every implementation must refuse to SIGN with it.
PUBLISHED_DEFAULT_SECRET = "changeme-in-production"

#: How each implementation is started. {port} is substituted; {python} is this
#: interpreter. Paths are relative to the repository root.
PRESETS: dict[str, list[str]] = {
    "python": ["{python}", "-m", "uvicorn", "service.main:app", "--host", "127.0.0.1",
               "--port", "{port}", "--workers", "1"],
    "go": ["ports/go/bin/rdocs-service" + (".exe" if os.name == "nt" else ""),
           "--addr", "127.0.0.1:{port}"],
    "dotnet": ["ports/dotnet/src/RussianDocs.Service/bin/Release/net8.0/rdocs-service"
               + (".exe" if os.name == "nt" else ""), "--addr", "127.0.0.1:{port}"],
    "java": ["java", "-jar", "ports/java/service/build/dist/rdocs-service.jar",
             "--addr", "127.0.0.1:{port}"],
}

#: Small limits so the throttle checks finish in seconds, not minutes.
MAX_ATTEMPTS = 3
LOCKOUT_SECONDS = 120

STRONG = "Str0ng-Pass"          # satisfies every rule
STRONG2 = "An0ther-Pass"
ADDRESS_FACTOR = 3


# --------------------------------------------------------------------------- tokens

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def forge(claims: dict[str, Any], secret: str, alg: str = "HS256") -> str:
    header = _b64(json.dumps({"alg": alg, "typ": "JWT"}).encode())
    claims = {"exp": int(time.time()) + 3600, **claims}
    body = _b64(json.dumps(claims).encode())
    signing = f"{header}.{body}"
    if alg == "none":
        return signing + "."
    sig = hmac.new(secret.encode(), signing.encode(), hashlib.sha256).digest()
    return f"{signing}.{_b64(sig)}"


def claims_of(token: str) -> dict[str, Any]:
    body = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))


# --------------------------------------------------------------------------- harness

class Failure(AssertionError):
    pass


class Report:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []

    def check(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Failure as error:
            self.failed.append((name, str(error)))
            print(f"    FAIL  {name}\n          {error}")
        except Exception as error:                              # noqa: BLE001
            self.failed.append((name, f"{type(error).__name__}: {error}"))
            print(f"    ERROR {name}\n          {type(error).__name__}: {error}")
            traceback.print_exc(limit=2)
        else:
            self.passed.append(name)
            print(f"    ok    {name}")


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def expect_status(response: httpx.Response, status: int, what: str) -> None:
    if response.status_code != status:
        raise Failure(f"{what}: expected HTTP {status}, got {response.status_code} "
                      f"{response.text[:300]!r}")


def detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", ""))
    except Exception:                                           # noqa: BLE001
        return response.text


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Service:
    """One running instance with its own empty data directory."""

    def __init__(self, cmd: list[str], env: dict[str, str], label: str) -> None:
        self.port = free_port()
        self.data_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"rdocs-auth-{label}-"))
        self.log_path = self.data_dir.parent / f"{self.data_dir.name}.log"
        argv = [part.replace("{port}", str(self.port)).replace("{python}", sys.executable)
                for part in cmd]
        if not os.path.isabs(argv[0]) and (REPO / argv[0]).exists():
            argv[0] = str(REPO / argv[0])
        full_env = {**os.environ,
                    "DATA_DIR": str(self.data_dir / "data"),
                    "DATA_WIPE_ON_START": "true",
                    "SEED_SAMPLES": "-1",
                    "WARMUP_IMAGE": "",
                    "LOGIN_MAX_ATTEMPTS": str(MAX_ATTEMPTS),
                    "LOGIN_LOCKOUT_SECONDS": str(LOCKOUT_SECONDS),
                    **env}
        for name in ("AUTH_MODE", "ADMIN_PASSWORD", "ADMIN_USERNAME", "JWT_SECRET"):
            if name not in env:
                full_env.pop(name, None)
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(argv, cwd=REPO, env=full_env, stdout=self.log,
                                     stderr=subprocess.STDOUT)
        self.base = f"http://127.0.0.1:{self.port}"
        # trust_env=False: a corporate HTTP_PROXY would otherwise route 127.0.0.1 through it.
        self.http = httpx.Client(base_url=self.base, timeout=60, trust_env=False)
        self._wait()

    def _wait(self, seconds: float = 180) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"service exited with {self.proc.returncode}; "
                                   f"log: {self.log_path}\n{self.log_text()[-3000:]}")
            try:
                if self.http.get("/health").status_code == 200 and \
                        self.http.get(f"{API}/auth/config").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise RuntimeError(f"service did not become ready; log: {self.log_path}")

    def log_text(self) -> str:
        self.log.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def stop(self) -> None:
        self.http.close()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=20)
        self.log.close()
        shutil.rmtree(self.data_dir, ignore_errors=True)

    # -- conveniences -------------------------------------------------------
    def get(self, path: str, token: str | None = None, **kw) -> httpx.Response:
        return self.http.get(API + path, headers=_auth(token, kw.pop("headers", None)), **kw)

    def post(self, path: str, token: str | None = None, **kw) -> httpx.Response:
        return self.http.post(API + path, headers=_auth(token, kw.pop("headers", None)), **kw)

    def patch(self, path: str, token: str | None = None, **kw) -> httpx.Response:
        return self.http.patch(API + path, headers=_auth(token, kw.pop("headers", None)), **kw)

    def put(self, path: str, token: str | None = None, **kw) -> httpx.Response:
        return self.http.put(API + path, headers=_auth(token, kw.pop("headers", None)), **kw)

    def delete(self, path: str, token: str | None = None, **kw) -> httpx.Response:
        return self.http.delete(API + path, headers=_auth(token, kw.pop("headers", None)), **kw)

    def login(self, username: str, password: str) -> httpx.Response:
        return self.post("/auth/login", json={"username": username, "password": password})


def _auth(token: str | None, extra: dict | None) -> dict:
    headers = dict(extra or {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# --------------------------------------------------------------------------- scenarios

def scenario_pin(svc: Service, r: Report, *, secret: str | None) -> None:
    state: dict[str, Any] = {}

    def config():
        resp = svc.get("/auth/config")
        expect_status(resp, 200, "GET /auth/config")
        body = resp.json()
        expect(body.get("mode") == "pin", f"mode {body.get('mode')!r}")
        expect(body.get("pin_required") is True, "pin_required must be true")
        expect(body.get("users_enabled") is False, "users_enabled must be false")
        expect(body.get("downgrade_reason") is None, f"downgrade_reason {body.get('downgrade_reason')!r}")
        expect("password_rules" not in body and "demo_credentials" not in body,
               f"PIN mode must not advertise account details: {sorted(body)}")
    r.check("config reports PIN mode", config)

    def health_public():
        expect_status(svc.http.get("/health"), 200, "GET /health without a token")
    r.check("/health is public", health_public)

    def wrong_pin():
        resp = svc.post("/auth/pin-login", json={"pin": "0000"})
        expect_status(resp, 401, "wrong PIN")
        expect(detail(resp) == "Wrong PIN", f"detail {detail(resp)!r}")
    r.check("a wrong PIN is refused", wrong_pin)

    def login_route_refused():
        resp = svc.login("admin", "1234")
        expect_status(resp, 409, "POST /auth/login in PIN mode")
    r.check("named sign-in answers 409 in PIN mode", login_route_refused)

    def pin_login():
        resp = svc.post("/auth/pin-login", json={"pin": "1234"})
        expect_status(resp, 200, "PIN 1234")
        body = resp.json()
        expect(body.get("token_type") == "bearer", f"token_type {body.get('token_type')!r}")
        expect(body.get("user") == {"name": "Operator", "role": "admin"}, f"user {body.get('user')!r}")
        claims = claims_of(body["access_token"])
        expect(claims.get("role") == "admin", f"PIN token role claim {claims.get('role')!r}")
        expect(claims.get("sub") == "operator", f"PIN token sub {claims.get('sub')!r}")
        expect("uid" not in claims and "tv" not in claims, f"PIN token carries account claims: {claims}")
        expect(isinstance(claims.get("exp"), int), "exp must be an integer")
        state["token"] = body["access_token"]
    r.check("PIN 1234 signs in as the administrator", pin_login)

    def pin_session_works():
        token = state["token"]
        for path in ("/status", "/documents", "/api-keys", "/settings", "/logs"):
            expect_status(svc.get(path, token), 200, f"GET {path} with the PIN session")
        me = svc.get("/auth/me", token)
        expect_status(me, 200, "GET /auth/me")
        expect(me.json().get("mode") == "pin", f"/auth/me mode {me.json().get('mode')!r}")
    r.check("the PIN session reaches every management route", pin_session_works)

    def no_users_in_pin_mode():
        token = state["token"]
        # Valid bodies, so the answer is about the mode and not about validation.
        for method, path, body in (
                ("GET", "/users", None),
                ("POST", "/users", {"username": "petrov", "password": STRONG, "role": "viewer"}),
                ("PATCH", "/users/1", {"role": "viewer"}),
                ("POST", "/users/1/password", {"new_password": STRONG}),
                ("DELETE", "/users/1", None)):
            kw = {"json": body} if body is not None else {}
            resp = svc.http.request(method, API + path, headers=_auth(token, None), **kw)
            expect(resp.status_code == 404, f"{method} {path}: expected 404, got {resp.status_code}")
        resp = svc.post("/auth/change-password", token,
                        json={"current_password": "x", "new_password": STRONG})
        expect_status(resp, 409, "change-password in PIN mode")
    r.check("user management does not exist in PIN mode (404)", no_users_in_pin_mode)

    def audit_in_pin_mode():
        resp = svc.get("/users/audit/entries", state["token"])
        expect_status(resp, 200, "GET /users/audit/entries in PIN mode")
        items = resp.json()["items"]
        actions = [(e["action"], e["actor"]) for e in items]
        expect(("login", "pin") in actions and ("login_failed", "pin") in actions,
               f"audit should hold the PIN login and the failed attempt: {actions}")
        expect(items[0]["id"] > items[-1]["id"], "audit must be newest first")
    r.check("the action log works in PIN mode, actor 'pin'", audit_in_pin_mode)

    def published_secret_cannot_sign():
        forged = forge({"sub": "operator", "name": "Operator", "role": "admin"},
                       PUBLISHED_DEFAULT_SECRET)
        expect_status(svc.get("/status", forged), 401, "token signed with the published default")
        expect_status(svc.get("/status", forge({"sub": "operator"}, "", alg="none")), 401,
                      "alg=none token")
        expect_status(svc.get("/status", "not.a.token"), 401, "garbage token")
    if secret is None:
        r.check("the published default secret signs nothing", published_secret_cannot_sign)

    if secret is not None:
        def secret_is_used():
            ok = forge({"sub": "operator", "name": "Operator", "role": "admin"}, secret)
            expect_status(svc.get("/status", ok), 200, "PIN token signed with JWT_SECRET")
        r.check("JWT_SECRET, when set, is the signing secret", secret_is_used)

        def account_token_refused():
            forged = forge({"sub": "viewer1", "uid": 2, "tv": 1, "role": "viewer",
                            "name": "viewer1"}, secret)
            expect_status(svc.get("/status", forged), 401,
                          "a named-account token presented to a PIN-mode service")
        r.check("PIN mode REFUSES a token carrying uid", account_token_refused)

        def wrong_alg_refused():
            forged = forge({"sub": "operator", "role": "admin"}, secret, alg="HS512")
            expect_status(svc.get("/status", forged), 401, "token whose header says HS512")
        r.check("the algorithm is pinned, not read from the token", wrong_alg_refused)

    def pin_throttle():
        # MAX_ATTEMPTS failures, then even the right PIN is refused with 429.
        for _ in range(MAX_ATTEMPTS):
            svc.post("/auth/pin-login", json={"pin": "9999"})
        resp = svc.post("/auth/pin-login", json={"pin": "1234"})
        expect_status(resp, 429, "PIN after too many failures")
        expect(resp.headers.get("Retry-After", "").isdigit(), "429 must carry Retry-After in seconds")
        expect(detail(resp).startswith("Too many attempts. Try again in "), f"detail {detail(resp)!r}")
    r.check("PIN guessing is throttled", pin_throttle)


def scenario_downgrade(svc: Service, r: Report) -> None:
    def downgraded():
        body = svc.get("/auth/config").json()
        expect(body.get("mode") == "pin", f"mode {body.get('mode')!r}")
        reason = body.get("downgrade_reason") or ""
        expect("bogus" in reason and "PIN" in reason,
               f"downgrade_reason must name the bad value and the fallback: {reason!r}")
        resp = svc.post("/auth/pin-login", json={"pin": "1234"})
        expect_status(resp, 200, "PIN sign-in on a downgraded service")
    r.check("a typo in AUTH_MODE serves PIN and says why", downgraded)

    def logged():
        expect("bogus" in svc.log_text(), "the downgrade must be in the startup log")
    r.check("the downgrade is logged", logged)


def scenario_users_admin_password(svc: Service, r: Report, admin_password: str) -> None:
    def not_published():
        body = svc.get("/auth/config").json()
        expect(body.get("mode") == "users", f"mode {body.get('mode')!r}")
        expect("demo_credentials" not in body, f"a real ADMIN_PASSWORD was published: {body}")
        expect(admin_password not in json.dumps(body), "ADMIN_PASSWORD appears in /auth/config")
    r.check("a real ADMIN_PASSWORD is not on the login page", not_published)

    def not_logged():
        expect(admin_password not in svc.log_text(), "ADMIN_PASSWORD appears in the log")
    r.check("a real ADMIN_PASSWORD is not in the log", not_logged)

    def works_and_must_change():
        resp = svc.login("admin", admin_password)
        expect_status(resp, 200, "sign-in with ADMIN_PASSWORD")
        expect(resp.json().get("must_change_password") is True, "seeded admin must change password")
        expect_status(svc.login("admin", "1234"), 401, "the demo password when ADMIN_PASSWORD is set")
    r.check("the configured password works and must be changed", works_and_must_change)


def scenario_users(svc: Service, r: Report) -> None:
    s: dict[str, Any] = {}

    # ---------------------------------------------------------------- config
    def config():
        body = svc.get("/auth/config").json()
        expect(body.get("mode") == "users", f"mode {body.get('mode')!r}")
        expect(body.get("pin_required") is False and body.get("users_enabled") is True,
               f"flags {body}")
        expect(body.get("downgrade_reason") is None, f"downgrade_reason {body.get('downgrade_reason')!r}")
        rules = body.get("password_rules") or []
        expect([x.get("code") for x in rules] == ["length", "digit", "letter", "upper"],
               f"password_rules codes {[x.get('code') for x in rules]}")
        expect(all(x.get("label") and x.get("pattern") for x in rules), "each rule needs label+pattern")
        expect(body.get("demo_credentials") == {"username": "admin", "password": "1234"},
               f"demo_credentials {body.get('demo_credentials')!r}")
        s["rules"] = rules
    r.check("config reports named accounts, rules and the demo credentials", config)

    def pin_refused():
        expect_status(svc.post("/auth/pin-login", json={"pin": "1234"}), 409,
                      "PIN sign-in in users mode")
    r.check("PIN sign-in answers 409 in users mode", pin_refused)

    # ---------------------------------------------------------------- first sign-in
    def wrong_credentials_identical():
        a = svc.login("admin", "wrong")
        b = svc.login("nobody-here", "wrong")
        expect_status(a, 401, "wrong password")
        expect_status(b, 401, "unknown user")
        expect(detail(a) == detail(b) == "Wrong username or password",
               f"replies differ or wrong text: {detail(a)!r} vs {detail(b)!r}")
    r.check("wrong password and unknown user get the same reply", wrong_credentials_identical)

    def first_login():
        resp = svc.login("admin", "1234")
        expect_status(resp, 200, "admin / 1234")
        body = resp.json()
        expect(body.get("must_change_password") is True, "first sign-in must require a change")
        expect(body.get("token_type") == "bearer", "token_type")
        user = body.get("user") or {}
        expect(user.get("username") == "admin" and user.get("role") == "admin", f"user {user}")
        expect("password_hash" not in user and "token_version" not in user,
               f"the public user leaks internals: {sorted(user)}")
        claims = claims_of(body["access_token"])
        expect(isinstance(claims.get("uid"), int) and isinstance(claims.get("tv"), int),
               f"account token must carry integer uid and tv: {claims}")
        expect(claims.get("role") == "admin" and claims.get("sub") == "admin", f"claims {claims}")
        s["restricted"] = body["access_token"]
    r.check("admin / 1234 signs in, restricted, must change password", first_login)

    def restricted_everywhere():
        t = s["restricted"]
        for path in ("/documents", "/status", "/users", "/api-keys", "/settings", "/logs",
                     "/users/audit/entries"):
            resp = svc.get(path, t)
            expect(resp.status_code == 403 and detail(resp) == "password_change_required",
                   f"GET {path} with a restricted session: {resp.status_code} {detail(resp)!r}")
        me = svc.get("/auth/me", t)
        expect_status(me, 200, "/auth/me with a restricted session")
        user = me.json().get("user", {})
        expect(me.json().get("mode") == "users", "me.mode")
        expect(user.get("must_change_password") is True and user.get("username") == "admin",
               f"/auth/me user {user}")
    r.check("a restricted session reaches only /auth/me and change-password", restricted_everywhere)

    def change_rules():
        t = s["restricted"]
        bad_current = svc.post("/auth/change-password", t,
                               json={"current_password": "nope", "new_password": STRONG})
        expect_status(bad_current, 400, "wrong current password")
        expect(detail(bad_current) == "Current password is incorrect", f"detail {detail(bad_current)!r}")
        weak = svc.post("/auth/change-password", t,
                        json={"current_password": "1234", "new_password": "weakpass"})
        expect_status(weak, 400, "weak new password")
        expect(detail(weak) == "Password needs: at least one digit, at least one capital letter",
               f"detail {detail(weak)!r}")
        cyr = svc.post("/auth/change-password", t,
                       json={"current_password": "1234", "new_password": "пароль12"})
        expect(detail(cyr) == "Password needs: at least one capital letter",
               f"Cyrillic letters must count as letters: {detail(cyr)!r}")
    r.check("password rules and the current-password check", change_rules)

    def change_ok():
        t = s["restricted"]
        resp = svc.post("/auth/change-password", t,
                        json={"current_password": "1234", "new_password": STRONG})
        expect_status(resp, 200, "change to a strong password")
        expect(resp.json() == {"status": "ok", "reauthenticate": True}, f"body {resp.json()}")
        expect_status(svc.get("/auth/me", t), 401, "the old token after a password change")
        expect_status(svc.login("admin", "1234"), 401, "the old password after the change")
        cfg = svc.get("/auth/config").json()
        expect("demo_credentials" not in cfg, "demo credentials must vanish once changed")
        resp = svc.login("admin", STRONG)
        expect_status(resp, 200, "sign-in with the new password")
        expect(resp.json().get("must_change_password") is False, "no further change required")
        s["admin"] = resp.json()["access_token"]
        s["admin_id"] = resp.json()["user"]["id"]
    r.check("changing the password ends the session; the new one works", change_ok)

    def same_password_refused():
        resp = svc.post("/auth/change-password", s["admin"],
                        json={"current_password": STRONG, "new_password": STRONG})
        expect_status(resp, 400, "new == current")
        expect(detail(resp) == "The new password must differ from the current one", detail(resp))
    r.check("the new password must differ", same_password_refused)

    # ---------------------------------------------------------------- user management
    def list_users():
        resp = svc.get("/users", s["admin"])
        expect_status(resp, 200, "GET /users")
        body = resp.json()
        expect(body.get("roles") == ["viewer", "operator", "admin"], f"roles {body.get('roles')}")
        expect([x["code"] for x in body.get("password_rules", [])] ==
               ["length", "digit", "letter", "upper"], "password_rules on /users")
        items = body.get("items", [])
        expect(len(items) == 1, f"one seeded user expected, got {len(items)}")
        want = {"id", "username", "display_name", "role", "is_active", "must_change_password",
                "created_at", "last_login_at"}
        expect(set(items[0]) == want, f"public user fields {sorted(items[0])} != {sorted(want)}")
        expect(items[0]["display_name"] == "Administrator", f"display_name {items[0]['display_name']!r}")
        expect(items[0]["created_at"].endswith("Z") and items[0]["last_login_at"].endswith("Z"),
               f"timestamps must be ISO UTC with Z: {items[0]['created_at']!r}")
    r.check("the user list: shape, roles, rules, no internals", list_users)

    def create_validation():
        a = s["admin"]
        cases = [
            ({"username": "аdmin", "password": STRONG, "role": "viewer"},  # Cyrillic а
             "Username may contain only Latin letters"),
            ({"username": " ", "password": STRONG, "role": "viewer"}, None),
            ({"username": "petrov", "password": "short1A", "role": "viewer"},
             "Password needs: at least 8 characters"),
            ({"username": "petrov", "password": STRONG, "role": "root"}, "Unknown role 'root'"),
            ({"username": "ADMIN", "password": STRONG, "role": "viewer"},
             "User 'ADMIN' already exists"),
        ]
        for body, prefix in cases:
            resp = svc.post("/users", a, json=body)
            expect(resp.status_code in (400, 422),
                   f"create {body['username']!r}/{body['role']}: expected 400, got {resp.status_code}")
            if prefix:
                expect(detail(resp).startswith(prefix), f"detail {detail(resp)!r} !~ {prefix!r}")
    r.check("user creation refuses bad names, weak passwords, unknown roles, duplicates",
            create_validation)

    def create_users():
        a = s["admin"]
        for name, role in (("viewer1", "viewer"), ("op1", "operator"), ("admin2", "admin")):
            resp = svc.post("/users", a, json={"username": name, "password": STRONG, "role": role,
                                                "display_name": f"Иван {name}"})
            expect_status(resp, 201, f"create {name}")
            body = resp.json()
            expect(body["role"] == role and body["must_change_password"] is True and
                   body["is_active"] is True, f"created {body}")
            expect(body["display_name"] == f"Иван {name}", f"display name round-trip {body['display_name']!r}")
            s[f"{name}_id"] = body["id"]
    r.check("an administrator creates viewer, operator and admin accounts", create_users)

    def first_login_of(name: str) -> str:
        resp = svc.login(name, STRONG)
        expect_status(resp, 200, f"{name} first sign-in")
        expect(resp.json()["must_change_password"] is True, f"{name} must change password")
        resp = svc.post("/auth/change-password", resp.json()["access_token"],
                        json={"current_password": STRONG, "new_password": STRONG2})
        expect_status(resp, 200, f"{name} changes password")
        resp = svc.login(name, STRONG2)
        expect_status(resp, 200, f"{name} second sign-in")
        return resp.json()["access_token"]

    def onboard():
        s["viewer"] = first_login_of("viewer1")
        s["op"] = first_login_of("op1")
        s["admin2"] = first_login_of("admin2")
    r.check("each new account changes its password at first sign-in", onboard)

    # ---------------------------------------------------------------- roles on every route
    def viewer_scope():
        v = s["viewer"]
        for path in ("/documents", "/status", "/documents/999999/progress"):
            resp = svc.get(path, v)
            expect(resp.status_code in (200, 404), f"viewer GET {path}: {resp.status_code}")
        forbidden = [
            ("DELETE", "/documents/1", None, "operator"),
            ("POST", "/documents/1/reprocess", None, "operator"),
            ("POST", "/documents/purge", None, "admin"),
            ("GET", "/api-keys", None, "admin"),
            ("POST", "/api-keys", {"label": "mine"}, "admin"),
            ("GET", "/settings", None, "admin"),
            ("PUT", "/settings", {}, "admin"),
            ("GET", "/logs", None, "admin"),
            ("GET", "/users", None, "admin"),
            ("GET", "/users/audit/entries", None, "admin"),
        ]
        for method, path, body, role in forbidden:
            kw = {"json": body} if body is not None else {}
            resp = svc.http.request(method, API + path, headers=_auth(v, None), **kw)
            expect(resp.status_code == 403 and detail(resp) == f"This action requires the {role} role",
                   f"viewer {method} {path}: {resp.status_code} {detail(resp)!r}")
        up = svc.post("/documents", v, files={"file": ("x.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")})
        expect(up.status_code == 403, f"viewer upload: {up.status_code} {detail(up)!r}")
    r.check("a viewer reads documents and nothing else", viewer_scope)

    def operator_scope():
        o = s["op"]
        resp = svc.delete("/documents/999999", o)
        expect(resp.status_code == 404, f"operator DELETE passes the guard (404 expected): {resp.status_code}")
        for method, path, body in (("POST", "/documents/purge", None), ("GET", "/api-keys", None),
                                   ("GET", "/users", None), ("GET", "/settings", None)):
            kw = {"json": body} if body is not None else {}
            resp = svc.http.request(method, API + path, headers=_auth(o, None), **kw)
            expect(resp.status_code == 403 and detail(resp) == "This action requires the admin role",
                   f"operator {method} {path}: {resp.status_code} {detail(resp)!r}")
    r.check("an operator writes documents but manages nothing", operator_scope)

    def api_key_scope():
        resp = svc.post("/api-keys", s["admin"], json={"label": "integration"})
        expect_status(resp, 201 if resp.status_code == 201 else 200, "admin mints a key")
        key = resp.json().get("key")
        expect(bool(key), f"minted key missing: {resp.json()}")
        hk = {"X-API-Key": key}
        expect_status(svc.get("/documents", headers=hk), 200, "API key on the document list")
        for path in ("/users", "/api-keys", "/settings", "/logs", "/status", "/users/audit/entries"):
            resp = svc.get(path, headers=hk)
            expect(resp.status_code == 401, f"API key on {path}: expected 401, got {resp.status_code}")
    r.check("an API key reaches documents only", api_key_scope)

    # ---------------------------------------------------------------- revocation
    def demotion_revokes():
        before = svc.get("/documents", s["op"])
        expect_status(before, 200, "operator before demotion")
        resp = svc.patch(f"/users/{s['op1_id']}", s["admin"], json={"role": "viewer"})
        expect_status(resp, 200, "demote op1")
        expect(resp.json()["role"] == "viewer", f"role after {resp.json()['role']}")
        expect_status(svc.get("/documents", s["op"]), 401, "op1's token after demotion")
        s["op"] = svc.login("op1", STRONG2).json()["access_token"]
        resp = svc.delete("/documents/999999", s["op"])
        expect(resp.status_code == 403, f"demoted op1 still writes: {resp.status_code}")
    r.check("a role change revokes the account's tokens at once", demotion_revokes)

    def profile_edit_keeps_tokens():
        resp = svc.patch(f"/users/{s['viewer1_id']}", s["admin"], json={"display_name": "  Viewer One "})
        expect_status(resp, 200, "rename viewer1")
        expect(resp.json()["display_name"] == "Viewer One", f"display_name {resp.json()['display_name']!r}")
        expect_status(svc.get("/documents", s["viewer"]), 200, "viewer token after a rename")
    r.check("a display-name edit does not revoke tokens", profile_edit_keeps_tokens)

    def reset_revokes():
        resp = svc.post(f"/users/{s['viewer1_id']}/password", s["admin"], json={"new_password": "weak"})
        expect_status(resp, 400, "reset to a weak password")
        resp = svc.post(f"/users/{s['viewer1_id']}/password", s["admin"],
                        json={"new_password": "Reset-Pass9"})
        expect_status(resp, 200, "reset viewer1")
        expect(resp.json()["must_change_password"] is True, "reset must force a change")
        expect_status(svc.get("/documents", s["viewer"]), 401, "viewer token after a reset")
        resp = svc.login("viewer1", "Reset-Pass9")
        expect(resp.status_code == 200 and resp.json()["must_change_password"] is True,
               f"sign-in after reset {resp.status_code}")
    r.check("an administrator's reset revokes tokens and forces a change", reset_revokes)

    def deactivate():
        resp = svc.patch(f"/users/{s['viewer1_id']}", s["admin"], json={"is_active": False})
        expect_status(resp, 200, "deactivate viewer1")
        expect(resp.json()["is_active"] is False, "is_active")
        resp = svc.login("viewer1", "Reset-Pass9")
        expect_status(resp, 401, "a disabled account signs in")
        expect(detail(resp) == "Wrong username or password", f"disabled reply differs: {detail(resp)!r}")
    r.check("a disabled account cannot sign in, and says nothing more", deactivate)

    def last_admin_rules():
        a = s["admin"]
        resp = svc.delete(f"/users/{s['admin_id']}", a)
        expect(resp.status_code == 400 and detail(resp) == "You cannot delete your own account",
               f"self-delete {resp.status_code} {detail(resp)!r}")
        # Remove the second admin so admin is the last one.
        resp = svc.patch(f"/users/{s['admin2_id']}", a, json={"role": "operator"})
        expect_status(resp, 200, "demote admin2 while two admins exist")
        resp = svc.patch(f"/users/{s['admin_id']}", a, json={"role": "viewer"})
        expect(resp.status_code == 400 and detail(resp) == "Cannot demote the last active administrator",
               f"demote last admin {resp.status_code} {detail(resp)!r}")
        resp = svc.patch(f"/users/{s['admin_id']}", a, json={"is_active": False})
        expect(resp.status_code == 400 and detail(resp) == "Cannot deactivate the last active administrator",
               f"deactivate last admin {resp.status_code} {detail(resp)!r}")
        resp = svc.patch(f"/users/{s['admin_id']}", a, json={"role": "boss"})
        expect(resp.status_code == 400 and detail(resp) == "Unknown role 'boss'", detail(resp))
        expect_status(svc.get("/users", a), 200, "admin still administers")
    r.check("the last active administrator cannot be removed", last_admin_rules)

    def delete_user():
        a = s["admin"]
        resp = svc.delete(f"/users/{s['op1_id']}", a)
        expect_status(resp, 204, "delete op1")
        expect_status(svc.login("op1", STRONG2), 401, "a deleted account signs in")
        expect_status(svc.get("/documents", s["op"]), 401, "a deleted account's token")
        for method, path, body in (("PATCH", "/users/999999", {"role": "viewer"}),
                                   ("DELETE", "/users/999999", None),
                                   ("POST", "/users/999999/password", {"new_password": STRONG})):
            kw = {"json": body} if body is not None else {}
            resp = svc.http.request(method, API + path, headers=_auth(a, None), **kw)
            expect(resp.status_code == 404 and detail(resp) == "No such user",
                   f"{method} {path}: {resp.status_code} {detail(resp)!r}")
    r.check("deleting an account ends it; unknown ids are 404", delete_user)

    # ---------------------------------------------------------------- tokens
    def forged_tokens():
        tv_guess = forge({"sub": "admin", "uid": s["admin_id"], "tv": 2, "role": "admin",
                          "name": "Administrator"}, PUBLISHED_DEFAULT_SECRET)
        expect_status(svc.get("/users", tv_guess), 401, "admin token forged with the published secret")
        expect_status(svc.get("/users", forge({"uid": 1, "tv": 2}, "", alg="none")), 401, "alg=none")
        pin_style = claims_of(s["admin"])
        pin_style.pop("uid", None)
        expect_status(svc.get("/users", forge(pin_style, PUBLISHED_DEFAULT_SECRET)), 401,
                      "a PIN-era token in users mode")
    r.check("forged and PIN-era tokens are refused", forged_tokens)

    # ---------------------------------------------------------------- the audit log
    def audit():
        resp = svc.get("/users/audit/entries", s["admin"], params={"limit": 1000})
        expect_status(resp, 200, "GET /users/audit/entries")
        body = resp.json()
        items = body["items"]
        expect(body.get("count") == len(items), "count must equal len(items)")
        actions = {e["action"] for e in items}
        for wanted in ("login", "login_failed", "password.change", "user.create", "user.update",
                       "user.password_reset", "user.delete"):
            expect(wanted in actions, f"audit is missing {wanted!r}: {sorted(actions)}")
        fields = {"id", "action", "actor", "target_type", "target_id", "detail", "at"}
        expect(all(set(e) == fields for e in items), f"audit entry fields {sorted(items[0])}")
        ids = [e["id"] for e in items]
        expect(ids == sorted(ids, reverse=True), "audit must be newest first")
        text = json.dumps(items, ensure_ascii=False)
        expect("127.0.0.1" not in text, "the audit log holds a client address")
        for secret in ("1234", STRONG, STRONG2, "Reset-Pass9", "wrong"):
            expect(f'"{secret}"' not in text and f" {secret}" not in text,
                   f"a password appears in the audit log: {secret!r}")
        created = [e for e in items if e["action"] == "user.create"]
        expect(any(e["detail"] == "viewer1 as viewer" and e["target_type"] == "user" and
                   e["actor"] == "admin" for e in created), f"user.create detail {created[:1]}")
        failed = [e for e in items if e["action"] == "login_failed"]
        expect(any(e["actor"] == "nobody-here" for e in failed), "failed sign-in records the username")
        limited = svc.get("/users/audit/entries", s["admin"], params={"limit": 2}).json()
        expect(len(limited["items"]) == 2, "limit")
        only = svc.get("/users/audit/entries", s["admin"], params={"action": "user.delete"}).json()
        expect(only["items"] and all(e["action"] == "user.delete" for e in only["items"]), "action filter")
        by = svc.get("/users/audit/entries", s["admin"], params={"actor": "VIEWER"}).json()
        expect(by["items"] and all("viewer" in e["actor"].lower() for e in by["items"]),
               "actor filter is a case-insensitive substring")
    r.check("the audit log: every action, newest first, nothing personal", audit)

    # ---------------------------------------------------------------- storage
    def storage():
        data = svc.data_dir / "data"
        users_file = data / "users.json"
        expect(users_file.is_file(), f"users.json not at {users_file}")
        rows = json.loads(users_file.read_text("utf-8"))
        expect(isinstance(rows, list) and rows, "users.json must be a JSON array")
        want = {"id", "username", "role", "password_hash", "display_name", "is_active",
                "must_change_password", "token_version", "created_at", "last_login_at"}
        expect(set(rows[0]) == want, f"users.json fields {sorted(rows[0])}")
        expect(all(u["password_hash"].startswith("$argon2id$v=19$m=65536,t=3,p=4$") for u in rows),
               "every hash must be an Argon2id PHC string with the OWASP parameters")
        expect((data / "audit.jsonl").is_file(), "audit.jsonl missing")
        blob = b"".join(p.read_bytes() for p in data.rglob("*") if p.is_file())
        for secret in (STRONG, STRONG2, "Reset-Pass9"):
            expect(secret.encode() not in blob, f"a plaintext password is on disk: {secret!r}")
    r.check("users.json holds Argon2id PHC hashes and no plaintext", storage)

    # ---------------------------------------------------------------- throttle (last: it locks the address)
    def throttle_per_account():
        for _ in range(MAX_ATTEMPTS):
            svc.login("admin", "not-it")
        resp = svc.login("admin", STRONG)
        expect_status(resp, 429, "the right password after too many failures")
        expect(resp.headers.get("Retry-After", "").isdigit(), "Retry-After")
        expect(detail(resp).startswith("Too many attempts. Try again in "), detail(resp))
    r.check("failed sign-ins lock the account from this address", throttle_per_account)

    def throttle_per_address():
        # Rotating usernames must not escape: 3x the per-account budget across names.
        blocked = False
        for i in range(MAX_ATTEMPTS * ADDRESS_FACTOR + 1):
            resp = svc.login(f"spray{i}", "guess")
            if resp.status_code == 429:
                blocked = True
                break
        expect(blocked, "rotating usernames was never throttled")
        expect_status(svc.login("admin2", STRONG2), 429, "a different account from the same address")
    r.check("rotating usernames does not escape the address lockout", throttle_per_address)


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", choices=sorted(PRESETS), help="an implementation preset")
    parser.add_argument("--cmd", help="a command line with {port}, instead of a preset")
    parser.add_argument("--only", nargs="*", help="run only these scenarios")
    args = parser.parse_args()
    if not args.port and not args.cmd:
        parser.error("give --port or --cmd")
    cmd = shlex.split(args.cmd, posix=os.name != "nt") if args.cmd else PRESETS[args.port]

    secret = "contract-test-secret-" + "x" * 32
    real_admin_password = "Real-Admin-Secret-77"
    scenarios = [
        ("pin", {}, lambda s, r: scenario_pin(s, r, secret=None)),
        ("pin-secret", {"AUTH_MODE": "pin", "JWT_SECRET": secret},
         lambda s, r: scenario_pin(s, r, secret=secret)),
        ("downgrade", {"AUTH_MODE": "bogus"}, scenario_downgrade),
        ("users", {"AUTH_MODE": "users"}, scenario_users),
        ("users-admin", {"AUTH_MODE": "users", "ADMIN_PASSWORD": real_admin_password},
         lambda s, r: scenario_users_admin_password(s, r, real_admin_password)),
    ]
    report = Report()
    for name, env, run in scenarios:
        if args.only and name not in args.only:
            continue
        print(f"\n== {name}  {env}")
        try:
            svc = Service(cmd, env, name)
        except Exception as error:                              # noqa: BLE001
            report.failed.append((f"{name}: start", str(error)))
            print(f"    ERROR could not start: {error}")
            continue
        try:
            run(svc, report)
        finally:
            svc.stop()

    print(f"\n{len(report.passed)} passed, {len(report.failed)} failed")
    for name, why in report.failed:
        print(f"  - {name}: {why}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
