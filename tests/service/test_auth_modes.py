"""Authentication: the two modes, and the guard that outlives this file.

The test that matters most here is ``test_every_route_declares_the_role_it_needs``.
Everything else checks behaviour that exists today; that one checks behaviour that
has to keep existing — it walks the router and fails when a route is added
without a decision about who may call it, or when a route's guard changes without
the table changing with it. The attacks themselves — a viewer deleting, a forged
token, a leaked password — are exercised in ``test_auth_security.py``.

The suite runs the app twice, once per mode, through ``TestClient`` with a
throwaway ``DATA_DIR``. Nothing here touches a real data directory and nothing
survives the run.
"""
from __future__ import annotations

import importlib
import tempfile

import pytest
from fastapi.testclient import TestClient

P = "/api/v1"

#: **Every API route and the exact guard it must carry.**
#:
#: This table replaced a check that asked only "does the route have *some*
#: authentication dependency". That check was green while a viewer could delete
#: documents, purge all of them, rewrite settings and mint an API key — because
#: every one of those routes did have a guard; it just was not the right one. A
#: coverage test that cannot tell "authenticated" from "authorised" is a test
#: that could not fail, and this project has learned to distrust those.
#:
#: So the rule now: a route is either listed here with its guard, or the build
#: fails. Adding an endpoint means deciding who may call it, in this file, in
#: the same change.
#:
#: Values are the dependency's name, or PUBLIC with the reason it must be public.
PUBLIC = "public"
EXPECTED_GUARDS = {
    ("GET", "/health"): (PUBLIC, "container health check, before anything is configured"),
    ("GET", "/api/docs"): (PUBLIC, "API reference; describes endpoints, returns no data"),
    ("HEAD", "/api/docs"): (PUBLIC, "same page, HEAD"),
    ("GET", "/api/v1/auth/config"): (PUBLIC, "the login page needs to know what to render"),
    ("POST", "/api/v1/auth/pin-login"): (PUBLIC, "signing in cannot require being signed in"),
    ("POST", "/api/v1/auth/login"): (PUBLIC, "signing in cannot require being signed in"),

    # The only two routes a session that still owes a password change may reach.
    ("GET", "/api/v1/auth/me"): "require_session_allow_password_change",
    ("POST", "/api/v1/auth/change-password"): "require_session_allow_password_change",

    # Documents: read = viewer, write = operator. API keys are admitted at any
    # level because their scope is exactly this API and nothing beyond it.
    ("GET", "/api/v1/documents"): "require_api_or_viewer",
    ("GET", "/api/v1/documents/{doc_id}"): "require_api_or_viewer",
    ("GET", "/api/v1/documents/{doc_id}/progress"): "require_api_or_viewer",
    ("GET", "/api/v1/documents/{doc_id}/image/{kind}"): "require_api_or_viewer",
    ("POST", "/api/v1/documents"): "require_api_or_operator",
    ("POST", "/api/v1/documents/{doc_id}/reprocess"): "require_api_or_operator",
    ("DELETE", "/api/v1/documents/{doc_id}"): "require_api_or_operator",
    # Everyone's data at once: a session, and an administrator's.
    ("POST", "/api/v1/documents/purge"): "require_admin",

    ("GET", "/api/v1/status"): "require_viewer",

    # Service management. An API key is a credential; minting one is an
    # administrator act, or a viewer could hand themselves a permanent,
    # role-free route into the document API.
    ("GET", "/api/v1/api-keys"): "require_admin",
    ("POST", "/api/v1/api-keys"): "require_admin",
    ("DELETE", "/api/v1/api-keys/{key_id}"): "require_admin",
    ("GET", "/api/v1/settings"): "require_admin",
    ("PUT", "/api/v1/settings"): "require_admin",
    ("GET", "/api/v1/logs"): "require_admin",
    ("GET", "/api/v1/users"): "require_admin",
    ("POST", "/api/v1/users"): "require_admin",
    ("PATCH", "/api/v1/users/{user_id}"): "require_admin",
    ("POST", "/api/v1/users/{user_id}/password"): "require_admin",
    ("DELETE", "/api/v1/users/{user_id}"): "require_admin",
    ("GET", "/api/v1/users/audit/entries"): "require_admin",
}


# app_factory lives in conftest.py, shared with test_auth_security.py.


def _sign_in_pin(client: TestClient) -> dict[str, str]:
    token = client.post(f"{P}/auth/pin-login", json={"pin": "1234"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _sign_in_admin(client: TestClient, password: str = "1234") -> tuple[dict[str, str], dict]:
    body = client.post(f"{P}/auth/login",
                       json={"username": "admin", "password": password}).json()
    return {"Authorization": f"Bearer {body['access_token']}"}, body


# --- the guard that has to outlive this file ---------------------------------

def _guards_of(route) -> set[str]:
    """Names of the require_* dependencies a route declares, top level only.

    getattr rather than .__name__: a security scheme such as HTTPBearer is a class
    instance with no __name__, which is how the first version of this test failed.
    """
    dependant = getattr(route, "dependant", None)
    if dependant is None:
        return set()
    names = {getattr(d.call, "__name__", type(d.call).__name__)
             for d in dependant.dependencies if d.call is not None}
    return {n for n in names if n.startswith("require")}


def test_every_route_declares_the_role_it_needs(app_factory):
    """Each API route carries exactly the guard EXPECTED_GUARDS names for it.

    Three failures, all loud: a route missing from the table (someone added an
    endpoint without deciding who may call it), a route whose guard differs from
    the table (someone loosened or tightened access without saying so here), and a
    table entry with no route behind it (the table is describing an API that no
    longer exists, and is therefore checking nothing).
    """
    app = app_factory("users")
    seen: dict[tuple[str, str], set[str]] = {}
    for route in app.routes:
        path = getattr(route, "path", "")
        if not (path.startswith("/api/") or path == "/health"):
            continue
        for method in getattr(route, "methods", None) or ():
            seen[(method, path)] = _guards_of(route)

    problems = []
    for key, guards in sorted(seen.items()):
        if key not in EXPECTED_GUARDS:
            problems.append(f"UNLISTED   {key[0]:6} {key[1]}  (declares {sorted(guards) or 'nothing'})")
            continue
        expected = EXPECTED_GUARDS[key]
        if isinstance(expected, tuple):          # public, with a reason
            if guards:
                problems.append(f"NOT PUBLIC {key[0]:6} {key[1]}  declares {sorted(guards)}")
        elif guards != {expected}:
            problems.append(f"WRONG      {key[0]:6} {key[1]}  declares {sorted(guards) or 'nothing'},"
                            f" expected {expected}")
    for key in sorted(set(EXPECTED_GUARDS) - set(seen)):
        problems.append(f"STALE      {key[0]:6} {key[1]}  is in the table but not in the API")

    listing = "\n  ".join(problems)
    assert not problems, (
        f"route guards do not match EXPECTED_GUARDS:\n  {listing}\n"
        "Decide who may call each route, and record it in EXPECTED_GUARDS."
    )


# --- PIN mode: the default, and it must not have changed ---------------------

def test_pin_mode_is_the_default_when_nothing_is_set(app_factory):
    with TestClient(app_factory(None)) as client:
        config = client.get(f"{P}/auth/config").json()
        assert config["mode"] == "pin"
        assert config["pin_required"] is True
        assert config["users_enabled"] is False
        # The seeded credentials are only ever advertised in users mode.
        assert "demo_credentials" not in config


def test_pin_mode_hides_users_entirely(app_factory):
    with TestClient(app_factory("pin")) as client:
        headers = _sign_in_pin(client)
        # 404, not an empty 200: "there are no users" and "users are not a thing
        # in this configuration" are different answers.
        assert client.get(f"{P}/users", headers=headers).status_code == 404
        assert client.post(f"{P}/auth/login",
                           json={"username": "admin", "password": "1234"}).status_code == 409


def test_pin_mode_still_serves_everything_it_used_to(app_factory):
    with TestClient(app_factory("pin")) as client:
        headers = _sign_in_pin(client)
        for path in ("/documents", "/api-keys", "/settings", "/status"):
            assert client.get(f"{P}{path}", headers=headers).status_code == 200, path


def test_unknown_mode_falls_back_to_pin_rather_than_failing(app_factory):
    with TestClient(app_factory("nonsense")) as client:
        config = client.get(f"{P}/auth/config").json()
        assert config["mode"] == "pin"
        # And it says why, rather than pretending this was the intention.
        assert config["downgrade_reason"]


# --- users mode ---------------------------------------------------------------

def test_seeded_admin_must_change_password_before_doing_anything(app_factory):
    with TestClient(app_factory("users")) as client:
        headers, body = _sign_in_admin(client)
        assert body["must_change_password"] is True

        blocked = client.get(f"{P}/users", headers=headers)
        assert blocked.status_code == 403
        assert blocked.json()["detail"] == "password_change_required"
        assert client.get(f"{P}/documents", headers=headers).status_code == 403

        # The password page itself stays reachable, and only that.
        assert client.get(f"{P}/auth/me", headers=headers).status_code == 200


def test_password_change_invalidates_the_session_that_made_it(app_factory):
    with TestClient(app_factory("users")) as client:
        headers, _ = _sign_in_admin(client)
        changed = client.post(f"{P}/auth/change-password", headers=headers,
                              json={"current_password": "1234",
                                    "new_password": "Str0ngPass"})
        assert changed.status_code == 200
        assert changed.json()["reauthenticate"] is True
        # The token that performed the change is dead, not merely stale.
        assert client.get(f"{P}/auth/me", headers=headers).status_code == 401


def test_password_policy_rejects_the_seeded_password(app_factory):
    with TestClient(app_factory("users")) as client:
        headers, _ = _sign_in_admin(client)
        again = client.post(f"{P}/auth/change-password", headers=headers,
                            json={"current_password": "1234", "new_password": "1234"})
        assert again.status_code == 400


def test_last_administrator_cannot_lock_everyone_out(app_factory):
    with TestClient(app_factory("users")) as client:
        headers, _ = _sign_in_admin(client)
        client.post(f"{P}/auth/change-password", headers=headers,
                    json={"current_password": "1234", "new_password": "Str0ngPass"})
        headers, _ = _sign_in_admin(client, "Str0ngPass")

        admin = next(u for u in client.get(f"{P}/users", headers=headers).json()["items"]
                     if u["username"] == "admin")
        for payload in ({"role": "viewer"}, {"is_active": False}):
            refused = client.patch(f"{P}/users/{admin['id']}", headers=headers, json=payload)
            assert refused.status_code == 400, payload
        assert client.delete(f"{P}/users/{admin['id']}", headers=headers).status_code == 400


def test_roles_are_enforced(app_factory):
    with TestClient(app_factory("users")) as client:
        headers, _ = _sign_in_admin(client)
        client.post(f"{P}/auth/change-password", headers=headers,
                    json={"current_password": "1234", "new_password": "Str0ngPass"})
        headers, _ = _sign_in_admin(client, "Str0ngPass")

        created = client.post(f"{P}/users", headers=headers,
                              json={"username": "petrov", "password": "Oper4tor",
                                    "role": "operator"})
        assert created.status_code == 201

        body = client.post(f"{P}/auth/login",
                           json={"username": "petrov", "password": "Oper4tor"}).json()
        op = {"Authorization": f"Bearer {body['access_token']}"}
        client.post(f"{P}/auth/change-password", headers=op,
                    json={"current_password": "Oper4tor", "new_password": "Petr0vPass"})
        body = client.post(f"{P}/auth/login",
                           json={"username": "petrov", "password": "Petr0vPass"}).json()
        op = {"Authorization": f"Bearer {body['access_token']}"}

        assert client.get(f"{P}/documents", headers=op).status_code == 200
        assert client.get(f"{P}/users", headers=op).status_code == 403


def test_deactivating_a_user_kills_their_live_session(app_factory):
    with TestClient(app_factory("users")) as client:
        headers, _ = _sign_in_admin(client)
        client.post(f"{P}/auth/change-password", headers=headers,
                    json={"current_password": "1234", "new_password": "Str0ngPass"})
        headers, _ = _sign_in_admin(client, "Str0ngPass")
        user_id = client.post(f"{P}/users", headers=headers,
                              json={"username": "petrov", "password": "Oper4tor",
                                    "role": "operator"}).json()["id"]
        body = client.post(f"{P}/auth/login",
                           json={"username": "petrov", "password": "Oper4tor"}).json()
        op = {"Authorization": f"Bearer {body['access_token']}"}

        client.patch(f"{P}/users/{user_id}", headers=headers, json={"is_active": False})
        # Immediately, not when the eight-hour token expires.
        assert client.get(f"{P}/auth/me", headers=op).status_code == 401


def test_usernames_do_not_collide_only_by_case(app_factory):
    with TestClient(app_factory("users")) as client:
        headers, _ = _sign_in_admin(client)
        client.post(f"{P}/auth/change-password", headers=headers,
                    json={"current_password": "1234", "new_password": "Str0ngPass"})
        headers, _ = _sign_in_admin(client, "Str0ngPass")
        first = client.post(f"{P}/users", headers=headers,
                            json={"username": "petrov", "password": "Oper4tor",
                                  "role": "viewer"})
        clash = client.post(f"{P}/users", headers=headers,
                            json={"username": "PETROV", "password": "Oper4tor",
                                  "role": "admin"})
        assert first.status_code == 201
        assert clash.status_code == 400


def test_failed_logins_are_throttled_and_audited(app_factory):
    with TestClient(app_factory("users")) as client:
        # Probed with a name that is not the one we sign in as afterwards: the
        # lockout is per (identity, address), so hammering "admin" would lock the
        # test out of its own next step — which is the throttle working, but it
        # makes the rest of the test unrunnable.
        codes = {client.post(f"{P}/auth/login",
                             json={"username": "nobody", "password": f"wrong{i}"}).status_code
                 for i in range(12)}
        assert 429 in codes, codes

        headers, _ = _sign_in_admin(client)
        client.post(f"{P}/auth/change-password", headers=headers,
                    json={"current_password": "1234", "new_password": "Str0ngPass"})
        headers, _ = _sign_in_admin(client, "Str0ngPass")
        entries = client.get(f"{P}/users/audit/entries", headers=headers).json()["items"]
        assert any(e["action"] == "login_failed" for e in entries)
        # The attempted password must never reach the log.
        assert not any("wrong" in str(e.get("detail", "")) for e in entries)
