"""Attacks on the authentication layer, each one actually carried out.

Every test here corresponds to a defect found in review of the first version —
not a hypothetical. The list is the point: a reference implementation earns trust
by showing the mistakes it no longer makes, and each one is a mistake that
passed every earlier test, because those tests checked what the code did rather
than what an attacker would do.

  * a viewer could delete, purge, reconfigure and mint an API key;
  * the published default JWT secret could sign a valid administrator token;
  * a token issued to a viewer became an administrator in PIN mode;
  * a real ADMIN_PASSWORD was printed on the login page and in the log;
  * two concurrent demotions could leave the service with no administrator;
  * a demotion made during someone's sign-in was undone by that sign-in;
  * rotating usernames walked straight past the lockout;
  * the client address was written to the "no personal data" audit log;
  * "admin" and "аdmin" (Cyrillic а) could both exist.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from jose import jwt

P = "/api/v1"

START, FINAL = "Start1234x", "Final1234x"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _admin(client: TestClient) -> dict[str, str]:
    """Sign in as the seeded admin and get past the forced password change."""
    token = client.post(f"{P}/auth/login",
                        json={"username": "admin", "password": "1234"}).json()["access_token"]
    client.post(f"{P}/auth/change-password", headers=_headers(token),
                json={"current_password": "1234", "new_password": "Adm1nPass"})
    token = client.post(f"{P}/auth/login",
                        json={"username": "admin", "password": "Adm1nPass"}).json()["access_token"]
    return _headers(token)


def _account(client: TestClient, admin: dict[str, str], username: str, role: str) -> dict[str, str]:
    """Create an account with a role and return a fully usable session for it."""
    created = client.post(f"{P}/users", headers=admin,
                          json={"username": username, "password": START, "role": role})
    assert created.status_code == 201, created.text
    token = client.post(f"{P}/auth/login",
                        json={"username": username, "password": START}).json()["access_token"]
    client.post(f"{P}/auth/change-password", headers=_headers(token),
                json={"current_password": START, "new_password": FINAL})
    token = client.post(f"{P}/auth/login",
                        json={"username": username, "password": FINAL}).json()["access_token"]
    return _headers(token)


# --- roles are enforced, not merely declared ----------------------------------

#: (method, path, body) that a viewer must be refused — every write and every
#: piece of service management. 403 specifically: 401 would mean the session was
#: not recognised, which would hide a missing role check behind a different bug.
VIEWER_FORBIDDEN = [
    ("post", "/documents/purge", None),
    ("delete", "/documents/1", None),
    ("post", "/documents/1/reprocess", None),
    ("get", "/api-keys", None),
    ("post", "/api-keys", {"label": "escalation"}),
    ("get", "/settings", None),
    ("put", "/settings", {"values": {}}),
    ("get", "/logs", None),
    ("get", "/users", None),
    ("get", "/users/audit/entries", None),
]


def test_a_viewer_can_read_and_nothing_else(app_factory):
    with TestClient(app_factory("users")) as client:
        viewer = _account(client, _admin(client), "reader", "viewer")

        assert client.get(f"{P}/documents", headers=viewer).status_code == 200
        assert client.get(f"{P}/status", headers=viewer).status_code == 200

        refused = []
        for method, path, body in VIEWER_FORBIDDEN:
            kwargs = {"headers": viewer}
            if body is not None:
                kwargs["json"] = body
            status = getattr(client, method)(f"{P}{path}", **kwargs).status_code
            if status != 403:
                refused.append(f"{method.upper()} {path} -> {status}")
        assert not refused, "a viewer reached: " + ", ".join(refused)

        # Upload is multipart, so it is checked on its own.
        upload = client.post(f"{P}/documents", headers=viewer,
                             files={"file": ("x.jpg", b"\xff\xd8\xff", "image/jpeg")})
        assert upload.status_code == 403


def test_a_viewer_cannot_mint_an_api_key(app_factory):
    """The worst of the role gaps, named on its own.

    An API key is admitted to the document API at any level, because that is its
    purpose. So a viewer able to create one could hand themselves a credential
    that no role check ever looks at again. Minting a key is an administrator act.
    """
    with TestClient(app_factory("users")) as client:
        viewer = _account(client, _admin(client), "reader", "viewer")
        response = client.post(f"{P}/api-keys", headers=viewer, json={"label": "mine"})
        assert response.status_code == 403
        assert "key" not in response.json()


def test_an_operator_can_write_documents_but_not_manage_the_service(app_factory):
    with TestClient(app_factory("users")) as client:
        operator = _account(client, _admin(client), "worker", "operator")

        # Not 403 — the role is sufficient, and the request fails for its own
        # reason (no such document), which is the proof the gate let it through.
        assert client.delete(f"{P}/documents/999", headers=operator).status_code == 404
        for path in ("/api-keys", "/settings", "/logs", "/users"):
            assert client.get(f"{P}{path}", headers=operator).status_code == 403, path
        assert client.post(f"{P}/documents/purge", headers=operator).status_code == 403


# --- tokens -----------------------------------------------------------------

def test_the_published_default_secret_cannot_sign_a_token(app_factory):
    """With JWT_SECRET left at the default, a token signed with that default fails.

    The default is in this repository, so it is public. The administrator is uid 1
    and token_version starts at 1 — a forged token is therefore a guess, not an
    attack, unless the default is never actually used to sign anything.
    """
    from service.core.auth import DEFAULT_JWT_SECRET

    with TestClient(app_factory("users", JWT_SECRET=DEFAULT_JWT_SECRET)) as client:
        forged = jwt.encode(
            {"sub": "admin", "uid": 1, "tv": 1, "role": "admin",
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            DEFAULT_JWT_SECRET, algorithm="HS256")
        assert client.get(f"{P}/auth/me", headers=_headers(forged)).status_code == 401
        # And the service still works — it signs with its own random secret.
        assert client.post(f"{P}/auth/login",
                           json={"username": "admin", "password": "1234"}).status_code == 200


def test_pin_mode_refuses_a_token_issued_for_a_named_account(app_factory):
    """Switching a service back to PIN must not promote its old viewers.

    Every PIN session is the administrator. The first version recognised a
    named-account token in PIN mode, ignored its uid — and granted it that
    administrator identity. The secret is the same across the switch, so a
    viewer's still-valid token became full control.
    """
    from service.core.auth import create_access_token

    with TestClient(app_factory("pin")) as client:
        viewer_token = create_access_token({"sub": "reader", "uid": 2, "tv": 1,
                                            "role": "viewer", "name": "Reader"})
        assert client.get(f"{P}/api-keys", headers=_headers(viewer_token)).status_code == 401
        # A genuine PIN session is unaffected.
        pin = client.post(f"{P}/auth/pin-login", json={"pin": "1234"}).json()["access_token"]
        assert client.get(f"{P}/api-keys", headers=_headers(pin)).status_code == 200


# --- what the service says out loud -----------------------------------------

def test_a_real_admin_password_is_never_published(app_factory, caplog):
    secret = "Sup3r-Secret-Value"
    with caplog.at_level(logging.INFO):
        with TestClient(app_factory("users", ADMIN_PASSWORD=secret)) as client:
            response = client.get(f"{P}/auth/config")
            assert "demo_credentials" not in response.json()
            assert secret not in response.text
    assert all(secret not in record.getMessage() for record in caplog.records), \
        "the configured admin password reached the log"


def test_demo_credentials_disappear_once_they_stop_working(app_factory):
    with TestClient(app_factory("users")) as client:
        assert client.get(f"{P}/auth/config").json()["demo_credentials"]["password"] == "1234"
        _admin(client)                                  # changes the password
        assert "demo_credentials" not in client.get(f"{P}/auth/config").json()


def test_the_audit_log_holds_no_client_address(app_factory):
    with TestClient(app_factory("users")) as client:
        client.post(f"{P}/auth/login", json={"username": "admin", "password": "wrong"})
        admin = _admin(client)
        entries = client.get(f"{P}/users/audit/entries", headers=admin).json()["items"]
        assert entries, "expected sign-in events in the log"
        # TestClient requests arrive from the address "testclient".
        assert not any("testclient" in str(entry.values()) for entry in entries)


# --- the lifecycle rules hold under concurrency -----------------------------

@pytest.fixture
def store(tmp_path):
    from service.core.database import FileStore
    return FileStore(tmp_path)


def test_concurrent_demotions_cannot_remove_every_administrator(store, monkeypatch):
    """Two admins demote each other at the same instant; one must survive.

    Without the write lock both checks run before either change lands, both see
    the other still active, and the service ends with nobody able to undo it.

    **The interleaving is forced, not hoped for.** The first version of this test
    started two threads at a barrier and let them race — and it passed with the
    lock removed, because check and write take microseconds and the threads almost
    never actually overlapped. A race test that the race cannot fail is exactly
    the kind of check this suite exists to replace. So the admin count now waits
    at a second barrier after it is computed: without the lock both threads count
    before either writes, which is the bug; with the lock the second thread cannot
    even reach the count, the barrier times out, and the first proceeds alone.
    """
    from service.repositories import users as repo

    first = repo.create(store, username="alpha", password="Alpha1234", role="admin")
    second = repo.create(store, username="bravo", password="Bravo1234", role="admin")

    counted = threading.Barrier(2)
    real_count = repo.count_active_admins

    def count_then_wait_for_the_other(db, **kwargs):
        result = real_count(db, **kwargs)
        try:
            counted.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass                      # the lock kept the other thread out: correct
        return result

    monkeypatch.setattr(repo, "count_active_admins", count_then_wait_for_the_other)

    start = threading.Barrier(2)
    outcomes: list[str] = []

    def demote(user):
        start.wait()
        try:
            repo.update(store, user, role="viewer")
            outcomes.append("demoted")
        except repo.UserError:
            outcomes.append("refused")

    threads = [threading.Thread(target=demote, args=(u,)) for u in (first, second)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert repo.count_active_admins(store) == 1
    assert sorted(outcomes) == ["demoted", "refused"]


def test_a_sign_in_does_not_undo_a_demotion_made_while_it_was_hashing(store, monkeypatch):
    """The login path re-reads the account before writing it back.

    Verification takes ~80 ms. The first version then saved the copy it had loaded
    *before* hashing, so an administrator's change made in that window — here, a
    demotion — was silently overwritten by the sign-in's last_login_at update.
    """
    from service.core import passwords
    from service.repositories import users as repo

    repo.create(store, username="alpha", password="Alpha1234", role="admin")
    target = repo.create(store, username="worker", password="Worker1234", role="operator")

    real_verify = passwords.verify_password
    fired = []

    def verify_and_meanwhile_demote(stored_hash, candidate):
        result = real_verify(stored_hash, candidate)
        if candidate == "Worker1234" and not fired:
            fired.append(True)
            repo.update(store, repo.get(store, target.id), role="viewer")
        return result

    monkeypatch.setattr(passwords, "verify_password", verify_and_meanwhile_demote)
    signed_in = repo.authenticate(store, "worker", "Worker1234")

    assert fired, "the race was not exercised"
    assert signed_in is not None and signed_in.role == "viewer"
    assert repo.get(store, target.id).role == "viewer"


def test_reads_hand_out_copies_not_the_indexed_object(store):
    from service.repositories import users as repo

    user = repo.create(store, username="alpha", password="Alpha1234", role="admin")
    held = repo.get(store, user.id)
    held.role = "viewer"                     # an edit nobody saved
    assert repo.get(store, user.id).role == "admin"


# --- the lockout and the identity rules ---------------------------------------

def test_rotating_usernames_does_not_escape_the_lockout(app_factory):
    """Password spraying: one guess per username, many usernames, one address."""
    with TestClient(app_factory("users")) as client:
        codes = [client.post(f"{P}/auth/login",
                             json={"username": f"user{i}", "password": "Guess123"}).status_code
                 for i in range(40)]
        assert 429 in codes, sorted(set(codes))


@pytest.mark.parametrize("name", [
    "аdmin",            # Cyrillic а — looks identical to "admin"
    "admin ",           # stripped to "admin", so fine... see below
    "ad min",
    "admin​",      # zero-width space
    "../etc",
    "",
])
def test_usernames_that_could_impersonate_are_refused(app_factory, name):
    with TestClient(app_factory("users")) as client:
        admin = _admin(client)
        response = client.post(f"{P}/users", headers=admin,
                               json={"username": name, "password": START, "role": "viewer"})
        # "admin " strips to the existing "admin" — refused as a duplicate, which
        # is the same answer for the same reason: it would be the same identity.
        assert response.status_code in (400, 422), (name, response.status_code)
