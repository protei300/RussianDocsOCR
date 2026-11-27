"""HTTP contract tests for /api/v1/documents (upload, list, detail, images, reprocess, delete, purge)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from service.api.documents import MAX_FILENAME_LEN, _safe_filename

P = "/api/v1"


def _patterned_jpeg(w: int = 320, h: int = 200) -> bytes:
    img = np.full((h, w, 3), 230, np.uint8)
    img[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 35
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _pin(client: TestClient) -> dict[str, str]:
    token = client.post(f"{P}/auth/pin-login", json={"pin": "1234"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd", "passwd"),
    (r"C:\Users\secret\scan.jpg", "scan.jpg"),
    ('bad<>:"|?*name.jpg', "badname.jpg"),
    ("   ", "upload"),
    (None, "upload"),
])
def test_safe_filename_sanitizes_paths_and_special_chars(raw, expected):
    assert _safe_filename(raw) == expected


def test_safe_filename_truncates_to_max_length():
    long_name = "a" * (MAX_FILENAME_LEN + 50) + ".jpg"
    assert len(_safe_filename(long_name)) == MAX_FILENAME_LEN


def test_upload_validation_errors(app_factory, monkeypatch):
    from service import worker
    monkeypatch.setattr(worker, "notify_new_work", lambda: None)

    with TestClient(app_factory("pin", MAX_UPLOAD_MB="1")) as client:
        h = _pin(client)
        assert client.post(f"{P}/documents", headers=h,
                           files={"file": ("empty.jpg", b"", "image/jpeg")}).status_code == 400
        assert client.post(f"{P}/documents", headers=h,
                           files={"file": ("doc.pdf", b"%PDF-1.7 body", "application/pdf")}).status_code == 415
        assert client.post(f"{P}/documents", headers=h,
                           files={"file": ("note.txt", b"plain text", "text/plain")}).status_code == 415
        oversized = b"\xff\xd8\xff" + b"0" * (1024 * 1024 + 10)
        assert client.post(f"{P}/documents", headers=h,
                           files={"file": ("huge.jpg", oversized, "image/jpeg")}).status_code == 413


def test_documents_crud_list_filters_images_reprocess_and_purge(app_factory, monkeypatch):
    from service import worker
    from service.core.database import get_store
    from service.repositories import documents as repo

    monkeypatch.setattr(worker, "notify_new_work", lambda: None)

    with TestClient(app_factory("pin")) as client:
        h = _pin(client)
        jpg = _patterned_jpeg()

        # Upload with hostile path -> sanitized display name, status 202 queued
        up = client.post(f"{P}/documents", headers=h,
                         files={"file": ("../../passport_scan.jpg", jpg, "image/jpeg")})
        assert up.status_code == 202, up.text
        doc_id = up.json()["id"]
        assert up.json()["filename"] == "passport_scan.jpg"
        assert up.json()["status"] == "queued"

        # List + invalid status 400
        assert client.get(f"{P}/documents", headers=h).json()["total"] == 1
        assert client.get(f"{P}/documents?status=bogus", headers=h).status_code == 400
        assert client.get(f"{P}/documents?status=queued", headers=h).json()["total"] == 1
        assert client.get(f"{P}/documents?status=done", headers=h).json()["total"] == 0

        # Detail & progress + 404 on missing id
        assert client.get(f"{P}/documents/{doc_id}", headers=h).status_code == 200
        assert client.get(f"{P}/documents/99999", headers=h).status_code == 404
        prog = client.get(f"{P}/documents/{doc_id}/progress", headers=h)
        assert prog.status_code == 200 and prog.json()["step"] == "queued"
        assert client.get(f"{P}/documents/99999/progress", headers=h).status_code == 404

        # Image endpoints: original works with private, no-cache; unknown/missing -> 404
        orig = client.get(f"{P}/documents/{doc_id}/image/original", headers=h)
        assert orig.status_code == 200
        assert orig.headers.get("cache-control") == "private, no-cache"
        assert client.get(f"{P}/documents/{doc_id}/image/canvas", headers=h).status_code == 404
        assert client.get(f"{P}/documents/{doc_id}/image/bogus", headers=h).status_code == 404

        # Reprocess refuses active (queued/processing) with 409, succeeds on failed
        assert client.post(f"{P}/documents/{doc_id}/reprocess", headers=h).status_code == 409
        db = get_store()
        repo.update_status(db, repo.get_by_id(db, doc_id), "failed", error="boom", error_code="INTERNAL")
        reproc = client.post(f"{P}/documents/{doc_id}/reprocess", headers=h)
        assert reproc.status_code == 200 and reproc.json()["status"] == "queued"
        assert reproc.json()["error"] is None
        assert client.post(f"{P}/documents/99999/reprocess", headers=h).status_code == 404

        # Upload a second document and mark it processing -> purge spares processing, deletes queued
        up2 = client.post(f"{P}/documents", headers=h,
                          files={"file": ("inflight.jpg", jpg, "image/jpeg")})
        doc2_id = up2.json()["id"]
        repo.update_status(db, repo.get_by_id(db, doc2_id), "processing")

        purged = client.post(f"{P}/documents/purge", headers=h)
        assert purged.status_code == 200 and purged.json() == {"deleted": 1}
        assert client.get(f"{P}/documents/{doc_id}", headers=h).status_code == 404
        assert client.get(f"{P}/documents/{doc2_id}", headers=h).status_code == 200

        # Delete removes remaining doc; second delete -> 404
        assert client.delete(f"{P}/documents/{doc2_id}", headers=h).status_code == 204
        assert client.delete(f"{P}/documents/{doc2_id}", headers=h).status_code == 404
