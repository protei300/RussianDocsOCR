"""POST /recognize and PROCESSING_MODE=memory: nothing about a document is kept.

Every test here checks a way something could be kept, not just that the route
answers: files under DATA_DIR, temporary files during upload parsing, log lines,
the journal, and the buffers themselves (which must be zeroed, not merely dropped).
Recognition is replaced by a fake so the suite runs without the models; the real
pipeline is exercised by the --runslow test at the bottom.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

P = "/api/v1"
SECRET_TEXT = "ИВАНОВ ЛЕНИНА 17"          # what the fake "recognises" — must never reach a log
FILENAME = "Ivanov_passport_scan.jpg"     # what the client calls the file — same


def _jpeg(w: int = 320, h: int = 200, noise: bool = False) -> bytes:
    img = (np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8) if noise
           else np.full((h, w, 3), 200, np.uint8))
    ok, buf = cv2.imencode(".png" if noise else ".jpg", img)
    return buf.tobytes()


def _files_under(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def fake(monkeypatch):
    """Replace model loading and recognition; record what recognition was given."""
    from service import worker
    from service.ml import runtime
    from service.repositories import recognitions

    async def no_runtime():
        return None
    monkeypatch.setattr(worker, "_init_runtime_bg", no_runtime)
    recognitions.reset_for_tests()

    seen: dict = {"arrays": []}

    def recognise_image(rgb, *, docconf, img_size, with_canvas, lease_timeout):
        seen["arrays"].append(rgb)
        seen["lease_timeout"] = lease_timeout
        if seen.get("raise"):
            raise seen["raise"]
        payload = {
            "doc_type": "INTPASSPORT_2011", "doc_type_base": "INTPASSPORT", "doc_type_era": "2011",
            "recognised": True, "device": "cpu", "canvas": {"width": 10, "height": 8, "is_fallback": False},
            "coord_space": "canvas", "coord_space_note": "", "boxes": [],
            "fields": [{"name": "Address", "value": SECRET_TEXT}],
            "ocr": {"Address": SECRET_TEXT}, "quality": {"DocConf": 0.97, "Blur": "good"},
            "timings": {}, "address": None,
        }
        png = bytearray(b"\x89PNG fake canvas bytes") if with_canvas else None
        seen["png"] = png
        return payload, png

    monkeypatch.setattr(runtime, "recognise_image", recognise_image)
    return seen


def _pin(client: TestClient) -> dict[str, str]:
    token = client.post(f"{P}/auth/pin-login", json={"pin": "1234"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _log_text(client: TestClient, h: dict) -> str:
    """Everything in the service log, as an operator sees it (GET /logs)."""
    import json
    entries = client.get(f"{P}/logs", params={"n": 2000}, headers=h).json()["entries"]
    return json.dumps(entries, ensure_ascii=False)


def _data_dir(client: TestClient) -> Path:
    from service.core.config import get_settings
    return Path(get_settings().data_dir)


# --------------------------------------------------------------------------- the route

def test_recognize_answers_and_stores_nothing(app_factory, fake):
    with TestClient(app_factory("pin", LOG_LEVEL="DEBUG")) as client:
        h = _pin(client)
        root = _data_dir(client)
        before = _files_under(root)
        resp = client.post(f"{P}/recognize", content=_jpeg(),
                           headers={**h, "Content-Type": "image/jpeg"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["stored"] is False and body["recognised"] is True
        assert body["ocr"]["Address"] == SECRET_TEXT
        assert body["doc_conf"] == 0.97 and "DocConf" not in body["quality"]
        assert "image_png_base64" not in body["canvas"]
        assert resp.headers["cache-control"] == "no-store"

        assert _files_under(root) == before, "a recognition left a file in DATA_DIR"
        assert client.get(f"{P}/documents", headers=h).json()["total"] == 0

        import json
        entries = client.get(f"{P}/logs", params={"n": 2000}, headers=h).json()["entries"]
        text = json.dumps(entries, ensure_ascii=False)
        assert SECRET_TEXT not in text and "INTPASSPORT" not in text
        lines = [e["message"] for e in entries if e["message"].startswith("[API] recognize")]
        assert len(lines) == 1 and lines[0].startswith("[API] recognize: outcome=recognised status=200 ms=")


def test_multipart_upload_is_split_in_memory_not_spooled(app_factory, fake, monkeypatch):
    """A >1 MB multipart part is spooled to a temp file by UploadFile; this route must not."""
    import tempfile

    import starlette.formparsers as fp

    def forbidden(*a, **k):
        raise AssertionError("a temporary file was created while parsing the upload")
    monkeypatch.setattr(fp, "SpooledTemporaryFile", forbidden)
    monkeypatch.setattr(tempfile, "SpooledTemporaryFile", forbidden)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", forbidden)
    monkeypatch.setattr(tempfile, "TemporaryFile", forbidden)

    big = _jpeg(1400, 1000, noise=True)
    assert len(big) > 1024 * 1024
    with TestClient(app_factory("pin", LOG_LEVEL="DEBUG")) as client:
        h = _pin(client)
        resp = client.post(f"{P}/recognize", headers=h,
                           files={"file": (FILENAME, big, "image/png")})
        assert resp.status_code == 200, resp.text
        assert FILENAME not in _log_text(client, h)


def test_buffers_are_zeroed_after_the_answer(app_factory, fake, monkeypatch):
    from service.api import recognize as rec

    captured = {}
    original = rec._recognise_sync

    def spy(image, *args):
        captured["image"] = image
        return original(image, *args)
    monkeypatch.setattr(rec, "_recognise_sync", spy)

    with TestClient(app_factory("pin")) as client:
        resp = client.post(f"{P}/recognize?images=1", content=_jpeg(),
                           headers={**_pin(client), "Content-Type": "image/jpeg"})
        assert resp.status_code == 200
        assert base64.b64decode(resp.json()["canvas"]["image_png_base64"]).startswith(b"\x89PNG")
    assert len(captured["image"]) > 0 and not any(captured["image"]), "the image bytes were not wiped"
    rgb = fake["arrays"][0]
    assert rgb.size > 0 and not rgb.any(), "the decoded pixels were not wiped"
    assert not any(fake["png"]), "the canvas PNG was not wiped"


@pytest.mark.parametrize("payload, ctype, status, outcome", [
    (b"", "image/jpeg", 400, "rejected"),
    (b"%PDF-1.7 not an image", "application/pdf", 415, "rejected"),
    (b"just some text", "text/plain", 415, "rejected"),
])
def test_bad_input_is_refused_and_journalled(app_factory, fake, payload, ctype, status, outcome):
    with TestClient(app_factory("pin")) as client:
        h = _pin(client)
        resp = client.post(f"{P}/recognize", content=payload, headers={**h, "Content-Type": ctype})
        assert resp.status_code == status
        items = client.get(f"{P}/recognitions", headers=h).json()["items"]
        assert items[0]["outcome"] == outcome and items[0]["status_code"] == status


def test_oversized_upload_is_refused(app_factory, fake):
    with TestClient(app_factory("pin", MAX_UPLOAD_MB="1")) as client:
        resp = client.post(f"{P}/recognize", content=b"\xff\xd8\xff" + b"0" * (2 * 1024 * 1024),
                           headers={**_pin(client), "Content-Type": "image/jpeg"})
        assert resp.status_code == 413


def test_busy_pipeline_is_503_with_retry_after(app_factory, fake):
    from service.ml.errors import PipelineBusy
    fake["raise"] = PipelineBusy("no pipeline")
    with TestClient(app_factory("pin", RECOGNIZE_WAIT_SEC="7")) as client:
        h = _pin(client)
        resp = client.post(f"{P}/recognize", content=_jpeg(), headers={**h, "Content-Type": "image/jpeg"})
        assert resp.status_code == 503 and resp.headers.get("retry-after")
        assert fake["lease_timeout"] == 7.0
        assert client.get(f"{P}/recognitions", headers=h).json()["items"][0]["outcome"] == "busy"


def test_unexpected_error_is_500_and_logs_only_the_type(app_factory, fake):
    fake["raise"] = ValueError(f"failed while reading {SECRET_TEXT}")
    with TestClient(app_factory("pin", LOG_LEVEL="DEBUG")) as client:
        h = _pin(client)
        resp = client.post(f"{P}/recognize", content=_jpeg(), headers={**h, "Content-Type": "image/jpeg"})
        assert resp.status_code == 500 and SECRET_TEXT not in resp.text
        text = _log_text(client, h)
        assert SECRET_TEXT not in text and "ValueError" in text


def test_journal_holds_outcome_and_time_only(app_factory, fake):
    with TestClient(app_factory("pin")) as client:
        h = _pin(client)
        client.post(f"{P}/recognize", content=_jpeg(), headers={**h, "Content-Type": "image/jpeg"})
        body = client.get(f"{P}/recognitions", headers=h).json()
        assert body["total"] == 1 and body["counts"]["recognised"] == 1
        assert set(body["items"][0]) == {"id", "at", "outcome", "status_code", "processing_ms", "stored"}


def test_roles(app_factory, fake):
    """A viewer may read the journal but not recognise; an API key may recognise."""
    with TestClient(app_factory("users")) as client:
        from test_auth_security import _admin   # noqa: PLC0415 - shared helper, same folder
        admin = _admin(client)
        client.post(f"{P}/users", headers=admin,
                    json={"username": "vera", "password": "Viewer-Pass1", "role": "viewer"})
        t = client.post(f"{P}/auth/login", json={"username": "vera", "password": "Viewer-Pass1"}).json()
        tok = {"Authorization": f"Bearer {t['access_token']}"}
        client.post(f"{P}/auth/change-password", headers=tok,
                    json={"current_password": "Viewer-Pass1", "new_password": "Viewer-Pass2"})
        t = client.post(f"{P}/auth/login", json={"username": "vera", "password": "Viewer-Pass2"}).json()
        viewer = {"Authorization": f"Bearer {t['access_token']}"}
        assert client.post(f"{P}/recognize", content=_jpeg(),
                           headers={**viewer, "Content-Type": "image/jpeg"}).status_code == 403
        assert client.get(f"{P}/recognitions", headers=viewer).status_code == 200
        key = client.post(f"{P}/api-keys", headers=admin, json={"label": "int"}).json()["key"]
        assert client.post(f"{P}/recognize", content=_jpeg(),
                           headers={"X-API-Key": key, "Content-Type": "image/jpeg"}).status_code == 200
        assert client.post(f"{P}/recognize", content=_jpeg(),
                           headers={"Content-Type": "image/jpeg"}).status_code == 401


# --------------------------------------------------------------------------- memory mode

def test_memory_mode_turns_document_storage_off(app_factory, fake, monkeypatch):
    import starlette.formparsers as fp

    def forbidden(*a, **k):
        raise AssertionError("the body of a refused /documents upload was parsed")
    monkeypatch.setattr(fp, "SpooledTemporaryFile", forbidden)

    with TestClient(app_factory("pin", PROCESSING_MODE="memory", SEED_SAMPLES="0")) as client:
        h = _pin(client)
        root = _data_dir(client)
        before = _files_under(root)
        cfg = client.get(f"{P}/auth/config").json()
        assert cfg["processing_mode"] == "memory"
        for method, path in [("GET", "/documents"), ("GET", "/documents/1"),
                             ("GET", "/documents/1/image/canvas"), ("DELETE", "/documents/1"),
                             ("POST", "/documents/purge")]:
            r = client.request(method, P + path, headers=h)
            assert r.status_code == 404 and "PROCESSING_MODE=memory" in r.json()["detail"], (method, path)
        big = _jpeg(1400, 1000, noise=True)
        r = client.post(f"{P}/documents", headers=h, files={"file": (FILENAME, big, "image/png")})
        assert r.status_code == 404
        assert client.post(f"{P}/recognize", content=_jpeg(),
                           headers={**h, "Content-Type": "image/jpeg"}).status_code == 200
        assert _files_under(root) == before, "memory mode wrote a file"
        status = client.get(f"{P}/status", headers=h).json()
        assert status["processing"]["mode"] == "memory"
        assert status["processing"]["recognitions"]["recognised"] == 1


def test_memory_mode_does_not_seed_samples(app_factory, fake):
    with TestClient(app_factory("pin", PROCESSING_MODE="memory", SEED_SAMPLES="0")) as client:
        assert not any(p.name == "record.json" for p in _data_dir(client).rglob("*"))


def test_a_mistyped_mode_means_memory_and_says_why(app_factory, fake):
    with TestClient(app_factory("pin", PROCESSING_MODE="memroy")) as client:
        h = _pin(client)
        assert client.get(f"{P}/auth/config").json()["processing_mode"] == "memory"
        reason = client.get(f"{P}/status", headers=h).json()["processing"]["reason"]
        assert "memroy" in reason
        assert client.get(f"{P}/documents", headers=h).status_code == 404
        assert "memroy" in _log_text(client, h)


def test_unset_mode_is_stored(app_factory, fake):
    with TestClient(app_factory("pin", PROCESSING_MODE=None)) as client:
        assert client.get(f"{P}/auth/config").json()["processing_mode"] == "stored"
        assert client.get(f"{P}/documents", headers=_pin(client)).status_code == 200


# --------------------------------------------------------------------------- the pipeline scrub

class _FakeResults:
    def __init__(self):
        self.original = np.full((50, 60, 3), 7, np.uint8)
        self._meta_results = {"crops": [np.full((8, 30, 3), 9, np.uint8)], "text": SECRET_TEXT}
        self.img_with_fixed_perspective = np.full((40, 50, 3), 5, np.uint8)


class _FakePipeline:
    """Holds a result and a per-image MRZ crop the way the real Pipeline does; keeps
    its own reference to what it produced so the test can inspect it after the wipe."""
    device = "cpu"

    def __init__(self, fail=False):
        self.fail, self.results, self._mrz_zone = fail, None, None
        self.weights = np.full((4,), 3, np.float32)       # stands for model state: must NOT be wiped
        self.produced = None

    def process_img(self, img, **kw):
        self.results = _FakeResults()
        self._mrz_zone = np.full((5, 5), 4, np.uint8)
        self.produced = (self.results, self._mrz_zone)
        if self.fail:
            raise RuntimeError("boom")
        return self.results


@pytest.mark.parametrize("fail", [False, True])
def test_recognise_image_wipes_the_pipeline_result_inside_the_lease(monkeypatch, fail):
    import contextlib

    from service.ml import runtime, transform

    pipe = _FakePipeline(fail=fail)
    released = {}

    @contextlib.contextmanager
    def lease(timeout):
        try:
            yield pipe
        finally:
            # the lease is being released: the scrub must already have happened
            released["clean"] = pipe.results is None and pipe._mrz_zone is None
    monkeypatch.setattr(runtime, "lease_pipeline", lease)
    monkeypatch.setattr(transform, "build_viewmodel", lambda results, **k: {"doc_type": "X"})

    rgb = np.full((10, 10, 3), 1, np.uint8)
    if fail:
        with pytest.raises(RuntimeError):
            runtime.recognise_image(rgb, with_canvas=True)
    else:
        payload, png = runtime.recognise_image(rgb, with_canvas=True)
        assert payload == {"doc_type": "X"} and png and bytes(png[:4]) == b"\x89PNG"
    res, mrz = pipe.produced
    assert released["clean"], "the pipeline went back to the pool still holding the document"
    assert not res.original.any() and not res.img_with_fixed_perspective.any()
    assert not res._meta_results["crops"][0].any() and not mrz.any()
    assert pipe.weights.all(), "model state was wiped"


# --------------------------------------------------------------------------- the real models

@pytest.mark.slow
def test_real_pipeline_in_memory_mode(app_factory, monkeypatch):
    """End to end on a real anonymised sample: recognised, nothing written, pipeline clean."""
    import time

    from service.ml import runtime

    sample = sorted(Path(__file__).resolve().parents[2].joinpath("samples").rglob("*.jpg"))[0]
    with TestClient(app_factory("pin", PROCESSING_MODE="memory")) as client:
        h = _pin(client)
        deadline = time.time() + 300
        while not runtime.is_ready() and time.time() < deadline:
            time.sleep(1)
        assert runtime.is_ready()
        root = _data_dir(client)
        before = _files_under(root)
        resp = client.post(f"{P}/recognize?images=1", content=sample.read_bytes(),
                           headers={**h, "Content-Type": "image/jpeg"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["recognised"] and body["fields"]
        assert base64.b64decode(body["canvas"]["image_png_base64"])[:4] == b"\x89PNG"
        assert _files_under(root) == before
        for pipe in list(runtime._POOL.queue):
            assert getattr(pipe, "results", None) is None
