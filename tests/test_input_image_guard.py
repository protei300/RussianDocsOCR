"""Unit and HTTP API tests for the Layer A technical input image guard.

All test fixtures are 100% synthetic (no real samples, no personal data).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import struct
import tempfile
import warnings
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from document_processing.pipeline.input_guard import (
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE,
    probe_header_dimensions as lib_probe_header_dimensions,
    validate_encoded_header,
    validate_image_array,
)
from document_processing.pipeline.pipeline import Pipeline
from service.core import image_guard

P = "/api/v1"


# ---------------------------------------------------------------------------
# Synthetic image helpers (zero PII)
# ---------------------------------------------------------------------------

def _patterned_bgr(width: int, height: int) -> np.ndarray:
    """Synthetic document-like patterned raster (light background + dark bands)."""
    img = np.full((height, width, 3), 235, dtype=np.uint8)
    y0, y1 = max(1, height // 5), max(2, 4 * height // 5)
    x0, x1 = max(1, width // 6), max(2, 5 * width // 6)
    img[y0:y1, x0:x1] = (200, 210, 220)
    # Dark text-like bars for realistic luminance contrast.
    bar_h = max(2, (y1 - y0) // 10)
    for idx in range(0, 5, 2):
        by0 = y0 + idx * bar_h * 2
        by1 = min(y1, by0 + bar_h)
        if by1 > by0:
            img[by0:by1, x0 + (x1 - x0) // 10: x1 - (x1 - x0) // 10] = (35, 35, 35)
    return img


def _encode(img: np.ndarray, ext: str = ".jpg") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok, f"cv2.imencode({ext}) failed"
    return buf.tobytes()


def _solid_bytes(width: int, height: int, value: int, ext: str = ".png") -> bytes:
    return _encode(np.full((height, width, 3), value, dtype=np.uint8), ext=ext)


def _synthetic_png_header(width: int, height: int, *, include_iend: bool = True) -> bytes:
    """Tiny valid PNG header declaring arbitrary dimensions without allocating pixels."""
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(ctype: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out = sig + chunk(b"IHDR", ihdr)
    if include_iend:
        out += chunk(b"IEND", b"")
    return out


def _synthetic_bmp_header(width: int, height: int) -> bytes:
    """Minimal 54-byte BMP header declaring arbitrary dimensions."""
    file_hdr = b"BM" + struct.pack("<IHHI", 54, 0, 0, 54)
    dib_hdr = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, 0, 2835, 2835, 0, 0)
    return file_hdr + dib_hdr


def _synthetic_jpeg_sof_header(width: int, height: int) -> bytes:
    """Minimal JPEG SOI + SOF0 header declaring arbitrary dimensions."""
    sof_body = struct.pack(">BHHB", 8, height, width, 3) + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8\xff\xc0" + struct.pack(">H", len(sof_body) + 2) + sof_body + b"\xff\xd9"


def _synthetic_webp_vp8x_header(width: int, height: int) -> bytes:
    """Minimal WEBP VP8X header declaring arbitrary dimensions."""
    w_m1 = (width - 1).to_bytes(3, "little")
    h_m1 = (height - 1).to_bytes(3, "little")
    vp8x = b"VP8X" + struct.pack("<I", 10) + b"\x00\x00\x00\x00" + w_m1 + h_m1
    return b"RIFF" + struct.pack("<I", 4 + len(vp8x)) + b"WEBP" + vp8x


# ---------------------------------------------------------------------------
# Unit tests: service.core.image_guard (Layer A)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dims", [
    (20_000, 20_000),   # 400 MP — Pillow DecompressionBombError
    (65_535, 65_535),   # 4.29 GP — extreme PNG decompression bomb
    (10_000, 10_000),   # 100 MP — Pillow DecompressionBombWarning range
    (7_000, 7_000),     # 49 MP — below Pillow default, above 40 MP limit
    (8_001, 2_000),     # 16 MP — above max_side=8_000
    (2_000, 8_001),     # portrait above max_side=8_000
])
def test_decompression_bomb_rejected_before_pixel_decode(dims, monkeypatch):
    """Oversized PNG headers must fail before cv2.imdecode allocates memory."""
    def forbidden_imdecode(*args, **kwargs):
        raise AssertionError("cv2.imdecode must not be called on an oversized header")

    monkeypatch.setattr(image_guard.cv2, "imdecode", forbidden_imdecode)
    payload = _synthetic_png_header(*dims)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(image_guard.ImageGuardError) as exc_info:
            image_guard.validate_image_bytes(payload)

    assert exc_info.value.code == image_guard.IMAGE_TOO_LARGE
    assert not caught, f"unexpected warnings leaked: {[w.message for w in caught]}"


@pytest.mark.parametrize("payload", [
    _synthetic_png_header(25_000, 25_000, include_iend=False),
    _synthetic_bmp_header(12_000, 12_000),
    _synthetic_jpeg_sof_header(15_000, 10_000),
    _synthetic_webp_vp8x_header(12_000, 8_000),
])
def test_header_fallback_rejects_bomb_across_formats_without_decode(payload, monkeypatch):
    """Even truncated/non-Pillow headers are refused before cv2.imdecode."""
    def forbidden_imdecode(*args, **kwargs):
        raise AssertionError("cv2.imdecode must not be called on an oversized header")

    monkeypatch.setattr(image_guard.cv2, "imdecode", forbidden_imdecode)
    with pytest.raises(image_guard.ImageGuardError) as exc_info:
        image_guard.validate_image_bytes(payload)
    assert exc_info.value.code == image_guard.IMAGE_TOO_LARGE


@pytest.mark.parametrize("width,height", [
    (3500, 2500),  # ~300 DPI A4 scan
    (4960, 3508),  # ~420 DPI A4 scan
    (150, 150),    # exact min_side boundary
    (200, 2000),   # exact 10.0:1 aspect ratio boundary
])
def test_valid_scan_and_boundary_sizes_pass(width, height):
    data = _encode(_patterned_bgr(width, height), ".jpg")
    dims = image_guard.validate_image_bytes(data)
    assert dims == (width, height)


@pytest.mark.parametrize("width,height", [
    (149, 300),
    (300, 149),
    (100, 100),
    (32, 32),
])
def test_tiny_frame_rejected_before_decode(width, height, monkeypatch):
    data = _encode(_patterned_bgr(width, height), ".png")

    def forbidden_imdecode(*args, **kwargs):
        raise AssertionError("cv2.imdecode should not run when header fails min_side")

    monkeypatch.setattr(image_guard.cv2, "imdecode", forbidden_imdecode)
    with pytest.raises(image_guard.ImageGuardError) as exc_info:
        image_guard.validate_image_bytes(data)
    assert exc_info.value.code == image_guard.IMAGE_TOO_SMALL


@pytest.mark.parametrize("width,height", [
    (150, 1600),
    (2000, 150),
    (200, 2500),
])
def test_extreme_aspect_ratio_rejected_before_decode(width, height, monkeypatch):
    data = _encode(_patterned_bgr(width, height), ".png")

    def forbidden_imdecode(*args, **kwargs):
        raise AssertionError("cv2.imdecode should not run when header fails aspect ratio")

    monkeypatch.setattr(image_guard.cv2, "imdecode", forbidden_imdecode)
    with pytest.raises(image_guard.ImageGuardError) as exc_info:
        image_guard.validate_image_bytes(data)
    assert exc_info.value.code == image_guard.IMAGE_EXTREME_ASPECT_RATIO


@pytest.mark.parametrize("value,ext", [
    (0, ".png"),    # solid black
    (255, ".png"),  # solid white
    (128, ".png"),  # flat mid-grey
    (200, ".jpg"),  # JPEG-compressed flat grey
])
def test_blank_or_uniform_frames_rejected(value, ext):
    data = _solid_bytes(400, 300, value, ext=ext)
    with pytest.raises(image_guard.ImageGuardError) as exc_info:
        image_guard.validate_image_bytes(data)
    assert exc_info.value.code == image_guard.IMAGE_BLANK


def test_corrupt_image_bytes_rejected_as_undecodable():
    with pytest.raises(image_guard.ImageGuardError) as exc_info:
        image_guard.validate_image_bytes(b"\xff\xd8\xff\xe0truncated-corrupt-jpeg")
    assert exc_info.value.code == image_guard.IMAGE_UNDECODABLE


def test_artifacts_decode_dimensions_refuses_bomb_without_imdecode(monkeypatch):
    from service.repositories import artifacts

    def forbidden_imdecode(*args, **kwargs):
        raise AssertionError("decode_dimensions must not call cv2.imdecode on a bomb")

    monkeypatch.setattr(artifacts.cv2, "imdecode", forbidden_imdecode)
    assert artifacts.decode_dimensions(_synthetic_png_header(30_000, 30_000)) is None


def test_decode_and_validate_bgr_preserves_small_valid_image():
    """Small valid frames (<= 256 px) must not have the returned BGR array zeroed."""
    data = _encode(_patterned_bgr(200, 180), ".png")
    bgr = image_guard.decode_and_validate_bgr(data)
    try:
        assert bgr.shape == (180, 200, 3)
        assert bgr.any() and float(bgr.std()) > 10.0
    finally:
        bgr.fill(0)


# ---------------------------------------------------------------------------
# Unit tests: document_processing Pipeline._load_image & Pipeline.warmup()
# ---------------------------------------------------------------------------

def test_pipeline_load_image_rejects_empty_invalid_and_oversized_arrays():
    pipe = object.__new__(Pipeline)
    pipe._start_document()

    with pytest.raises(ValueError):
        pipe._load_image(np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        pipe._load_image(np.zeros((10,), dtype=np.uint8))
    with pytest.raises(ValueError):
        pipe._load_image(np.zeros((2, 200, 200, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        pipe._load_image(np.zeros((200, 200, 5), dtype=np.uint8))

    # Zero-memory stride views to test huge raster rejection without allocating GBs.
    too_wide = np.broadcast_to(np.uint8(0), (2000, MAX_IMAGE_SIDE + 1, 3))
    with pytest.raises(ValueError, match="maximum raster size"):
        pipe._load_image(too_wide)

    too_many_pixels = np.broadcast_to(np.uint8(0), (9000, 9000, 3))
    assert 9000 * 9000 > MAX_IMAGE_PIXELS
    with pytest.raises(ValueError, match="maximum raster size"):
        pipe._load_image(too_many_pixels)


def test_pipeline_load_image_rejects_bomb_file_before_imdecode(tmp_path, monkeypatch):
    bomb_path = tmp_path / "bomb.png"
    bomb_path.write_bytes(_synthetic_png_header(20_000, 20_000))

    import document_processing.pipeline.pipeline as pipeline_mod

    def forbidden_imdecode(*args, **kwargs):
        raise AssertionError("Pipeline._load_image must reject huge headers before cv2.imdecode")

    monkeypatch.setattr(pipeline_mod.cv2, "imdecode", forbidden_imdecode)
    pipe = object.__new__(Pipeline)
    pipe._start_document()

    with pytest.raises(ValueError, match="maximum raster size"):
        pipe._load_image(bomb_path)


def test_pipeline_input_guard_helpers_accept_normal_and_warmup_arrays():
    assert lib_probe_header_dimensions(_synthetic_png_header(3500, 2500)) == (3500, 2500)
    validate_encoded_header(_synthetic_png_header(3500, 2500))
    # Uniform grey warmup array must pass _load_image without error.
    warmup_frame = np.full((900, 1400, 3), 200, dtype=np.uint8)
    validate_image_array(warmup_frame)


def test_pipeline_warmup_works_cleanly_with_default_grey_frame(capsys):
    """Pipeline.warmup() without arguments passes a uniform 900x1400 grey frame."""
    pipe = object.__new__(Pipeline)
    pipe.probe = None
    seen_shapes: list[tuple[int, ...]] = []

    def fake_find_documents(frame, img_size):
        seen_shapes.append(frame.shape)
        return []

    def fake_read_document(frame, documents, index, img_size, options):
        return pipe.results

    pipe._find_documents = fake_find_documents
    pipe._read_document = fake_read_document

    pipe.warmup()

    captured = capsys.readouterr()
    assert "non-fatal" not in captured.out
    assert seen_shapes == [(900, 1400, 3)]


# ---------------------------------------------------------------------------
# HTTP API tests: POST /documents, POST /recognize, POST /recognize/registration
# ---------------------------------------------------------------------------

_HAS_SERVICE_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("fastapi", "sqlalchemy", "jwt")
)


@pytest.fixture
def api_app_factory(monkeypatch):
    if not _HAS_SERVICE_DEPS:
        pytest.skip("service dependencies are not installed")

    from service import worker
    from service.core import auth as auth_core
    from service.core import logging as service_logging
    from service.ml import runtime
    from service.repositories import recognitions
    from collections import deque

    async def no_runtime():
        return None

    monkeypatch.setattr(worker, "_init_runtime_bg", no_runtime)
    monkeypatch.setattr(worker, "notify_new_work", lambda: None)
    monkeypatch.setattr(service_logging, "_buffer", deque(maxlen=service_logging._buffer.maxlen))
    auth_core._attempts.clear()
    auth_core._process_secret = None
    recognitions.reset_for_tests()

    seen: dict = {"recognise_calls": 0, "registration_calls": 0, "arrays": []}

    def fake_recognise_image(rgb, *, docconf, img_size, with_canvas, lease_timeout,
                             quality_gate="warn"):
        seen["recognise_calls"] += 1
        seen["arrays"].append(rgb)
        payload = {
            "doc_type": "INTPASSPORT_2011", "doc_type_base": "INTPASSPORT", "doc_type_era": "2011",
            "recognised": True, "device": "cpu", "canvas": {"width": 10, "height": 8, "is_fallback": False},
            "coord_space": "canvas", "coord_space_note": "", "boxes": [],
            "fields": [], "ocr": {}, "quality": {"DocConf": 0.95},
            "timings": {}, "address": None,
        }
        return payload, None

    def fake_recognise_registration(rgb, *, docconf, img_size, lease_timeout):
        seen["registration_calls"] += 1
        seen["arrays"].append(rgb)
        return {
            "doc_type": "INTPASSPORTADDR_ALL", "decision": "registration", "recognised": True,
            "doc_conf": 0.95, "types_in_frame": ["INTPASSPORTADDR_ALL"], "notes": [],
            "coord_space": "input", "pages": [], "order_by": "date",
            "stamps": [], "quality": {}, "timings": {},
        }

    monkeypatch.setattr(runtime, "recognise_image", fake_recognise_image)
    monkeypatch.setattr(runtime, "recognise_registration", fake_recognise_registration)

    def build(**env: str | None):
        data_dir = Path(tempfile.mkdtemp())
        base = {
            "DATA_DIR": str(data_dir),
            "JWT_SECRET": "test-secret",
            "SEED_SAMPLES": "-1",
            "WARMUP_IMAGE": "",
            "COMPUTE_DEVICE": "cpu",
            "AUTH_MODE": "pin",
            "LOG_LEVEL": "DEBUG",
        }
        base.update(env)
        for name, value in base.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)

        from service.core.config import get_settings
        get_settings.cache_clear()
        import service.main as main
        importlib.reload(main)
        return main.app, data_dir, seen

    yield build
    auth_core._attempts.clear()
    auth_core._process_secret = None


def _pin_headers(client) -> dict[str, str]:
    token = client.post(f"{P}/auth/pin-login", json={"pin": "1234"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _files_under(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def _bad_api_cases():
    return [
        pytest.param(_synthetic_png_header(20_000, 20_000), "image/png", "bomb.png",
                     image_guard.IMAGE_TOO_LARGE, id="bomb-20000x20000"),
        pytest.param(_synthetic_png_header(7_000, 7_000), "image/png", "huge.png",
                     image_guard.IMAGE_TOO_LARGE, id="huge-7000x7000"),
        pytest.param(_synthetic_png_header(8_001, 2_000), "image/png", "wide.png",
                     image_guard.IMAGE_TOO_LARGE, id="max-side-8001x2000"),
        pytest.param(_encode(_patterned_bgr(120, 200), ".png"), "image/png", "tiny.png",
                     image_guard.IMAGE_TOO_SMALL, id="tiny-120x200"),
        pytest.param(_encode(_patterned_bgr(150, 2000), ".png"), "image/png", "strip.png",
                     image_guard.IMAGE_EXTREME_ASPECT_RATIO, id="aspect-150x2000"),
        pytest.param(_solid_bytes(400, 300, 0, ".png"), "image/png", "black.png",
                     image_guard.IMAGE_BLANK, id="blank-black"),
        pytest.param(_solid_bytes(400, 300, 255, ".png"), "image/png", "white.png",
                     image_guard.IMAGE_BLANK, id="blank-white"),
        pytest.param(_solid_bytes(400, 300, 180, ".jpg"), "image/jpeg", "grey.jpg",
                     image_guard.IMAGE_BLANK, id="blank-grey"),
        pytest.param(b"\xff\xd8\xff\xe0not-a-valid-jpeg", "image/jpeg", "corrupt.jpg",
                     image_guard.IMAGE_UNDECODABLE, id="corrupt-jpeg"),
    ]


@pytest.mark.service
@pytest.mark.parametrize("payload,ctype,filename,expected_code", _bad_api_cases())
def test_post_documents_rejects_invalid_images_with_422_before_db_or_queue(
    api_app_factory, payload, ctype, filename, expected_code,
):
    from fastapi.testclient import TestClient

    app, data_dir, _ = api_app_factory()
    with TestClient(app) as client:
        headers = _pin_headers(client)
        before_files = _files_under(data_dir)

        resp = client.post(
            f"{P}/documents",
            headers=headers,
            files={"file": (filename, payload, ctype)},
        )
        assert resp.status_code == 422, resp.text
        body = resp.json()
        assert body["error_code"] == expected_code
        assert isinstance(body["detail"], str) and body["detail"]

        # Nothing stored on disk and no document row created.
        assert _files_under(data_dir) == before_files
        assert client.get(f"{P}/documents", headers=headers).json()["total"] == 0


@pytest.mark.service
def test_post_documents_accepts_300dpi_synthetic_scan(api_app_factory):
    from fastapi.testclient import TestClient

    app, _, _ = api_app_factory()
    scan = _encode(_patterned_bgr(3500, 2500), ".jpg")
    with TestClient(app) as client:
        headers = _pin_headers(client)
        resp = client.post(
            f"{P}/documents",
            headers=headers,
            files={"file": ("synthetic_a4_300dpi.jpg", scan, "image/jpeg")},
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["status"] == "queued"
        assert client.get(f"{P}/documents", headers=headers).json()["total"] == 1


@pytest.mark.service
@pytest.mark.parametrize("payload,ctype,filename,expected_code", _bad_api_cases())
def test_post_recognize_rejects_invalid_images_with_422_and_zero_retention(
    api_app_factory, payload, ctype, filename, expected_code,
):
    from fastapi.testclient import TestClient

    secret_filename = f"SECRET_CITIZEN_{filename}"
    app, data_dir, seen = api_app_factory()
    with TestClient(app) as client:
        headers = _pin_headers(client)
        before_files = _files_under(data_dir)

        resp = client.post(
            f"{P}/recognize",
            headers=headers,
            files={"file": (secret_filename, payload, ctype)},
        )
        assert resp.status_code == 422, resp.text
        assert resp.headers.get("cache-control") == "no-store"
        body = resp.json()
        assert body["error_code"] == expected_code
        assert isinstance(body["detail"], str) and body["detail"]

        # Runtime was never invoked and nothing touched DATA_DIR.
        assert seen["recognise_calls"] == 0
        assert _files_under(data_dir) == before_files

        # Recognition journal records only outcome + status + timing.
        journal = client.get(f"{P}/recognitions", headers=headers).json()["items"]
        assert journal[0]["outcome"] == "rejected"
        assert journal[0]["status_code"] == 422

        # Zero-retention log check: no filename or raw dimensions leaked.
        entries = client.get(f"{P}/logs", params={"n": 2000}, headers=headers).json()["entries"]
        log_text = json.dumps(entries, ensure_ascii=False)
        assert secret_filename not in log_text
        assert "20000" not in log_text
        assert "9000" not in log_text
        assert "10001" not in log_text
        assert "120x200" not in log_text


@pytest.mark.service
def test_post_recognize_wipes_buffers_when_blank_frame_rejected(api_app_factory, monkeypatch):
    from fastapi.testclient import TestClient
    from service.api import recognize as rec

    captured: dict = {}
    original = rec._recognise_sync

    def spy(image, *args):
        captured["image"] = image
        return original(image, *args)

    monkeypatch.setattr(rec, "_recognise_sync", spy)

    app, _, seen = api_app_factory()
    blank = _solid_bytes(400, 300, 255, ".png")
    with TestClient(app) as client:
        resp = client.post(
            f"{P}/recognize",
            content=blank,
            headers={**_pin_headers(client), "Content-Type": "image/png"},
        )
        assert resp.status_code == 422
        assert resp.json()["error_code"] == image_guard.IMAGE_BLANK

    assert seen["recognise_calls"] == 0
    assert len(captured["image"]) > 0 and not any(captured["image"]), \
        "uploaded image bytes were not wiped on 422 rejection"


@pytest.mark.service
def test_post_recognize_accepts_300dpi_synthetic_scan(api_app_factory):
    from fastapi.testclient import TestClient

    app, _, seen = api_app_factory()
    scan = _encode(_patterned_bgr(3500, 2500), ".jpg")
    with TestClient(app) as client:
        resp = client.post(
            f"{P}/recognize",
            content=scan,
            headers={**_pin_headers(client), "Content-Type": "image/jpeg"},
        )
        assert resp.status_code == 200, resp.text
        assert seen["recognise_calls"] == 1
        assert not seen["arrays"][0].any(), "decoded RGB buffer was not zeroed"


@pytest.mark.service
def test_post_recognize_registration_rejects_bomb_and_blank(api_app_factory):
    from fastapi.testclient import TestClient

    app, _, seen = api_app_factory()
    with TestClient(app) as client:
        headers = _pin_headers(client)

        r_bomb = client.post(
            f"{P}/recognize/registration",
            content=_synthetic_png_header(20_000, 20_000),
            headers={**headers, "Content-Type": "image/png"},
        )
        assert r_bomb.status_code == 422
        assert r_bomb.json()["error_code"] == image_guard.IMAGE_TOO_LARGE

        r_blank = client.post(
            f"{P}/recognize/registration",
            content=_solid_bytes(400, 300, 0, ".png"),
            headers={**headers, "Content-Type": "image/png"},
        )
        assert r_blank.status_code == 422
        assert r_blank.json()["error_code"] == image_guard.IMAGE_BLANK

        assert seen["registration_calls"] == 0
