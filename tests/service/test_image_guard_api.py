"""Service unit and HTTP API tests for Layer A technical input image guard.

All test fixtures are 100% synthetic (no real samples, no personal data).
"""
from __future__ import annotations

import json
import struct
import warnings
import zlib
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from service.core import image_guard

P = "/api/v1"


def _patterned_bgr(width: int, height: int) -> np.ndarray:
    img = np.full((height, width, 3), 235, dtype=np.uint8)
    y0, y1 = max(1, height // 5), max(2, 4 * height // 5)
    x0, x1 = max(1, width // 6), max(2, 5 * width // 6)
    img[y0:y1, x0:x1] = (200, 210, 220)
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
    file_hdr = b"BM" + struct.pack("<IHHI", 54, 0, 0, 54)
    dib_hdr = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, 0, 2835, 2835, 0, 0)
    return file_hdr + dib_hdr


def _synthetic_jpeg_sof_header(width: int, height: int) -> bytes:
    sof_body = struct.pack(">BHHB", 8, height, width, 3) + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    return b"\xff\xd8\xff\xc0" + struct.pack(">H", len(sof_body) + 2) + sof_body + b"\xff\xd9"


def _synthetic_webp_vp8x_header(width: int, height: int) -> bytes:
    w_m1 = (width - 1).to_bytes(3, "little")
    h_m1 = (height - 1).to_bytes(3, "little")
    vp8x = b"VP8X" + struct.pack("<I", 10) + b"\x00\x00\x00\x00" + w_m1 + h_m1
    return b"RIFF" + struct.pack("<I", 4 + len(vp8x)) + b"WEBP" + vp8x


# ---------------------------------------------------------------------------
# Unit tests: service.core.image_guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dims", [
    (20_000, 20_000),
    (65_535, 65_535),
    (10_000, 10_000),
    (7_000, 7_000),
    (8_001, 2_000),
    (2_000, 8_001),
])
def test_decompression_bomb_rejected_before_pixel_decode(dims, monkeypatch):
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
    def forbidden_imdecode(*args, **kwargs):
        raise AssertionError("cv2.imdecode must not be called on an oversized header")

    monkeypatch.setattr(image_guard.cv2, "imdecode", forbidden_imdecode)
    with pytest.raises(image_guard.ImageGuardError) as exc_info:
        image_guard.validate_image_bytes(payload)
    assert exc_info.value.code == image_guard.IMAGE_TOO_LARGE


@pytest.mark.parametrize("width,height", [
    (3500, 2500),
    (4960, 3508),
    (150, 150),
    (200, 2000),
])
def test_valid_scan_and_boundary_sizes_pass(width, height):
    data = _encode(_patterned_bgr(width, height), ".jpg")
    assert image_guard.validate_image_bytes(data) == (width, height)


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
    (0, ".png"),
    (255, ".png"),
    (128, ".png"),
    (200, ".jpg"),
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
    data = _encode(_patterned_bgr(200, 180), ".png")
    bgr = image_guard.decode_and_validate_bgr(data)
    try:
        assert bgr.shape == (180, 200, 3)
        assert bgr.any() and float(bgr.std()) > 10.0
    finally:
        bgr.fill(0)


# ---------------------------------------------------------------------------
# HTTP API tests: POST /documents, POST /recognize, POST /recognize/registration
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_runtime(monkeypatch):
    from service import worker
    from service.core import logging as service_logging
    from service.ml import runtime
    from service.repositories import recognitions

    monkeypatch.setattr(worker, "notify_new_work", lambda: None)
    monkeypatch.setattr(service_logging, "_buffer", deque(maxlen=service_logging._buffer.maxlen))
    recognitions.reset_for_tests()

    seen: dict = {"recognise_calls": 0, "registration_calls": 0, "arrays": []}

    def fake_recognise_image(rgb, *, docconf, img_size, with_canvas, lease_timeout,
                             quality_gate="warn"):
        seen["recognise_calls"] += 1
        seen["arrays"].append(rgb)
        return {
            "doc_type": "INTPASSPORT_2011", "doc_type_base": "INTPASSPORT", "doc_type_era": "2011",
            "recognised": True, "device": "cpu", "canvas": {"width": 10, "height": 8, "is_fallback": False},
            "coord_space": "canvas", "coord_space_note": "", "boxes": [],
            "fields": [], "ocr": {}, "quality": {"DocConf": 0.95},
            "timings": {}, "address": None,
        }, None

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
    return seen


def _pin_headers(client: TestClient) -> dict[str, str]:
    token = client.post(f"{P}/auth/pin-login", json={"pin": "1234"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _files_under(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def _data_dir() -> Path:
    from service.core.config import get_settings
    return Path(get_settings().data_dir)


def _bad_api_cases():
    return [
        ("bomb-20000x20000", _synthetic_png_header(20_000, 20_000), "image/png", "bomb.png",
         image_guard.IMAGE_TOO_LARGE),
        ("huge-7000x7000", _synthetic_png_header(7_000, 7_000), "image/png", "huge.png",
         image_guard.IMAGE_TOO_LARGE),
        ("max-side-8001x2000", _synthetic_png_header(8_001, 2_000), "image/png", "wide.png",
         image_guard.IMAGE_TOO_LARGE),
        ("tiny-120x200", _encode(_patterned_bgr(120, 200), ".png"), "image/png", "tiny.png",
         image_guard.IMAGE_TOO_SMALL),
        ("aspect-150x2000", _encode(_patterned_bgr(150, 2000), ".png"), "image/png", "strip.png",
         image_guard.IMAGE_EXTREME_ASPECT_RATIO),
        ("blank-black", _solid_bytes(400, 300, 0, ".png"), "image/png", "black.png",
         image_guard.IMAGE_BLANK),
        ("blank-white", _solid_bytes(400, 300, 255, ".png"), "image/png", "white.png",
         image_guard.IMAGE_BLANK),
        ("blank-grey", _solid_bytes(400, 300, 180, ".jpg"), "image/jpeg", "grey.jpg",
         image_guard.IMAGE_BLANK),
        ("corrupt-jpeg", b"\xff\xd8\xff\xe0not-a-valid-jpeg", "image/jpeg", "corrupt.jpg",
         image_guard.IMAGE_UNDECODABLE),
    ]


def test_post_documents_rejects_invalid_images_with_422_before_db_or_queue(app_factory, fake_runtime):
    with TestClient(app_factory("pin", LOG_LEVEL="DEBUG")) as client:
        headers = _pin_headers(client)
        data_dir = _data_dir()
        before_files = _files_under(data_dir)

        for case_id, payload, ctype, filename, expected_code in _bad_api_cases():
            resp = client.post(
                f"{P}/documents",
                headers=headers,
                files={"file": (filename, payload, ctype)},
            )
            assert resp.status_code == 422, f"{case_id}: {resp.text}"
            body = resp.json()
            assert body["error_code"] == expected_code, case_id
            assert isinstance(body["detail"], str) and body["detail"], case_id

        assert _files_under(data_dir) == before_files
        assert client.get(f"{P}/documents", headers=headers).json()["total"] == 0


def test_post_documents_accepts_300dpi_synthetic_scan(app_factory, fake_runtime):
    scan = _encode(_patterned_bgr(3500, 2500), ".jpg")
    with TestClient(app_factory("pin")) as client:
        headers = _pin_headers(client)
        resp = client.post(
            f"{P}/documents",
            headers=headers,
            files={"file": ("synthetic_a4_300dpi.jpg", scan, "image/jpeg")},
        )
        assert resp.status_code == 202, resp.text
        assert resp.json()["status"] == "queued"
        assert client.get(f"{P}/documents", headers=headers).json()["total"] == 1


def test_post_recognize_rejects_invalid_images_with_422_and_zero_retention(app_factory, fake_runtime):
    cases = _bad_api_cases()
    with TestClient(app_factory("pin", LOG_LEVEL="DEBUG")) as client:
        headers = _pin_headers(client)
        data_dir = _data_dir()
        before_files = _files_under(data_dir)

        for case_id, payload, ctype, filename, expected_code in cases:
            secret_filename = f"SECRET_CITIZEN_{filename}"
            resp = client.post(
                f"{P}/recognize",
                headers=headers,
                files={"file": (secret_filename, payload, ctype)},
            )
            assert resp.status_code == 422, f"{case_id}: {resp.text}"
            assert resp.headers.get("cache-control") == "no-store", case_id
            body = resp.json()
            assert body["error_code"] == expected_code, case_id
            assert isinstance(body["detail"], str) and body["detail"], case_id

        assert fake_runtime["recognise_calls"] == 0
        assert _files_under(data_dir) == before_files

        journal = client.get(f"{P}/recognitions", headers=headers).json()["items"]
        assert len(journal) == len(cases)
        assert all(item["outcome"] == "rejected" and item["status_code"] == 422 for item in journal)

        entries = client.get(f"{P}/logs", params={"n": 2000}, headers=headers).json()["entries"]
        log_text = json.dumps(entries, ensure_ascii=False)
        for _, _, _, filename, _ in cases:
            assert f"SECRET_CITIZEN_{filename}" not in log_text
        for forbidden_dim in ("20000", "7000", "8001", "120x200", "150x2000"):
            assert forbidden_dim not in log_text


def test_post_recognize_wipes_buffers_when_blank_frame_rejected(app_factory, fake_runtime, monkeypatch):
    from service.api import recognize as rec

    captured: dict = {}
    original = rec._recognise_sync

    def spy(image, *args):
        captured["image"] = image
        return original(image, *args)

    monkeypatch.setattr(rec, "_recognise_sync", spy)

    blank = _solid_bytes(400, 300, 255, ".png")
    with TestClient(app_factory("pin")) as client:
        resp = client.post(
            f"{P}/recognize",
            content=blank,
            headers={**_pin_headers(client), "Content-Type": "image/png"},
        )
        assert resp.status_code == 422
        assert resp.json()["error_code"] == image_guard.IMAGE_BLANK

    assert fake_runtime["recognise_calls"] == 0
    assert len(captured["image"]) > 0 and not any(captured["image"]), \
        "uploaded image bytes were not wiped on 422 rejection"


def test_post_recognize_accepts_300dpi_synthetic_scan(app_factory, fake_runtime):
    scan = _encode(_patterned_bgr(3500, 2500), ".jpg")
    with TestClient(app_factory("pin")) as client:
        resp = client.post(
            f"{P}/recognize",
            content=scan,
            headers={**_pin_headers(client), "Content-Type": "image/jpeg"},
        )
        assert resp.status_code == 200, resp.text
        assert fake_runtime["recognise_calls"] == 1
        assert not fake_runtime["arrays"][0].any(), "decoded RGB buffer was not zeroed"


def test_post_recognize_registration_rejects_bomb_and_blank(app_factory, fake_runtime):
    with TestClient(app_factory("pin")) as client:
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

        assert fake_runtime["registration_calls"] == 0
