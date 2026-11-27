"""Unit tests for the library Layer A input image guard (document_processing only).

Service-side guard & HTTP endpoint tests live in ``tests/service/test_image_guard_api.py``.
All fixtures are 100% synthetic (no real samples, no personal data).
"""
from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from document_processing.pipeline.input_guard import (
    MAX_IMAGE_PIXELS,
    MAX_IMAGE_SIDE,
    probe_header_dimensions,
    validate_encoded_header,
    validate_image_array,
)
from document_processing.pipeline.pipeline import Pipeline


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
    assert probe_header_dimensions(_synthetic_png_header(3500, 2500)) == (3500, 2500)
    validate_encoded_header(_synthetic_png_header(3500, 2500))
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
