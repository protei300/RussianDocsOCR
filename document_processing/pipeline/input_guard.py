"""Minimal protective input guard for ``Pipeline._load_image``.

Protects the core library against physically invalid inputs and decompression
bombs before full pixel allocation:

* empty / degenerate arrays or unsupported channel layouts;
* huge rasters exceeding ``MAX_IMAGE_PIXELS`` (40 MP) or ``MAX_IMAGE_SIDE``
  (8,000 px), checked from the file header *before* ``cv2.imdecode`` when
  loading from disk.

Deliberately NOT enforced here (enforced at the HTTP service boundary in
``service/core/image_guard.py`` instead):

* solid-color / blank-frame rejection — ``Pipeline.warmup()`` passes a uniform
  grey ``(900, 1400, 3)`` array;
* tiny-frame (``< 150`` px) and extreme-aspect (``> 10:1``) rejection — existing
  library unit tests intentionally exercise small synthetic arrays and
  ``1x5000`` aspect-clamping strips through ``_prepare_image`` / ``process_img``.
"""
from __future__ import annotations

import io
import struct
import warnings

MAX_IMAGE_PIXELS: int = 40_000_000
MAX_IMAGE_SIDE: int = 8_000


def _probe_header_fallback(data: bytes) -> tuple[int, int] | None:
    """Header-only dimension parser for PNG, BMP, JPEG and WEBP (no Pillow)."""
    n = len(data)
    if n >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h)

    if n >= 26 and data[:2] == b"BM":
        dib_size = struct.unpack("<I", data[14:18])[0]
        if dib_size == 12:
            w, h = struct.unpack("<HH", data[18:22])
            return int(w), int(h)
        if dib_size >= 40:
            w, h = struct.unpack("<ii", data[18:26])
            return int(w), int(abs(h))

    if n >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X":
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return w, h
        if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            w = struct.unpack("<H", data[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", data[28:30])[0] & 0x3FFF
            return int(w), int(h)
        if chunk == b"VP8L" and data[20] == 0x2F:
            bits = struct.unpack("<I", data[21:25])[0]
            return int((bits & 0x3FFF) + 1), int(((bits >> 14) & 0x3FFF) + 1)

    if n >= 4 and data[0] == 0xFF and data[1] == 0xD8:
        pos = 2
        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3,
            0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB,
            0xCD, 0xCE, 0xCF,
        }
        while pos + 1 < n:
            if data[pos] != 0xFF:
                break
            while pos < n and data[pos] == 0xFF:
                pos += 1
            if pos >= n:
                break
            marker = data[pos]
            pos += 1
            if marker in (0xD9, 0xDA):
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue
            if pos + 2 > n:
                break
            seg_len = struct.unpack(">H", data[pos:pos + 2])[0]
            if seg_len < 2 or pos + seg_len > n:
                break
            if marker in sof_markers and seg_len >= 7:
                h, w = struct.unpack(">HH", data[pos + 3:pos + 7])
                return int(w), int(h)
            pos += seg_len

    return None


def probe_header_dimensions(
    data: bytes,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_side: int = MAX_IMAGE_SIDE,
    source: str = "image",
) -> tuple[int, int] | None:
    """Read ``(width, height)`` from encoded bytes without decoding pixels."""
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore[assignment]

    if Image is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as im:
                    w, h = im.size
                    return int(w), int(h)
        except Image.DecompressionBombError:
            raise ValueError(
                f"Image exceeds maximum raster size "
                f"(max {max_side} px per side, {max_pixels} pixels total): {source}"
            ) from None
        except Exception:
            pass

    return _probe_header_fallback(data)


def validate_raster_dimensions(
    width: int,
    height: int,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_side: int = MAX_IMAGE_SIDE,
    source: str = "image",
) -> None:
    """Reject degenerate or oversized rasters."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Degenerate image dimensions ({width}x{height}): {source}")
    if max(width, height) > max_side or (width * height) > max_pixels:
        raise ValueError(
            f"Image dimensions ({width}x{height}) exceed maximum raster size "
            f"(max {max_side} px per side, {max_pixels} pixels total): {source}"
        )


def validate_encoded_header(
    data: bytes,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_side: int = MAX_IMAGE_SIDE,
    source: str = "image",
) -> tuple[int, int] | None:
    """Check encoded image header before full pixel decoding."""
    dims = probe_header_dimensions(
        data, max_pixels=max_pixels, max_side=max_side, source=source,
    )
    if dims is not None:
        validate_raster_dimensions(
            dims[0], dims[1],
            max_pixels=max_pixels, max_side=max_side, source=source,
        )
    return dims


def validate_image_array(
    img,
    *,
    max_pixels: int = MAX_IMAGE_PIXELS,
    max_side: int = MAX_IMAGE_SIDE,
    source: str = "image array",
) -> None:
    """Validate an already-decoded numpy image array shape and raster bounds."""
    if img.size == 0 or img.ndim not in (2, 3):
        raise ValueError(f"Empty or invalid image array: shape={img.shape}")
    if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported image channel count: shape={img.shape}")
    h, w = int(img.shape[0]), int(img.shape[1])
    validate_raster_dimensions(
        w, h, max_pixels=max_pixels, max_side=max_side, source=source,
    )
