"""Layer A technical input image guard for the HTTP service.

Rejects physically invalid uploads before queueing (``POST /documents``) or
before model inference (``POST /recognize``, ``POST /recognize/registration``):

1. **Decompression bomb / huge raster** — header dimensions are inspected
   *before* full pixel decoding so anomalous rasters are refused in sub-ms
   without allocating gigabytes of RAM.
2. **Micro-resolution frame** — ``min(width, height) < 150`` px.
3. **Extreme aspect ratio** — ``max(width, height) / min(width, height) > 10.0``.
4. **Blank / uniform frame** — near-zero luminance spread (``std < 2.0`` on a
   downscaled grayscale copy), catching solid black, white or flat grey frames.

Zero-retention note: every intermediate buffer created here (memory-view reader,
downscaled copy, grayscale copy, and ``bgr`` on rejection) is explicitly released
or zeroed in ``finally`` blocks. Error messages intentionally state only the
violated limit and never echo filenames, dimensions, or document metadata.
"""
from __future__ import annotations

import io
import struct
import warnings
from typing import Union

import cv2
import numpy as np

#: Maximum total pixel count allowed for an input frame (40 MP).
#: Comfortably covers 300–600 DPI A4 scans (600 DPI A4 is 4960x7016 = 34.8 MP)
#: and all repository samples/conformance fixtures while rejecting oversized
#: rasters and decompression bombs well below Pillow's ~89.5 MP threshold.
DEFAULT_MAX_PIXELS: int = 40_000_000

#: Maximum length of either side in pixels.
DEFAULT_MAX_SIDE: int = 8_000

#: Minimum length of the shorter side in pixels.
DEFAULT_MIN_SIDE: int = 150

#: Maximum ratio between the longer and shorter sides.
DEFAULT_MAX_ASPECT_RATIO: float = 10.0

#: Minimum grayscale standard deviation (0–255 scale) on a downscaled copy.
DEFAULT_MIN_BRIGHTNESS_STD: float = 2.0

_BLANK_CHECK_MAX_SIDE: int = 256

# Machine-readable reason codes returned in HTTP 422 responses.
IMAGE_UNDECODABLE = "image_undecodable"
IMAGE_TOO_LARGE = "image_too_large"
IMAGE_TOO_SMALL = "image_too_small"
IMAGE_EXTREME_ASPECT_RATIO = "image_extreme_aspect_ratio"
IMAGE_BLANK = "image_blank"

BytesLike = Union[bytes, bytearray, memoryview]


class ImageGuardError(ValueError):
    """Raised when an uploaded image fails a Layer A technical check."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


class _MemoryViewReader(io.RawIOBase):
    """Seekable read-only stream over a ``memoryview`` with zero extra copies."""

    def __init__(self, data: BytesLike):
        self._mv = memoryview(data)
        self._pos = 0
        self._len = len(self._mv)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._len + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if new_pos < 0:
            raise ValueError("Negative seek position")
        self._pos = new_pos
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            out = self._mv[self._pos:].tobytes()
            self._pos = self._len
            return out
        end = min(self._len, self._pos + size)
        out = self._mv[self._pos:end].tobytes()
        self._pos = end
        return out

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        n = min(len(b), self._len - self._pos)
        if n <= 0:
            return 0
        b[:n] = self._mv[self._pos:self._pos + n]
        self._pos += n
        return n

    def close(self) -> None:
        if not self.closed:
            self._mv.release()
            super().close()


def _probe_header_fallback(mv: memoryview) -> tuple[int, int] | None:
    """Lightweight header-only dimension parser for PNG, BMP, JPEG and WEBP.

    Used as a fallback when Pillow cannot open a truncated/malformed stream,
    ensuring crafted decompression-bomb headers are still refused before
    ``cv2.imdecode`` allocates memory.
    """
    n = len(mv)
    # PNG: signature (8) + length (4) + 'IHDR' (4) + width (4) + height (4)
    if n >= 24 and mv[:8].tobytes() == b"\x89PNG\r\n\x1a\n" and mv[12:16].tobytes() == b"IHDR":
        w, h = struct.unpack(">II", mv[16:24])
        return int(w), int(h)

    # BMP: 'BM' + file header (14) + DIB header
    if n >= 26 and mv[:2].tobytes() == b"BM":
        dib_size = struct.unpack("<I", mv[14:18])[0]
        if dib_size == 12:
            w, h = struct.unpack("<HH", mv[18:22])
            return int(w), int(h)
        if dib_size >= 40:
            w, h = struct.unpack("<ii", mv[18:26])
            return int(w), int(abs(h))

    # WEBP: 'RIFF' + size + 'WEBP' + chunk
    if n >= 30 and mv[:4].tobytes() == b"RIFF" and mv[8:12].tobytes() == b"WEBP":
        chunk = mv[12:16].tobytes()
        if chunk == b"VP8X":
            w = 1 + int.from_bytes(mv[24:27], "little")
            h = 1 + int.from_bytes(mv[27:30], "little")
            return w, h
        if chunk == b"VP8 " and mv[23:26].tobytes() == b"\x9d\x01\x2a":
            w = struct.unpack("<H", mv[26:28])[0] & 0x3FFF
            h = struct.unpack("<H", mv[28:30])[0] & 0x3FFF
            return int(w), int(h)
        if chunk == b"VP8L" and mv[20] == 0x2F:
            bits = struct.unpack("<I", mv[21:25])[0]
            return int((bits & 0x3FFF) + 1), int(((bits >> 14) & 0x3FFF) + 1)

    # JPEG: SOI (FF D8) followed by marker segments until SOF*
    if n >= 4 and mv[0] == 0xFF and mv[1] == 0xD8:
        pos = 2
        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3,
            0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB,
            0xCD, 0xCE, 0xCF,
        }
        while pos + 1 < n:
            if mv[pos] != 0xFF:
                break
            while pos < n and mv[pos] == 0xFF:
                pos += 1
            if pos >= n:
                break
            marker = mv[pos]
            pos += 1
            if marker in (0xD9, 0xDA):  # EOI / SOS
                break
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue
            if pos + 2 > n:
                break
            seg_len = struct.unpack(">H", mv[pos:pos + 2])[0]
            if seg_len < 2 or pos + seg_len > n:
                break
            if marker in sof_markers and seg_len >= 7:
                h, w = struct.unpack(">HH", mv[pos + 3:pos + 7])
                return int(w), int(h)
            pos += seg_len

    return None


def probe_header_dimensions(
    data: BytesLike,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
) -> tuple[int, int] | None:
    """Read ``(width, height)`` from the image header without decoding pixels.

    Raises ``ImageGuardError(IMAGE_TOO_LARGE, ...)`` immediately if Pillow flags
    a decompression bomb. Returns ``None`` only when the header cannot be parsed
    by either Pillow or the binary fallback (caller then lets ``cv2.imdecode``
    decide whether the file is corrupt).
    """
    from PIL import Image

    with _MemoryViewReader(data) as reader:
        try:
            with warnings.catch_warnings():
                # Suppress Pillow's warning locally: we enforce our own stricter
                # pixel/side limits below, and the warning text includes raw
                # dimensions which must not leak to stderr on zero-retention routes.
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(reader) as im:
                    w, h = im.size
                    return int(w), int(h)
        except Image.DecompressionBombError:
            raise ImageGuardError(
                IMAGE_TOO_LARGE,
                f"Image dimensions exceed the maximum supported size "
                f"(max {max_side} px per side, {max_pixels} pixels total)",
            ) from None
        except Exception:
            pass

    mv = memoryview(data)
    try:
        return _probe_header_fallback(mv)
    finally:
        mv.release()


def validate_dimensions(
    width: int,
    height: int,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
    min_side: int = DEFAULT_MIN_SIDE,
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
) -> None:
    """Validate ``(width, height)`` against Layer A physical dimension limits."""
    if width <= 0 or height <= 0:
        raise ImageGuardError(
            IMAGE_UNDECODABLE,
            "The image could not be decoded — it may be corrupt",
        )
    longer = max(width, height)
    shorter = min(width, height)
    if longer > max_side or (width * height) > max_pixels:
        raise ImageGuardError(
            IMAGE_TOO_LARGE,
            f"Image dimensions exceed the maximum supported size "
            f"(max {max_side} px per side, {max_pixels} pixels total)",
        )
    if shorter < min_side:
        raise ImageGuardError(
            IMAGE_TOO_SMALL,
            f"Image resolution is too small (minimum {min_side} px on the shorter side)",
        )
    if (longer / shorter) > max_aspect_ratio:
        raise ImageGuardError(
            IMAGE_EXTREME_ASPECT_RATIO,
            f"Image aspect ratio is too extreme (maximum {max_aspect_ratio:g}:1)",
        )


def validate_image_header(
    data: BytesLike,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
    min_side: int = DEFAULT_MIN_SIDE,
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
) -> tuple[int, int] | None:
    """Inspect header dimensions and reject out-of-bounds images before decoding."""
    dims = probe_header_dimensions(data, max_pixels=max_pixels, max_side=max_side)
    if dims is not None:
        validate_dimensions(
            dims[0], dims[1],
            max_pixels=max_pixels, max_side=max_side,
            min_side=min_side, max_aspect_ratio=max_aspect_ratio,
        )
    return dims


def check_not_blank(
    bgr: np.ndarray,
    *,
    min_brightness_std: float = DEFAULT_MIN_BRIGHTNESS_STD,
) -> None:
    """Reject completely blank, black, white, or flat uniform frames."""
    h, w = bgr.shape[:2]
    small = gray = None
    try:
        longer = max(h, w)
        if longer > _BLANK_CHECK_MAX_SIDE:
            scale = _BLANK_CHECK_MAX_SIDE / longer
            small_w = max(1, int(round(w * scale)))
            small_h = max(1, int(round(h * scale)))
            small = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
            src = small
        else:
            src = bgr

        if src.ndim == 2:
            gray = src
        elif src.shape[2] == 4:
            gray = cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        std = float(gray.std())
    finally:
        if gray is not None and gray is not bgr and gray is not small:
            gray.fill(0)
        if small is not None and small is not bgr:
            small.fill(0)

    if std < min_brightness_std:
        raise ImageGuardError(
            IMAGE_BLANK,
            "Image appears blank or uniform (insufficient visual contrast)",
        )


def decode_and_validate_bgr(
    data: BytesLike,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
    min_side: int = DEFAULT_MIN_SIDE,
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
    min_brightness_std: float = DEFAULT_MIN_BRIGHTNESS_STD,
) -> np.ndarray:
    """Header-check, decode to BGR, re-check dimensions, and check not blank.

    On any validation error every decoded pixel buffer is zeroed before raising.
    On success the caller owns the returned BGR array (and must zero it on
    zero-retention paths).
    """
    validate_image_header(
        data,
        max_pixels=max_pixels, max_side=max_side,
        min_side=min_side, max_aspect_ratio=max_aspect_ratio,
    )

    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0 or bgr.ndim < 2:
        raise ImageGuardError(
            IMAGE_UNDECODABLE,
            "The image could not be decoded — it may be corrupt",
        )

    try:
        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        validate_dimensions(
            w, h,
            max_pixels=max_pixels, max_side=max_side,
            min_side=min_side, max_aspect_ratio=max_aspect_ratio,
        )
        check_not_blank(bgr, min_brightness_std=min_brightness_std)
    except Exception:
        bgr.fill(0)
        raise

    return bgr


def validate_image_bytes(
    data: BytesLike,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_side: int = DEFAULT_MAX_SIDE,
    min_side: int = DEFAULT_MIN_SIDE,
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO,
    min_brightness_std: float = DEFAULT_MIN_BRIGHTNESS_STD,
) -> tuple[int, int]:
    """Run all Layer A checks on encoded image bytes and return ``(width, height)``."""
    bgr = decode_and_validate_bgr(
        data,
        max_pixels=max_pixels, max_side=max_side,
        min_side=min_side, max_aspect_ratio=max_aspect_ratio,
        min_brightness_std=min_brightness_std,
    )
    try:
        return int(bgr.shape[1]), int(bgr.shape[0])
    finally:
        bgr.fill(0)
