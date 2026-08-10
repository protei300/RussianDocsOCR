"""Binary artifacts: the uploaded original and the rendered canvas.

**This layer stays on the filesystem even after a SQL migration.** Multi-megabyte
PNGs do not belong in a database — in a real deployment this module grows an S3
implementation, not a BLOB column. That is why it is separate from
``documents.py`` rather than folded into it.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from service.core.database import FileStore, atomic_write_bytes

log = logging.getLogger(__name__)

#: Formats we accept, keyed by the magic bytes that actually identify them.
#: Sniffed rather than trusting the client's Content-Type, which is
#: attacker-controlled and routinely wrong even when it isn't.
_MAGIC: tuple[tuple[bytes, str, str], ...] = (
    (b"\xff\xd8\xff", ".jpg", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
    (b"BM", ".bmp", "image/bmp"),
    (b"II*\x00", ".tif", "image/tiff"),
    (b"MM\x00*", ".tif", "image/tiff"),
)


def sniff_image(data: bytes) -> tuple[str, str] | None:
    """``(extension, media type)`` for supported images, else ``None``.

    WEBP needs a two-part check: 'RIFF' at 0 and 'WEBP' at 8.
    """
    for magic, ext, media in _MAGIC:
        if data.startswith(magic):
            return ext, media
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    return None


def is_pdf(data: bytes) -> bool:
    """Detected separately so the error can say *why* — users will try PDFs."""
    return data.startswith(b"%PDF")


def doc_dir(db: FileStore, doc_id: int) -> Path:
    directory = db.doc_dir(doc_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_original(db: FileStore, doc_id: int, data: bytes, ext: str) -> Path:
    """Store the upload byte-for-byte under a fixed name.

    The client's filename is kept on the record for display only and never
    touches the filesystem — so it cannot be a path-traversal vector no matter
    what it contains.
    """
    path = doc_dir(db, doc_id) / f"original{ext}"
    atomic_write_bytes(path, data)
    return path


def decode_dimensions(data: bytes) -> tuple[int, int] | None:
    """``(width, height)`` of the upload, or ``None`` if it cannot be decoded.

    Done synchronously at upload time so an undecodable file becomes an
    immediate, actionable 422 instead of a mysterious failed job minutes later.
    """
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return int(image.shape[1]), int(image.shape[0])


def save_canvas(db: FileStore, doc_id: int, rgb: np.ndarray) -> tuple[Path, int, int]:
    """Write the corrected canvas as PNG.

    ``img_with_fixed_perspective`` is **RGB**; ``cv2.imwrite`` interprets its
    input as **BGR**. Skipping this conversion swaps red and blue in every
    displayed document — and the result looks plausible enough on a passport
    that it can ship unnoticed. Hence the explicit ``cvtColor`` and the
    regression test that asserts a known-red pixel stays red.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    path = doc_dir(db, doc_id) / "canvas.png"
    # The temp name must keep the real extension: cv2.imwrite picks its encoder
    # from the suffix, so writing to 'canvas.png.tmp' fails with
    # "could not find a writer for the specified extension".
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    os.replace(tmp, path)
    return path, int(rgb.shape[1]), int(rgb.shape[0])


def save_thumbnail(db: FileStore, doc_id: int, rgb: np.ndarray, width: int = 96) -> Path:
    """Small JPEG for the list page.

    Without this the log page pulls full canvases for every visible row on each
    3-second poll — megabytes per refresh for images rendered at 56 px wide.
    """
    height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    small = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    path = doc_dir(db, doc_id) / "thumb.jpg"
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")   # see save_canvas
    cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    os.replace(tmp, path)
    return path


def open_artifact(db: FileStore, doc_id: int, kind: str) -> tuple[Path, str] | None:
    """``(path, media type)`` for ``original`` | ``canvas`` | ``thumb``."""
    directory = db.doc_dir(doc_id)
    if kind == "canvas":
        # PNG for anything this service rendered; JPEG for the pre-computed seed
        # fixtures, which trade exactness for a committable repository footprint.
        for name, media in (("canvas.png", "image/png"), ("canvas.jpg", "image/jpeg")):
            path = directory / name
            if path.is_file():
                return path, media
        return None
    if kind == "thumb":
        path = directory / "thumb.jpg"
        if path.is_file():
            return path, "image/jpeg"
        return open_artifact(db, doc_id, "canvas")      # fall back to full size
    if kind == "original":
        for candidate in sorted(directory.glob("original.*")):
            if candidate.suffix == ".tmp":
                continue
            sniffed = sniff_image(candidate.read_bytes()[:16])
            return candidate, (sniffed[1] if sniffed else "application/octet-stream")
    return None


def load_result(db: FileStore, doc_id: int) -> dict[str, Any] | None:
    import json
    path = db.doc_dir(doc_id) / "result.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        log.exception("[STORE] unreadable result.json for doc %s", doc_id)
        return None
