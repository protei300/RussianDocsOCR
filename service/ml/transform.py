"""Turn ``PipelineResults`` into a JSON-safe, client-shaped view model.

This module is pure and has no I/O, so it is unit-testable without loading
215 MB of models: feed it a recorded ``PipelineResults`` (or a stub with the
same attributes) and assert on the dict.

**It must be called while the pipeline lease is still held.** ``results`` *is*
``pipeline.results``, and the next ``process_img()`` call rebinds it — reading
it after releasing the lease is a live data race that returns another
document's fields. See ``runtime.recognise``.

Numeric types — verified empirically, not assumed
-------------------------------------------------
Measured across all 7 document types in ``samples/`` on 2026-08-03
(``scratchpad/check_types.py``):

===============================  ======================  =========================
value                            actual type             json.dumps without a cast
===============================  ======================  =========================
``bbox[i][0:4]``, ``bbox[i][5]``  ``int``                 works
``bbox[i][4]`` (confidence)       ``numpy.float64``       works
``bbox[i][6]``                    ``str``                 works
``Quality['DocConf']``            ``numpy.float64``       works
``Quality[Glare|Blur|*Spoofing]`` ``str`` (``'good'``)    works
===============================  ======================  =========================

So ``json.dumps(results.full_report)`` already succeeds unmodified — an earlier
design note claiming ``np.float32`` breaks serialisation was wrong
(``np.float64`` subclasses Python ``float``; ``np.float32`` does not, but does
not appear here). The casts below are kept for two *other* reasons:

1. **Rounding.** A raw ``float64`` serialises with 17 significant digits of
   noise. Rounding on the server keeps the wire format stable and is what lets
   a future Go implementation reproduce byte-identical golden files.
2. **Future-proofing.** If the library switches a dtype, an explicit cast fails
   loudly here rather than silently emitting something a strict client rejects.

The address branch (``obbox``, ``p_handwritten``) is **not** covered by that
measurement: ``samples/`` contains no ``INTPASSPORTADDR``. Those values are
therefore cast defensively via ``_num``, which handles both native and numpy
types. Confirm the real dtypes when an anonymised address-page sample exists.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from service.ml import labels

#: Decimal places for every float on the wire. Chosen so that a port in
#: another language can reproduce the JSON exactly — float formatting differs
#: between runtimes past this precision, and golden-file comparison then fails
#: for reasons that have nothing to do with recognition.
FLOAT_PRECISION = 4


def _num(value: Any, precision: int = FLOAT_PRECISION) -> float | None:
    """Coerce anything float-ish (Python, numpy) to a rounded Python float."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None  # JSON has no NaN/Infinity; emit null rather than invalid JSON
    return round(out, precision)


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str(value: Any) -> str | None:
    return None if value is None else str(value)


def json_default(obj: Any) -> Any:
    """Last-resort hook for ``json.dumps``.

    The explicit casts above are the contract — they document what the library
    returns. This is the safety net for a field added by a future library
    version that nobody has cast yet: better a slightly-wrong-looking number in
    the response than a 500 from the serialiser.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Boxes
# ---------------------------------------------------------------------------
def _build_boxes(results: Any, ocr: dict[str, str]) -> list[dict[str, Any]]:
    """Axis-aligned text-field boxes, in *canvas* pixel coordinates.

    Coordinate space matters and is easy to get wrong: these come from the
    detector run against ``img_with_fixed_perspective``, so they line up
    pixel-for-pixel with the canvas image the service serves — and with nothing
    else. They cannot be mapped back onto the original upload, because the
    library does not retain the deskew angle.

    Several boxes can share one label (split fields such as ``Birth_place_ru``,
    and the doubled ``Licence_number`` on internal passports — the pipeline
    de-duplicates the *field*, not the boxes). The text is therefore attached
    to the highest-confidence box only, and the rest are flagged ``ambiguous``
    so a client can grey them out instead of repeating the same string.
    """
    meta = results.text_fields_meta
    if not meta or not meta.get("bbox"):
        return []

    raw = []
    for row in meta["bbox"]:
        # row layout: [x1, y1, x2, y2, conf, cls_idx, label]
        raw.append({
            "x1": _int(row[0]), "y1": _int(row[1]),
            "x2": _int(row[2]), "y2": _int(row[3]),
            "conf": _num(row[4]),
            "cls": _int(row[5]),
            "label": _str(row[6]) or "",
        })

    # Decide which box owns the recognised text for each label.
    best_by_label: dict[str, int] = {}
    for i, b in enumerate(raw):
        prev = best_by_label.get(b["label"])
        if prev is None or (b["conf"] or 0) > (raw[prev]["conf"] or 0):
            best_by_label[b["label"]] = i

    boxes = []
    for i, b in enumerate(raw):
        label = b["label"]
        owns_text = best_by_label.get(label) == i
        boxes.append({
            "id": f"b{i}",
            "label": label,
            "display": labels.field_display(label),
            "kind": "visual" if label in labels.NON_TEXT_LABELS else "text",
            "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"],
            "conf": b["conf"],
            "cls": b["cls"],
            "text": ocr.get(label) if owns_text else None,
            "ambiguous": (label in ocr) and not owns_text,
        })
    return boxes


def _build_address(results: Any) -> dict[str, Any] | None:
    """Oriented address-line boxes plus their printed/handwritten verdicts.

    Only present for ``INTPASSPORTADDR``. The two source lists can desynchronise:
    ``_address_lines`` skips empty patches with a bare ``continue`` without
    recording a slot, so ``obbox[i]`` and ``Address_lines[i]`` stop lining up.
    Zipping them blindly would caption boxes with the wrong text, so when the
    lengths disagree we set ``aligned=False``, drop the geometry and keep only
    the text — a client that respects the flag then suppresses the overlay
    rather than drawing a confident lie.
    """
    meta = results.meta_results.get("AddressLinesDetector") or {}
    obboxes = meta.get("obbox") or []
    lines = results.meta_results.get("Address_lines") or []
    if not obboxes and not lines:
        return None

    aligned = len(obboxes) == len(lines)
    out_lines = []
    for i, line in enumerate(lines):
        entry: dict[str, Any] = {
            "id": f"o{i}",
            "kind": _str(line.get("kind")),
            "text": _str(line.get("text")),
            "p_handwritten": _num(line.get("p_handwritten")),
            "obbox": None,
        }
        if aligned:
            # row layout: [cx, cy, w, h, angle_rad, conf, cls_idx, label]
            row = obboxes[i]
            entry["obbox"] = {
                "cx": _num(row[0]), "cy": _num(row[1]),
                "w": _num(row[2]), "h": _num(row[3]),
                "angle_rad": _num(row[4], precision=6),  # radians: needs finer precision
                "conf": _num(row[5]),
                "label": _str(row[7]) if len(row) > 7 else None,
            }
        out_lines.append(entry)

    return {"aligned": aligned, "lines": out_lines}


def _build_fields(doc_type: str | None, ocr: dict[str, str],
                  boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recognised fields as an ordered array, each linked to its box(es).

    An array rather than a dict, deliberately: JSON objects have no guaranteed
    order, and even if they did, the library's insertion order is not the order
    a human reads the document in. The ordering lives in ``labels.FIELD_ORDER``.

    ``box_ids`` is a list because one field can legitimately own several boxes.
    """
    by_label: dict[str, list[str]] = {}
    conf_by_label: dict[str, float | None] = {}
    for b in boxes:
        by_label.setdefault(b["label"], []).append(b["id"])
        if b["text"] is not None:
            conf_by_label[b["label"]] = b["conf"]

    fields = []
    for name in labels.order_fields(doc_type, list(ocr.keys())):
        fields.append({
            "name": name,
            "display": labels.field_display(name),
            "value": _str(ocr.get(name)),
            "script": labels.field_script(name),
            "conf": conf_by_label.get(name),
            "box_ids": by_label.get(name, []),
        })
    return fields


def build_viewmodel(results: Any, *, device: str | None = None,
                    include_debug: bool = False) -> dict[str, Any]:
    """``PipelineResults`` -> the dict the service stores and serves.

    Must be called inside the pipeline lease (see module docstring).

    The canvas image itself is *not* included — it is a numpy array and is
    persisted separately by the artifact layer. Its dimensions are, because the
    client needs them to scale the box overlay.
    """
    report = results.full_report
    doc_type = _str(report.get("DocType"))
    ocr = {str(k): str(v) for k, v in (report.get("OCR") or {}).items()}

    boxes = _build_boxes(results, ocr)

    quality: dict[str, Any] = {}
    for key, value in (report.get("Quality") or {}).items():
        # Glare/Blur/*Spoofing are the strings 'good'/'bad'; DocConf is a float.
        quality[str(key)] = _num(value) if isinstance(value, (int, float, np.generic)) else _str(value)

    timings = {str(k): _num(v) for k, v in (report.get("Timings") or {}).items()}

    canvas_w = canvas_h = None
    canvas_fallback = False
    try:
        canvas = results.img_with_fixed_perspective
        if canvas is not None:
            canvas_h, canvas_w = int(canvas.shape[0]), int(canvas.shape[1])
    except (KeyError, AttributeError, IndexError):
        # Short-circuited runs (doctype == 'NONE') never populate the warped
        # image, and the property raises rather than returning None.
        canvas_fallback = True

    payload: dict[str, Any] = {
        "doc_type": doc_type,
        "doc_type_base": labels.base_doc_type(doc_type) or None,
        "doc_type_era": labels.doc_type_era(doc_type),
        "recognised": bool(doc_type) and doc_type != "NONE",
        "device": _str(device),
        "canvas": {"width": canvas_w, "height": canvas_h, "is_fallback": canvas_fallback},
        "coord_space": "canvas",
        "coord_space_note": (
            "Box coordinates are in canvas pixel space and match the canvas image "
            "exactly. They cannot be mapped onto the original upload: the library "
            "does not retain the deskew angle."
        ),
        "boxes": boxes,
        "fields": _build_fields(doc_type, ocr, boxes),
        "ocr": ocr,
        "quality": quality,
        "timings": timings,
        "address": _build_address(results),
    }

    if include_debug:
        segm = (results.meta_results.get("DocDetector") or {}).get("segm")
        payload["debug"] = {
            "doc_outline": {
                # Explicitly tagged: this polygon is in the pre-perspective-warp
                # space, NOT canvas space. Without the tag someone will
                # eventually draw it on the canvas and file a bug.
                "coord_space": "prewarp",
                "polygon": [np.asarray(c).reshape(-1, 2).astype(int).tolist() for c in segm]
                if segm is not None else None,
            }
        }

    return payload


def build_search_text(filename: str, payload: dict[str, Any]) -> str:
    """Lowercased haystack for the list page's free-text search.

    Precomputed at write time so filtering never has to parse the stored result
    blob. In a SQL backend this becomes an indexed computed column.
    """
    parts = [filename, payload.get("doc_type") or ""]
    parts.extend(str(v) for v in (payload.get("ocr") or {}).values())
    address = payload.get("address")
    if address:
        parts.extend(str(line.get("text") or "") for line in address.get("lines", []))
    return " ".join(parts).lower()
