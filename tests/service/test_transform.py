"""Tests for the recognition-result transform.

Two layers:

* Pure unit tests against stub result objects — fast, no models, no GPU.
* An opt-in integration test that runs the real pipeline over ``samples/``.
  Marked ``slow`` because it loads 215 MB of models and takes ~30 s; run it
  with ``pytest -m slow`` or ``pytest --runslow``.

The central assertion in both is the same: **the view model must survive
``json.dumps`` without the ``default=`` escape hook.** That hook exists as a
safety net for future library changes, but if it is ever load-bearing we have
lost track of what the library returns — which is exactly the drift these tests
exist to catch.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from service.ml import labels, transform  # noqa: E402

SAMPLES = pathlib.Path(__file__).resolve().parents[2] / "samples"


class StubResults:
    """Minimal stand-in for ``PipelineResults``.

    Mirrors only the attributes ``build_viewmodel`` touches. Types are chosen to
    match what the real library returns, as measured on 2026-08-03: ints for box
    corners, ``numpy.float64`` for confidences, ``str`` for quality verdicts.
    """

    def __init__(self, *, doc_type="INTPASSPORT_2011", ocr=None, bbox=None,
                 quality=None, timings=None, canvas_shape=(701, 505, 3),
                 meta_extra=None):
        self._bbox = bbox
        self.meta_results = {
            "DocType": doc_type,
            "OCR": ocr or {},
            "Quality": quality if quality is not None else {
                "DocConf": np.float64(0.9312345678901234),
                "Glare": "good", "Blur": "good",
                "PrintSpoofing": "good", "LCDSpoofing": "good",
            },
        }
        if bbox is not None:
            self.meta_results["TextFieldsDetector"] = {"bbox": bbox}
        if meta_extra:
            self.meta_results.update(meta_extra)
        self._timings = timings or {"_ocr": 0.31, "total": 0.4812345}
        self._canvas = (np.zeros(canvas_shape, dtype=np.uint8)
                        if canvas_shape else None)

    @property
    def full_report(self):
        return {
            "DocType": self.meta_results.get("DocType"),
            "OCR": self.meta_results.get("OCR"),
            "Quality": self.meta_results.get("Quality"),
            "Timings": self._timings,
        }

    @property
    def text_fields_meta(self):
        return self.meta_results.get("TextFieldsDetector")

    @property
    def img_with_fixed_perspective(self):
        if self._canvas is None:
            raise KeyError("DocDetector")  # what the real property does
        return self._canvas


def _box(x1, y1, x2, y2, conf, cls, label):
    """A bbox row shaped exactly like the library's."""
    return [x1, y1, x2, y2, np.float64(conf), cls, label]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def test_viewmodel_serialises_without_the_numpy_hook():
    results = StubResults(
        ocr={"Last_name_ru": "БАТУРИНА", "Birth_date": "01.01.1990"},
        bbox=[_box(10, 20, 100, 40, 0.9251, 1, "Last_name_ru"),
              _box(10, 60, 100, 80, 0.8812, 10, "Birth_date")],
    )
    payload = transform.build_viewmodel(results, device="gpu")
    # No default= on purpose: this is the contract.
    json.dumps(payload, ensure_ascii=False)


def test_floats_are_rounded_for_cross_language_reproducibility():
    """17 digits of float64 noise would break golden-file comparison in a port."""
    results = StubResults(bbox=[_box(0, 0, 10, 10, 0.123456789012345, 1, "Sex_ru")])
    payload = transform.build_viewmodel(results)
    assert payload["boxes"][0]["conf"] == 0.1235
    assert payload["quality"]["DocConf"] == 0.9312
    assert payload["timings"]["total"] == 0.4812


def test_cyrillic_survives_round_trip():
    results = StubResults(ocr={"Last_name_ru": "ЖЁЛТЫЙ ЩЪЫЬЭЮЯ"})
    payload = transform.build_viewmodel(results)
    assert json.loads(json.dumps(payload))["ocr"]["Last_name_ru"] == "ЖЁЛТЫЙ ЩЪЫЬЭЮЯ"


def test_nan_becomes_null_not_invalid_json():
    """json.dumps would happily emit bare NaN, which is not valid JSON."""
    results = StubResults(bbox=[_box(0, 0, 10, 10, float("nan"), 1, "Sex_ru")])
    payload = transform.build_viewmodel(results)
    assert payload["boxes"][0]["conf"] is None
    assert "NaN" not in json.dumps(payload)


# ---------------------------------------------------------------------------
# Field / box linking
# ---------------------------------------------------------------------------
def test_shared_label_marks_all_but_the_best_box_ambiguous():
    """Split fields and the doubled Licence_number on internal passports.

    Observed live on INTPASSPORT_1997 (4 shared-label boxes) — without this the
    same string would be repeated across several boxes as if each were a
    separate reading.
    """
    results = StubResults(
        ocr={"Licence_number": "4011 123456"},
        bbox=[_box(0, 0, 10, 10, 0.70, 5, "Licence_number"),
              _box(0, 20, 10, 30, 0.95, 5, "Licence_number")],
    )
    payload = transform.build_viewmodel(results)
    by_id = {b["id"]: b for b in payload["boxes"]}
    assert by_id["b1"]["text"] == "4011 123456"   # highest confidence owns it
    assert by_id["b0"]["text"] is None
    assert by_id["b0"]["ambiguous"] is True
    assert by_id["b1"]["ambiguous"] is False
    # ...and the field points at both boxes, so hovering highlights the pair.
    field = next(f for f in payload["fields"] if f["name"] == "Licence_number")
    assert field["box_ids"] == ["b0", "b1"]


def test_non_text_boxes_are_flagged_and_carry_no_value():
    results = StubResults(ocr={}, bbox=[_box(0, 0, 50, 60, 0.94, 0, "Face"),
                                        _box(0, 70, 50, 90, 0.88, 9, "Signature")])
    payload = transform.build_viewmodel(results)
    assert {b["kind"] for b in payload["boxes"]} == {"visual"}
    assert all(b["text"] is None for b in payload["boxes"])


def test_fields_are_in_document_reading_order_not_dict_order():
    results = StubResults(ocr={
        "Issue_date": "01.02.2015", "Last_name_ru": "А", "Birth_date": "03.04.1990",
    })
    payload = transform.build_viewmodel(results)
    assert [f["name"] for f in payload["fields"]] == [
        "Last_name_ru", "Birth_date", "Issue_date",
    ]


def test_unknown_fields_are_kept_not_dropped():
    """A field the label map doesn't know about must still reach the client."""
    results = StubResults(ocr={"Last_name_ru": "А", "Zzz_future_field": "x"})
    payload = transform.build_viewmodel(results)
    names = [f["name"] for f in payload["fields"]]
    assert names == ["Last_name_ru", "Zzz_future_field"]
    assert payload["fields"][1]["display"] == "Zzz_future_field"


def test_script_marker_drives_font_choice():
    results = StubResults(ocr={"Last_name_ru": "А", "Last_name_en": "A",
                               "Licence_number": "1234"})
    payload = transform.build_viewmodel(results)
    script = {f["name"]: f["script"] for f in payload["fields"]}
    assert script["Last_name_ru"] == "ru"
    assert script["Last_name_en"] == "en"
    assert script["Licence_number"] == "num"   # monospace, digits only


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------
def test_unrecognised_document_does_not_crash():
    results = StubResults(doc_type="NONE", ocr={}, bbox=None, canvas_shape=None)
    payload = transform.build_viewmodel(results)
    assert payload["recognised"] is False
    assert payload["boxes"] == []
    assert payload["canvas"]["is_fallback"] is True
    json.dumps(payload)


def test_desynchronised_address_lines_suppress_geometry():
    """obbox and Address_lines can drift apart when a patch comes back empty.

    Zipping them anyway would caption boxes with another line's text, so the
    transform drops the geometry and says so.
    """
    results = StubResults(doc_type="INTPASSPORTADDR_ALL", meta_extra={
        "AddressLinesDetector": {"obbox": [[10.0, 20.0, 30.0, 8.0, 0.01, 0.9, 0,
                                            "Living_region_ru"]]},
        "Address_lines": [
            {"kind": "printed", "p_handwritten": 0.02, "text": "line one"},
            {"kind": "handwritten", "p_handwritten": 0.98, "text": None},
        ],
    })
    payload = transform.build_viewmodel(results)
    assert payload["address"]["aligned"] is False
    assert all(line["obbox"] is None for line in payload["address"]["lines"])
    assert payload["address"]["lines"][0]["text"] == "line one"


def test_aligned_address_lines_keep_geometry():
    results = StubResults(doc_type="INTPASSPORTADDR_ALL", meta_extra={
        "AddressLinesDetector": {"obbox": [[10.0, 20.0, 30.0, 8.0, 0.0123456, 0.9,
                                            0, "Living_region_ru"]]},
        "Address_lines": [{"kind": "printed", "p_handwritten": 0.02, "text": "г. Москва"}],
    })
    payload = transform.build_viewmodel(results)
    assert payload["address"]["aligned"] is True
    obb = payload["address"]["lines"][0]["obbox"]
    # Angles keep more precision than other floats — a 4-dp radian is ~0.006°
    # of slop, visible on a long line.
    assert obb["angle_rad"] == 0.012346
    json.dumps(payload)


def test_debug_outline_is_tagged_with_its_own_coordinate_space():
    """DocDetector.segm is pre-warp, unlike every other coordinate we emit."""
    results = StubResults(meta_extra={
        "DocDetector": {"segm": [np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32)]},
    })
    payload = transform.build_viewmodel(results, include_debug=True)
    assert payload["debug"]["doc_outline"]["coord_space"] == "prewarp"
    json.dumps(payload)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("doc_type,base,era", [
    ("INTPASSPORT_2011", "INTPASSPORT", "2011"),
    ("INTPASSPORTADDR_ALL", "INTPASSPORTADDR", "ALL"),
    ("EXTPASSPORTBIO_2007", "EXTPASSPORTBIO", "2007"),
    ("NONE", "NONE", None),
    (None, "", None),
])
def test_doc_type_is_split_without_the_librarys_valueerror(doc_type, base, era):
    """pipeline.py raises on a label with no underscore; we must not."""
    assert labels.base_doc_type(doc_type) == base
    assert labels.doc_type_era(doc_type) == era


# ---------------------------------------------------------------------------
# Integration (opt-in)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(not SAMPLES.is_dir(), reason="samples/ not present")
def test_real_documents_serialise_cleanly():
    from service.ml import runtime

    info = runtime.init_runtime(compute_device="auto", ocr_mode="accurate")
    assert info.state == "ready", info.error
    try:
        images = [next(iter(sorted(d.glob("*.jpg"))))
                  for d in sorted(SAMPLES.iterdir()) if d.is_dir()
                  and any(d.glob("*.jpg"))]
        assert images, "no sample images found"
        for image in images:
            payload, canvas = runtime.recognise(image)
            json.dumps(payload, ensure_ascii=False)   # no default= — the contract
            assert payload["doc_type"], f"{image} produced no doc type"
            if payload["recognised"]:
                assert canvas is not None
                assert payload["canvas"]["width"] == canvas.shape[1]
                assert payload["canvas"]["height"] == canvas.shape[0]
    finally:
        runtime.shutdown_runtime()
