"""Tests for Layer B document suitability quality gate (library only).

Service-side viewmodel, settings schema, and HTTP tests live in
``tests/service/test_quality_gate_api.py``. All fixtures are 100% synthetic.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from document_processing.geometry import Chain, Scale
from document_processing.pipeline.pipeline import Pipeline
from document_processing.pipeline.quality_gate import (
    evaluate_quality_gate,
    normalize_quality_gate_mode,
)


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def _sharp_doc_rgb(width: int = 900, height: int = 600, bg: int = 215, fg: int = 35) -> np.ndarray:
    """Synthetic document-like RGB canvas with crisp text-like horizontal bars and grid."""
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    for y in range(60, height - 60, 42):
        for x in range(60, width - 80, 48):
            cv2.rectangle(img, (x, y), (x + 26, y + 14), (fg, fg, fg), -1)
    cv2.rectangle(img, (35, 35), (width - 35, height - 35), (fg, fg, fg), 3)
    return img


def _white_paper_scan_rgb(width: int = 900, height: int = 600) -> np.ndarray:
    """Clean white-paper scan (~85% >=250 white background + crisp dark ink lines)."""
    img = np.full((height, width, 3), 254, dtype=np.uint8)
    for y in range(70, height - 70, 55):
        cv2.line(img, (70, y), (width - 70, y), (25, 25, 25), 3)
        for x in range(80, width - 90, 60):
            cv2.rectangle(img, (x, y - 16), (x + 20, y - 4), (30, 30, 30), -1)
    return img


def _rect_contour(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)


def _reason_map(gate: dict) -> dict[str, dict]:
    return {r["code"]: r for r in gate["reasons"]}


# ---------------------------------------------------------------------------
# Unit tests: document_processing.pipeline.quality_gate
# ---------------------------------------------------------------------------

def test_clean_document_passes():
    img = _sharp_doc_rgb()
    gate = evaluate_quality_gate(
        doc_img=img,
        frame_shape=(1000, 1400),
        document_box=[100, 100, 1100, 800],
        segm=[_rect_contour(100, 100, 1100, 800)],
        pre_borders_geom=Chain(()),
        doc_conf=0.98,
        quality_scores={"Blur": 0.96, "Glare": 0.0},
        mode="warn",
    )
    assert gate["status"] == "pass"
    assert gate["mode"] == "warn"
    assert gate["stopped"] is False
    assert gate["reasons"] == []
    assert gate["metrics"]["doc_min_side"] == 700.0
    assert gate["metrics"]["doc_max_side"] == 1000.0


def test_normalize_quality_gate_mode():
    assert normalize_quality_gate_mode(None) == "warn"
    assert normalize_quality_gate_mode("WARN") == "warn"
    assert normalize_quality_gate_mode(" reject ") == "reject"
    assert normalize_quality_gate_mode("off") == "off"
    with pytest.raises(ValueError):
        normalize_quality_gate_mode("strict")


@pytest.mark.parametrize("box,expected_sev", [
    ([0, 0, 240, 400], "reject"),   # min_side < 250
    ([0, 0, 260, 340], "reject"),   # max_side < 350
    ([0, 0, 360, 600], "warn"),     # min_side < 380
    ([0, 0, 400, 520], "warn"),     # max_side < 550
    ([0, 0, 400, 600], None),       # clean
])
def test_doc_too_small_thresholds(box, expected_sev):
    gate = evaluate_quality_gate(
        doc_img=_sharp_doc_rgb(),
        frame_shape=(1200, 1600),
        document_box=box,
        doc_conf=0.95,
        quality_scores={"Blur": 0.95, "Glare": 0.0},
        mode="reject",
    )
    rmap = _reason_map(gate)
    if expected_sev is None:
        assert "doc_too_small" not in rmap
        assert gate["status"] == "pass"
        assert gate["stopped"] is False
    else:
        assert rmap["doc_too_small"]["severity"] == expected_sev
        assert gate["status"] == expected_sev
        assert gate["stopped"] is (expected_sev == "reject")


def test_blur_recalibrated_coef_and_laplacian_fallback():
    sharp = _sharp_doc_rgb()

    # Legacy Blur='bad' range (0.60..0.90) must pass Layer B (cuts 23% false rejects)
    gate_ok = evaluate_quality_gate(
        doc_img=sharp,
        frame_shape=(1000, 1400),
        document_box=[50, 50, 1050, 750],
        doc_conf=0.95,
        quality_scores={"Blur": 0.65, "Glare": 0.0},
    )
    assert gate_ok["status"] == "pass"

    # Warn band: blur_coef in [0.20, 0.50)
    gate_warn = evaluate_quality_gate(
        doc_img=sharp,
        frame_shape=(1000, 1400),
        document_box=[50, 50, 1050, 750],
        doc_conf=0.95,
        quality_scores={"Blur": 0.35, "Glare": 0.0},
        mode="reject",
    )
    assert _reason_map(gate_warn)["blur"]["severity"] == "warn"
    assert gate_warn["stopped"] is False

    # Reject band via neural blur_coef < 0.20
    gate_rej = evaluate_quality_gate(
        doc_img=sharp,
        frame_shape=(1000, 1400),
        document_box=[50, 50, 1050, 750],
        doc_conf=0.95,
        quality_scores={"Blur": 0.12, "Glare": 0.0},
        mode="reject",
    )
    assert _reason_map(gate_rej)["blur"]["severity"] == "reject"
    assert gate_rej["stopped"] is True

    # Reject via Laplacian fallback when blur_coef is absent on heavily blurred image
    heavy_blur = cv2.GaussianBlur(sharp, (0, 0), sigmaX=7.0)
    gate_lap = evaluate_quality_gate(
        doc_img=heavy_blur,
        frame_shape=(1000, 1400),
        document_box=[50, 50, 1050, 750],
        doc_conf=0.95,
        quality_scores={},
        mode="reject",
    )
    assert _reason_map(gate_lap)["blur"]["severity"] == "reject"
    assert gate_lap["stopped"] is True


def test_photometry_dark_bright_contrast_and_white_scan_protection():
    # Clean white-paper scan (>75% >=250 pixels, high dynamic range from dark ink) must PASS
    white_scan = _white_paper_scan_rgb()
    gate_white = evaluate_quality_gate(
        doc_img=white_scan,
        frame_shape=(1000, 1400),
        document_box=[50, 50, 1050, 750],
        doc_conf=0.95,
        quality_scores={"Blur": 0.95, "Glare": 0.0},
    )
    assert gate_white["metrics"]["photo_bright250_frac"] > 0.75
    assert "too_bright" not in _reason_map(gate_white)
    assert gate_white["status"] == "pass"

    # Severely washed-out overexposure (>75% >=250 AND low dynamic range <40) -> reject
    washed = np.full((600, 900, 3), 253, dtype=np.uint8)
    cv2.rectangle(washed, (100, 100), (300, 200), (225, 225, 225), -1)
    gate_over = evaluate_quality_gate(
        doc_img=washed,
        frame_shape=(1000, 1400),
        document_box=[50, 50, 1050, 750],
        doc_conf=0.95,
        quality_scores={"Blur": 0.95, "Glare": 0.0},
        mode="reject",
    )
    rmap_over = _reason_map(gate_over)
    assert rmap_over["too_bright"]["severity"] == "reject"
    assert rmap_over["low_contrast"]["severity"] == "reject"
    assert gate_over["stopped"] is True

    # Severely darkened frame -> too_dark reject + low_contrast reject
    dark = (_sharp_doc_rgb().astype(np.float32) * 0.12).astype(np.uint8)
    gate_dark = evaluate_quality_gate(
        doc_img=dark,
        frame_shape=(1000, 1400),
        document_box=[50, 50, 1050, 750],
        doc_conf=0.95,
        quality_scores={"Blur": 0.95, "Glare": 0.0},
        mode="reject",
    )
    rmap_dark = _reason_map(gate_dark)
    assert rmap_dark["too_dark"]["severity"] == "reject"
    assert gate_dark["stopped"] is True


@pytest.mark.parametrize("conf,expected_sev", [
    (0.20, "reject"),   # < 0.35
    (0.34, "reject"),   # < 0.35
    (0.35, "warn"),     # [0.35, 0.75)
    (0.50, "warn"),     # [0.35, 0.75)
    (0.74, "warn"),     # [0.35, 0.75)
    (0.75, None),       # >= 0.75
])
def test_low_doc_conf_decision_035_reject_075_warn(conf, expected_sev):
    gate = evaluate_quality_gate(
        doc_img=_sharp_doc_rgb(),
        frame_shape=(1000, 1400),
        document_box=[50, 50, 1050, 750],
        doc_conf=conf,
        quality_scores={"Blur": 0.95, "Glare": 0.0},
        mode="reject",
    )
    rmap = _reason_map(gate)
    if expected_sev is None:
        assert "low_doc_conf" not in rmap
        assert gate["status"] == "pass"
        assert gate["stopped"] is False
    else:
        assert rmap["low_doc_conf"]["severity"] == expected_sev
        assert gate["status"] == expected_sev
        assert gate["stopped"] is (expected_sev == "reject")


def test_glare_and_perspective_are_warn_only():
    # 1 glare tile (1/28 = 0.0357) passes; 2+ tiles (>=0.07) warns; even 0.50 never rejects
    gate_1tile = evaluate_quality_gate(
        doc_img=_sharp_doc_rgb(),
        frame_shape=(1000, 1400),
        document_box=[50, 50, 1050, 750],
        doc_conf=0.95,
        quality_scores={"Blur": 0.95, "Glare": 1.0 / 28.0},
        mode="reject",
    )
    assert "glare" not in _reason_map(gate_1tile)

    skewed_quad = np.array([[[250, 100]], [[950, 100]], [[1150, 750]], [[50, 750]]], dtype=np.float32)
    gate_warn = evaluate_quality_gate(
        doc_img=_sharp_doc_rgb(),
        frame_shape=(1000, 1400),
        document_box=[50, 50, 1150, 750],
        segm=[skewed_quad],
        pre_borders_geom=Chain(()),
        doc_conf=0.95,
        quality_scores={"Blur": 0.95, "Glare": 0.50},
        mode="reject",
    )
    rmap = _reason_map(gate_warn)
    assert rmap["glare"]["severity"] == "warn"
    assert rmap["extreme_perspective"]["severity"] == "warn"
    assert gate_warn["status"] == "warn"
    assert gate_warn["stopped"] is False


def test_doc_clipped_is_info_only_and_never_raises_status_to_warn():
    # Quad touching the left and top borders of the original frame (within <= 2 px)
    clipped_quad = _rect_contour(1.0, 0.5, 900.0, 650.0)
    gate = evaluate_quality_gate(
        doc_img=_sharp_doc_rgb(),
        frame_shape=(1000, 1400),
        document_box=[0, 0, 900, 650],
        segm=[clipped_quad],
        pre_borders_geom=Chain(()),
        doc_conf=0.98,
        quality_scores={"Blur": 0.96, "Glare": 0.0},
        mode="reject",
    )
    rmap = _reason_map(gate)
    assert "doc_clipped" in rmap
    assert rmap["doc_clipped"]["severity"] == "info"
    assert rmap["doc_clipped"]["value"] == 2
    assert gate["status"] == "pass"
    assert gate["stopped"] is False


def test_doc_clipped_maps_quad_through_pre_borders_geometry():
    # Pre-borders image was scaled by 0.5x from a 1400x1000 original frame
    # Quad edge at x=0.5 in scaled space -> x=1.0 in original space (clipped <= 2 px)
    scaled_quad = _rect_contour(0.5, 100.0, 450.0, 380.0)
    gate = evaluate_quality_gate(
        doc_img=_sharp_doc_rgb(),
        frame_shape=(1000, 1400),
        document_box=[0, 200, 900, 760],
        segm=[scaled_quad],
        pre_borders_geom=Chain((Scale(0.5, 0.5),)),
        doc_conf=0.98,
        quality_scores={"Blur": 0.96, "Glare": 0.0},
        mode="warn",
    )
    assert _reason_map(gate)["doc_clipped"]["severity"] == "info"
    assert gate["status"] == "pass"


# ---------------------------------------------------------------------------
# Pipeline integration tests (stubbed models, zero weight loading)
# ---------------------------------------------------------------------------

class _StubModel:
    def __init__(self, model_name: str, payload):
        self.model_name = model_name
        self._payload = payload

    def predict(self, img):
        return {self.model_name: self._payload}


def _make_stub_pipeline(blur_coef: float = 0.95, glare_coef: float = 0.0, doc_conf: float = 0.98):
    pipe = object.__new__(Pipeline)
    pipe.probe = None
    pipe.page_registrar = None
    pipe.page_geometry = None
    pipe.document_detector = None
    pipe._shared_executor = None

    pipe.glare = _StubModel("Glare", ("bad" if glare_coef > 0 else "good", glare_coef))
    pipe.blur = _StubModel("Blur", ("good" if blur_coef > 0.9 else "bad", blur_coef))
    pipe.print_spoofing = _StubModel("PrintSpoofing", ("REAL", 0.99))
    pipe.lcd_spoofing = _StubModel("LCDSpoofing", ("REAL", 0.99))

    calls = {"fields": 0, "ocr": 0}

    def fake_doctype_angle(img):
        pipe.results._meta_results["DocType"] = "DL_2020"
        pipe.results._meta_results["Quality"]["DocConf"] = doc_conf
        pipe.results._meta_results["Angle90"] = {
            "angle": 0,
            "confidence": 0.99,
            "warped_img": img,
            "geometry": Chain(()),
        }

    def fake_doc_detector(img):
        h, w = img.shape[:2]
        pipe.results._meta_results["DocDetector"] = {
            "warped_img": img,
            "segm": [_rect_contour(20, 20, w - 20, h - 20)],
        }

    def fake_deskew(img):
        return None

    def fake_borders_geometry():
        return Chain(())

    def fake_fields_detector(img, rotate_licence=False):
        calls["fields"] += 1
        pipe.results._meta_results["TextFieldsDetector"] = {"bbox": [], "warped_img": []}

    def fake_split_words(tf, dt):
        pipe.results._meta_results["WordsDetector"] = {}

    def fake_ocr(words, dt):
        calls["ocr"] += 1

    class _StubDocDet:
        def predict_transform(self, img, stack="auto", max_pages=1):
            h, w = img.shape[:2]
            return {
                "DocDetector": {
                    "warped_img": img,
                    "segm": [_rect_contour(20, 20, w - 20, h - 20)],
                }
            }

    pipe.doc_detector = _StubDocDet()
    pipe._doctype_angle = fake_doctype_angle
    pipe._doc_detector = fake_doc_detector
    pipe._deskew = fake_deskew
    pipe._borders_geometry = fake_borders_geometry
    pipe._fields_detector = fake_fields_detector
    pipe._split_words = fake_split_words
    pipe._ocr = fake_ocr
    return pipe, calls


def test_pipeline_quality_gate_warn_reject_off_and_contract_preservation():
    frame = _sharp_doc_rgb(900, 600)

    # 1. Default soft mode ('warn'): records QualityScores + QualityGate, keeps Quality keys exact, runs Fields
    pipe_warn, calls_warn = _make_stub_pipeline(blur_coef=0.10, glare_coef=0.10, doc_conf=0.95)
    res_warn = pipe_warn.process_img(frame, quality_gate="warn")
    assert set(res_warn.quality.keys()) == {"DocConf", "Glare", "Blur", "PrintSpoofing", "LCDSpoofing"}
    assert res_warn.quality_scores["Blur"] == pytest.approx(0.10)
    assert res_warn.quality_scores["Glare"] == pytest.approx(0.10)
    assert res_warn.quality_gate["status"] == "reject"
    assert res_warn.quality_gate["stopped"] is False
    assert calls_warn["fields"] == 1
    # Encapsulation: mutating returned dict does not affect stored state
    res_warn.quality_gate["status"] = "tampered"
    assert res_warn.quality_gate["status"] == "reject"

    # 2. Strict mode ('reject'): stops BEFORE Fields/OCR when reject-level reason fires
    pipe_rej, calls_rej = _make_stub_pipeline(blur_coef=0.10, glare_coef=0.0, doc_conf=0.95)
    res_rej = pipe_rej.process_img(frame, quality_gate="reject")
    assert res_rej.quality_gate["status"] == "reject"
    assert res_rej.quality_gate["stopped"] is True
    assert calls_rej["fields"] == 0
    assert res_rej.ocr is None

    # 3. Off mode ('off'): quality_gate is None and Fields run normally
    pipe_off, calls_off = _make_stub_pipeline(blur_coef=0.10, glare_coef=0.0, doc_conf=0.95)
    res_off = pipe_off.process_img(frame, quality_gate="off")
    assert res_off.quality_gate is None
    assert calls_off["fields"] == 1
