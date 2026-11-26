"""Tests for Layer B document suitability quality gate (library + service API).

All fixtures are 100% synthetic (zero PII).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from document_processing.geometry import Chain, Scale
from document_processing.pipeline.pipeline import Pipeline
from document_processing.pipeline.quality_gate import (
    DEFAULT_THRESHOLDS,
    evaluate_quality_gate,
    normalize_quality_gate_mode,
)
from document_processing.pipeline.results import PipelineResults


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


# ---------------------------------------------------------------------------
# Service layer tests: transform, settings schema, /recognize, /documents/{id}
# ---------------------------------------------------------------------------

def test_build_viewmodel_preserves_14_keys_by_default_and_adds_quality_gate_when_requested():
    from service.ml import transform

    res = PipelineResults()
    res._meta_results["DocType"] = "DL_2020"
    res._meta_results["Quality"] = {
        "DocConf": 0.98, "Glare": "good", "Blur": "good",
        "PrintSpoofing": "REAL", "LCDSpoofing": "REAL",
    }
    res._meta_results["QualityGate"] = {
        "status": "warn",
        "mode": "warn",
        "stopped": False,
        "reasons": [{"code": "glare", "severity": "warn", "value": np.float64(0.1071428), "threshold": 0.07}],
        "metrics": {"glare_coef": np.float64(0.1071428), "quad_clipped_edges": np.int64(0)},
    }

    vm_default = transform.build_viewmodel(res, device="cpu")
    assert "quality_gate" not in vm_default
    assert len(vm_default) == 14

    vm_with_gate = transform.build_viewmodel(res, device="cpu", include_quality_gate=True)
    assert "quality_gate" in vm_with_gate
    gate = vm_with_gate["quality_gate"]
    assert gate["status"] == "warn"
    assert gate["reasons"][0]["value"] == 0.1071
    assert gate["metrics"]["quad_clipped_edges"] == 0
    # Must be plain JSON-serialisable
    json.dumps(vm_with_gate)


def test_settings_schema_includes_quality_gate_mode():
    from service.core.settings_schema import SCHEMA_BY_KEY, SettingValidationError, coerce

    assert "quality_gate_mode" in SCHEMA_BY_KEY
    assert SCHEMA_BY_KEY["quality_gate_mode"].default == "warn"
    assert coerce("quality_gate_mode", "reject") == "reject"
    assert coerce("quality_gate_mode", "off") == "off"
    with pytest.raises(SettingValidationError):
        coerce("quality_gate_mode", "invalid")


@pytest.mark.service
def test_recognize_and_documents_endpoints_surface_quality_gate(tmp_path, monkeypatch):
    from collections import deque
    import importlib
    from fastapi.testclient import TestClient
    from service.core import auth as auth_core
    from service.core import config
    from service.core import logging as service_logging
    from service.ml import runtime
    from service.repositories import documents as doc_repo
    from service.repositories import recognitions
    from service import worker

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AUTH_MODE", "pin")
    monkeypatch.setenv("AUTH_PIN", "1234")
    monkeypatch.setenv("JWT_SECRET", "test-secret-for-quality-gate-endpoint")
    monkeypatch.setenv("PROCESSING_MODE", "stored")
    monkeypatch.setenv("SEED_SAMPLES", "-1")
    monkeypatch.setenv("WARMUP_IMAGE", "")
    monkeypatch.setenv("COMPUTE_DEVICE", "cpu")
    config.get_settings.cache_clear()
    auth_core._attempts.clear()
    auth_core._process_secret = None
    monkeypatch.setattr(service_logging, "_buffer", deque(maxlen=service_logging._buffer.maxlen))

    async def no_runtime():
        return None
    monkeypatch.setattr(worker, "_init_runtime_bg", no_runtime)
    recognitions.reset_for_tests()

    seen = {}

    def fake_recognise_image(rgb, *, docconf, img_size, with_canvas, lease_timeout, quality_gate="warn"):
        seen["quality_gate"] = quality_gate
        return {
            "doc_type": "DL_2020", "doc_type_base": "DL", "doc_type_era": "2020",
            "recognised": True, "device": "cpu",
            "canvas": {"width": 900, "height": 600, "is_fallback": False},
            "coord_space": "canvas", "coord_space_note": "",
            "boxes": [], "fields": [], "ocr": {},
            "quality": {"DocConf": 0.96, "Blur": "good", "Glare": "good"},
            "quality_gate": {
                "status": "reject", "mode": quality_gate,
                "stopped": quality_gate == "reject",
                "reasons": [{"code": "blur", "severity": "reject", "value": 0.08, "threshold": 0.2}],
                "metrics": {"blur_coef": 0.08},
            },
            "timings": {}, "address": None,
        }, None

    monkeypatch.setattr(runtime, "recognise_image", fake_recognise_image)

    import service.main as main
    importlib.reload(main)
    app = main.app
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(_sharp_doc_rgb(400, 300), cv2.COLOR_RGB2BGR))
    assert ok
    img_bytes = enc.tobytes()

    with TestClient(app) as client:
        token = client.post("/api/v1/auth/pin-login", json={"pin": "1234"}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Default mode comes from settings ('warn')
        r_warn = client.post("/api/v1/recognize", content=img_bytes,
                             headers={**headers, "Content-Type": "image/jpeg"})
        assert r_warn.status_code == 200
        assert seen["quality_gate"] == "warn"
        body_warn = r_warn.json()
        assert body_warn["quality_gate"]["status"] == "reject"
        assert body_warn["quality_gate"]["stopped"] is False

        # Explicit query param override (?quality_gate=reject)
        r_rej = client.post("/api/v1/recognize?quality_gate=reject", content=img_bytes,
                            headers={**headers, "Content-Type": "image/jpeg"})
        assert r_rej.status_code == 200
        assert seen["quality_gate"] == "reject"
        assert r_rej.json()["quality_gate"]["stopped"] is True

        # Invalid mode rejected with 400
        r_bad = client.post("/api/v1/recognize?quality_gate=bogus", content=img_bytes,
                            headers={**headers, "Content-Type": "image/jpeg"})
        assert r_bad.status_code == 400

        # Stored document detail returns quality_gate from record.result
        from service.core.database import get_store
        db = get_store()
        doc_id = doc_repo.reserve_id(db)
        rec = doc_repo.create(
            db, doc_id=doc_id, filename="synthetic.jpg",
            content_type="image/jpeg", size_bytes=len(img_bytes),
            original_ext=".jpg", original_w=400, original_h=300,
            search_text="synthetic.jpg",
        )
        doc_repo.save_result(
            db, rec,
            payload={
                "doc_type": "DL_2020", "recognised": True,
                "ocr": {}, "fields": [], "quality": {"DocConf": 0.96, "Blur": "good"},
                "quality_gate": {"status": "pass", "mode": "warn", "stopped": False, "reasons": [], "metrics": {}},
                "canvas": {"width": 400, "height": 300, "is_fallback": False},
            },
            search_text="synthetic.jpg dl_2020",
            processing_ms=42,
        )
        r_doc = client.get(f"/api/v1/documents/{rec.id}", headers=headers)
        assert r_doc.status_code == 200
        assert r_doc.json()["quality_gate"]["status"] == "pass"

    auth_core._attempts.clear()
    auth_core._process_secret = None
    config.get_settings.cache_clear()
