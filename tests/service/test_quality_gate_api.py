"""Service-layer tests for Layer B quality gate: viewmodel, settings schema, HTTP routes."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from document_processing.pipeline.results import PipelineResults


def _sharp_doc_bgr(width: int = 400, height: int = 300) -> np.ndarray:
    img = np.full((height, width, 3), 215, dtype=np.uint8)
    for y in range(40, height - 40, 36):
        for x in range(40, width - 50, 42):
            cv2.rectangle(img, (x, y), (x + 20, y + 12), (35, 35, 35), -1)
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (35, 35, 35), 3)
    return img


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
    json.dumps(vm_with_gate)


def test_settings_schema_includes_quality_gate_mode():
    from service.core.settings_schema import SCHEMA_BY_KEY, SettingValidationError, coerce

    assert "quality_gate_mode" in SCHEMA_BY_KEY
    assert SCHEMA_BY_KEY["quality_gate_mode"].default == "warn"
    assert coerce("quality_gate_mode", "reject") == "reject"
    assert coerce("quality_gate_mode", "off") == "off"
    with pytest.raises(SettingValidationError):
        coerce("quality_gate_mode", "invalid")


def test_recognize_and_documents_endpoints_surface_quality_gate(app_factory, monkeypatch):
    from service.core.database import get_store
    from service.ml import runtime
    from service.repositories import documents as doc_repo
    from service.repositories import recognitions

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

    ok, enc = cv2.imencode(".jpg", _sharp_doc_bgr())
    assert ok
    img_bytes = enc.tobytes()

    with TestClient(app_factory("pin", PROCESSING_MODE="stored")) as client:
        token = client.post("/api/v1/auth/pin-login", json={"pin": "1234"}).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        r_warn = client.post("/api/v1/recognize", content=img_bytes,
                             headers={**headers, "Content-Type": "image/jpeg"})
        assert r_warn.status_code == 200
        assert seen["quality_gate"] == "warn"
        body_warn = r_warn.json()
        assert body_warn["quality_gate"]["status"] == "reject"
        assert body_warn["quality_gate"]["stopped"] is False

        r_rej = client.post("/api/v1/recognize?quality_gate=reject", content=img_bytes,
                            headers={**headers, "Content-Type": "image/jpeg"})
        assert r_rej.status_code == 200
        assert seen["quality_gate"] == "reject"
        assert r_rej.json()["quality_gate"]["stopped"] is True

        r_bad = client.post("/api/v1/recognize?quality_gate=bogus", content=img_bytes,
                            headers={**headers, "Content-Type": "image/jpeg"})
        assert r_bad.status_code == 400

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
