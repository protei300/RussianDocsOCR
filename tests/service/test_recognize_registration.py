"""POST /recognize/registration: a registration page in, stamps and their fields out, nothing kept.

The reading is replaced by a fake (the real one needs the field network, the
stamp OCR and the dictionaries, which are not in git); what is checked here is
the route's contract - the answer, nothing stored, nothing logged, buffers
zeroed, a missing installation reported as 503 - plus the pure parts of the page
reader (order and links). The real reader runs in the --runslow test at the end.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

P = "/api/v1"
SECRET_TEXT = "ЛЕНИНА"


def _jpeg(w: int = 320, h: int = 200) -> bytes:
    img = np.full((h, w, 3), 220, np.uint8)
    img[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 40
    return cv2.imencode(".jpg", img)[1].tobytes()


def _files_under(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def _pin(client: TestClient) -> dict[str, str]:
    token = client.post(f"{P}/auth/pin-login", json={"pin": "1234"}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def fake(monkeypatch):
    from service import worker
    from service.core import logging as service_logging
    from service.ml import runtime
    from service.repositories import recognitions

    async def no_runtime():
        return None
    monkeypatch.setattr(worker, "_init_runtime_bg", no_runtime)
    monkeypatch.setattr(service_logging, "_buffer", deque(maxlen=service_logging._buffer.maxlen))
    recognitions.reset_for_tests()
    seen: dict = {"arrays": []}

    def recognise_registration(rgb, *, docconf, img_size, lease_timeout):
        seen["arrays"].append(rgb)
        if seen.get("raise"):
            raise seen["raise"]
        return {
            "doc_type": "INTPASSPORTADDR_ALL", "decision": "registration", "recognised": True,
            "doc_conf": 0.95, "types_in_frame": ["INTPASSPORTADDR_ALL"], "notes": [],
            "coord_space": "input", "pages": [], "order_by": "date",
            "stamps": [{"index": 1, "page": 0, "kind": "registered", "conf": 0.97, "quad": None,
                        "fields": [{"name": "street", "display": "Street", "value": SECRET_TEXT}],
                        "ocr": {"street": SECRET_TEXT}, "boxes": [], "cancels": None, "current": True}],
            "quality": {}, "timings": {},
        }
    monkeypatch.setattr(runtime, "recognise_registration", recognise_registration)
    return seen


def test_answers_and_stores_nothing(app_factory, fake):
    with TestClient(app_factory("pin", LOG_LEVEL="DEBUG")) as client:
        h = _pin(client)
        from service.core.config import get_settings
        root = Path(get_settings().data_dir)
        before = _files_under(root)
        resp = client.post(f"{P}/recognize/registration", content=_jpeg(),
                           headers={**h, "Content-Type": "image/jpeg"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["stored"] is False and body["decision"] == "registration"
        assert body["stamps"][0]["ocr"]["street"] == SECRET_TEXT and body["stamps"][0]["current"] is True
        assert "processing_ms" in body and resp.headers["cache-control"] == "no-store"
        assert _files_under(root) == before
        entries = client.get(f"{P}/logs", params={"n": 2000}, headers=h).json()["entries"]
        text = json.dumps(entries, ensure_ascii=False)
        assert SECRET_TEXT not in text and "INTPASSPORTADDR" not in text
        lines = [e["message"] for e in entries if e["message"].startswith("[API] recognize")]
        assert len(lines) == 1 and "outcome=recognised status=200" in lines[0]


def test_buffers_are_zeroed(app_factory, fake):
    with TestClient(app_factory("pin")) as client:
        h = _pin(client)
        resp = client.post(f"{P}/recognize/registration", content=_jpeg(),
                           headers={**h, "Content-Type": "image/jpeg"})
        assert resp.status_code == 200
        assert not fake["arrays"][0].any(), "the decoded page was not zeroed"


def test_missing_installation_is_503_naming_the_variables(app_factory, fake):
    from service.ml.errors import RegistrationUnavailable
    fake["raise"] = RegistrationUnavailable(["RDOCS_STAMP_FIELDS_MODEL", "RDOCS_STAMP_OCR"])
    with TestClient(app_factory("pin")) as client:
        resp = client.post(f"{P}/recognize/registration", content=_jpeg(),
                           headers={**_pin(client), "Content-Type": "image/jpeg"})
        assert resp.status_code == 503
        assert "RDOCS_STAMP_FIELDS_MODEL" in resp.json()["detail"]


def test_not_an_image_is_415(app_factory, fake):
    with TestClient(app_factory("pin")) as client:
        resp = client.post(f"{P}/recognize/registration", content=b"not an image at all",
                           headers={**_pin(client), "Content-Type": "application/octet-stream"})
        assert resp.status_code == 415 and not fake["arrays"]


def test_needs_a_role(app_factory, fake):
    from test_auth_security import _account, _admin

    with TestClient(app_factory("users")) as client:
        # Anonymous is refused with 401.
        anon = client.post(f"{P}/recognize/registration", content=_jpeg(),
                           headers={"Content-Type": "image/jpeg"})
        assert anon.status_code == 401

        admin = _admin(client)
        viewer = _account(client, admin, "regviewer", "viewer")
        operator = _account(client, admin, "regop", "operator")
        key = client.post(f"{P}/api-keys", headers=admin, json={"label": "reg-test"}).json()["key"]

        # Viewer cannot invoke recognition (403); operator and API key can (200).
        as_viewer = client.post(f"{P}/recognize/registration", content=_jpeg(),
                                headers={**viewer, "Content-Type": "image/jpeg"})
        assert as_viewer.status_code == 403

        as_operator = client.post(f"{P}/recognize/registration", content=_jpeg(),
                                  headers={**operator, "Content-Type": "image/jpeg"})
        assert as_operator.status_code == 200, as_operator.text

        as_key = client.post(f"{P}/recognize/registration", content=_jpeg(),
                             headers={"X-API-Key": key, "Content-Type": "image/jpeg"})
        assert as_key.status_code == 200, as_key.text


# --------------------------------------------------------------------------- order and links

def _stamp(n, kind, date=None):
    return {"index": n, "kind": kind, "ocr": {"date": date} if date else {}}


def test_order_by_date_freshest_first_and_links():
    from service.ml.registration.page import RegistrationReader
    st = [_stamp(1, "registered", "01.02.2010"), _stamp(2, "deregistered", "05.06.2015"),
          _stamp(3, "registered", "07.06.2015")]
    assert RegistrationReader._order_and_links(st) == "date"
    assert [s["index"] for s in st] == [3, 2, 1]
    by = {s["index"]: s for s in st}
    assert by[2]["cancels"] == 1 and by[3]["current"] and not by[1]["current"]


def test_order_by_place_when_a_date_is_missing():
    from service.ml.registration.page import RegistrationReader
    st = [_stamp(1, "registered", "01.02.2010"), _stamp(2, "deregistered")]
    assert RegistrationReader._order_and_links(st) == "position"
    by = {s["index"]: s for s in st}
    assert by[2]["cancels"] == 1 and not any(s["current"] for s in st)


def test_order_and_links_edge_cases():
    from service.ml.registration.page import RegistrationReader

    # Empty list -> None
    empty: list[dict] = []
    assert RegistrationReader._order_and_links(empty) is None

    # parsed['date'] takes precedence over raw ocr['date']; orphan deregistered & unclear handled
    st = [
        {"index": 1, "kind": "deregistered", "ocr": {"date": "01.01.2010"}},
        {"index": 2, "kind": "unclear", "ocr": {"date": "02.02.2012"}},
        {"index": 3, "kind": "registered", "ocr": {"date": "bad"}, "parsed": {"date": "03.03.2015"}},
    ]
    assert RegistrationReader._order_and_links(st) == "date"
    assert [s["index"] for s in st] == [3, 2, 1]
    by = {s["index"]: s for s in st}
    assert by[1]["cancels"] is None and by[2]["cancels"] is None
    assert by[3]["current"] is True


def test_registration_reader_cut_decisions_and_notes():
    from types import SimpleNamespace
    from service.ml.registration.config import RegistrationConfig
    from service.ml.registration.page import ADDR, RegistrationReader

    reader = RegistrationReader(RegistrationConfig())
    rgb = np.full((200, 300, 3), 220, np.uint8)

    def _res(doctype, quality=None):
        return SimpleNamespace(
            doctype=doctype,
            quality=quality or {"DocConf": 0.9, "Blur": "good", "Glare": "good"},
            geometry=None,
            img_with_fixed_perspective=rgb,
            _meta_results={},
            to_input=lambda pts: pts,
        )

    def _pipe(results, best_stamp_conf=0.0):
        obbs = [[150.0, 100.0, 80.0, 50.0, 0.0, best_stamp_conf, 0, "registered"]] if best_stamp_conf else []
        det = SimpleNamespace(
            model_name="RegistrationStamps",
            KINDS={"registered": "registered"},
            predict=lambda img: {"RegistrationStamps": {"obbox": obbs}},
        )
        cls = SimpleNamespace(
            model_name="DocTypeAngles",
            predict=lambda img: {"DocTypeAngles": {"doc_type": "NONE", "doc_type_confidence": 0.1}},
        )
        return SimpleNamespace(
            process_frame=lambda *a, **kw: results,
            registration_stamps=det,
            doctype_angles=cls,
        )

    # 1) Multiple different document types -> multiple_documents + note
    cut_multi = reader.cut(_pipe([_res(ADDR), _res("DL_2020")]), rgb)
    assert cut_multi["decision"] == "multiple_documents" and cut_multi["stamps"] == []
    ans_multi = reader.read_stamps(cut_multi)
    assert "one_document_expected" in ans_multi["notes"] and ans_multi["recognised"] is False

    # 2) Non-registration document -> not_registration + note
    cut_other = reader.cut(_pipe([_res("INTPASSPORT_2011", {"DocConf": 0.95, "Blur": "bad"})]), rgb)
    assert cut_other["decision"] == "not_registration"
    ans_other = reader.read_stamps(cut_other)
    assert set(ans_other["notes"]) >= {"not_registration_page", "quality_low"}

    # 3) NONE with weak stamp -> no_document
    cut_none = reader.cut(_pipe([_res("NONE")], best_stamp_conf=0.1), rgb)
    assert cut_none["decision"] == "no_document"
    assert "no_document" in reader.read_stamps(cut_none)["notes"]

    # 4) NONE with confident stamp -> stamp_fragment
    cut_frag = reader.cut(_pipe([_res("NONE")], best_stamp_conf=0.92), rgb)
    assert cut_frag["decision"] == "stamp_fragment" and len(cut_frag["stamps"]) == 1

    # 5) _pages splits a wide spread into 2 pages & detects cut page outside frame
    wide = np.full((100, 260, 3), 220, np.uint8)
    spread_res = SimpleNamespace(
        img_with_fixed_perspective=wide,
        _meta_results={},
        to_input=lambda pts: pts,
    )
    pages = reader._pages(spread_res, wide)
    assert len(pages) == 2 and pages[0]["x0"] == 0.0 and pages[1]["x0"] == 130.0

    # 6) _page_kind trusts confident stamp detector (>= stamp_conf_fallback=0.92) over NONE classifier
    kind_low, _, _ = reader._page_kind(_pipe([_res("NONE")], best_stamp_conf=0.90), rgb)
    assert kind_low == "not_registration"
    kind_hi, t, _ = reader._page_kind(_pipe([_res("NONE")], best_stamp_conf=0.95), rgb)
    assert kind_hi == "registration" and t == "NONE"


# --------------------------------------------------------------------------- the real reader

def test_partial_ocr_installation_names_missing_metadata(tmp_path):
    from service.ml.registration.config import RegistrationConfig

    for name in ('fields.onnx', 'model.onnx'):
        (tmp_path / name).touch()
    cfg = RegistrationConfig(fields_model=tmp_path / 'fields.onnx', ocr_dir=tmp_path)
    assert cfg.missing() == ['RDOCS_STAMP_OCR']
    (tmp_path / 'model.json').touch()
    assert cfg.missing() == []


def test_real_runtime_reports_missing_installation_before_loading_reader(tmp_path, monkeypatch):
    import sys
    from service.ml import runtime
    from service.ml.errors import RegistrationUnavailable

    monkeypatch.setattr(runtime, '_REGISTRATION', None)
    monkeypatch.setenv('RDOCS_STAMP_FIELDS_MODEL', str(tmp_path / 'missing.onnx'))
    # Missing artifacts must be reported before importing optional reader dependencies.
    monkeypatch.setitem(sys.modules, 'service.ml.registration.page', None)
    with pytest.raises(RegistrationUnavailable) as error:
        runtime.registration_reader()
    assert 'RDOCS_STAMP_FIELDS_MODEL' in error.value.missing

@pytest.mark.slow
def test_real_reader_end_to_end(app_factory):
    """The real reader through the route: the synthetic registration spread of the
    conformance suite (no personal data) gives stamps with fields; a passport main
    page is answered as not a registration page. Skipped where the reader's files
    (not in git) are missing."""
    import time

    from service.ml import runtime
    from service.ml.registration.config import RegistrationConfig

    if RegistrationConfig().missing():
        pytest.skip("registration reader files are not on this machine")
    repo = Path(__file__).resolve().parents[2]
    spread = repo / "conformance/material/INTPASSPORTADDR_ALL/synth_01_INTPASSPORTADDR_ALL.jpg"
    passport = sorted((repo / "samples/INTPASSPORT_2011").glob("*.jpg"))[0]
    with TestClient(app_factory("pin", load_runtime=True, PROCESSING_MODE="memory")) as client:
        h = _pin(client)
        deadline = time.time() + 300
        while not runtime.is_ready() and time.time() < deadline:
            time.sleep(1)
        assert runtime.is_ready()
        resp = client.post(f"{P}/recognize/registration", content=spread.read_bytes(),
                           headers={**h, "Content-Type": "image/jpeg"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["decision"] == "registration" and body["stamps"]
        assert any(s["fields"] for s in body["stamps"])
        assert sum(s["current"] for s in body["stamps"]) <= 1
        resp = client.post(f"{P}/recognize/registration", content=passport.read_bytes(),
                           headers={**h, "Content-Type": "image/jpeg"})
        assert resp.status_code == 200 and resp.json()["decision"] == "not_registration"
        assert "not_registration_page" in resp.json()["notes"]
        for pipe in list(runtime._POOL.queue):
            assert getattr(pipe, "results", None) is None
