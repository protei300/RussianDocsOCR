"""Unit tests for the background worker (failure classification, retries, progress, deletion)."""
from __future__ import annotations

import asyncio
import contextlib

import numpy as np
import pytest

from service import worker
from service.core import database
from service.ml import runtime
from service.ml.errors import PipelineBusy, RecognitionError, RuntimeNotReady
from service.repositories import artifacts
from service.repositories import documents as repo
from service.repositories import settings as settings_repo


def _transient_recognition_error(msg: str) -> RecognitionError:
    err = RecognitionError(msg)
    err.transient = True  # type: ignore[attr-defined]
    return err


@pytest.mark.parametrize("exc,expected", [
    (asyncio.TimeoutError(), ("TIMEOUT", True)),
    (PipelineBusy("busy"), ("BUSY", True)),
    (RuntimeNotReady("loading"), ("BUSY", True)),
    (RecognitionError("bad jpeg"), ("RECOGNITION_FAILED", False)),
    (_transient_recognition_error("gpu hiccup"), ("RECOGNITION_FAILED", True)),
    (RuntimeError("CUDA out of memory"), ("GPU_ERROR", True)),
    (RuntimeError("cuDNN error"), ("GPU_ERROR", True)),
    (OSError("disk hiccup"), ("IO_ERROR", True)),
    (ValueError("unexpected bug"), ("INTERNAL", False)),
])
def test_classify_failure_separates_transient_and_deterministic(exc, expected):
    assert worker._classify_failure(exc) == expected


def test_progress_tracks_step_and_clears():
    doc_id = 987654
    assert worker.get_document_progress(doc_id) is None
    worker._set_step(doc_id, "recognizing")
    try:
        prog = worker.get_document_progress(doc_id)
        assert prog is not None and prog["step"] == "recognizing"
        assert 5.0 <= prog["pct"] <= 95.0
        assert prog["eta_sec"] >= 0.0
    finally:
        worker._clear_step(doc_id)
    assert worker.get_document_progress(doc_id) is None


def _patch_store(monkeypatch, store):
    @contextlib.contextmanager
    def _session():
        yield store

    monkeypatch.setattr(worker, "get_store", lambda: store)
    monkeypatch.setattr(worker, "db_session", _session)
    monkeypatch.setattr(worker, "_EXECUTOR", None)


def _seed_doc(store, name="doc.jpg"):
    doc_id = repo.reserve_id(store)
    artifacts.save_original(store, doc_id, b"\xff\xd8\xff" + b"\x00" * 32, ".jpg")
    return repo.create(store, doc_id=doc_id, filename=name, content_type="image/jpeg",
                       size_bytes=35, original_ext=".jpg")


def test_process_document_success_saves_result_and_artifacts(tmp_path, monkeypatch):
    store = database.init_store(tmp_path / "data", wipe=True)
    _patch_store(monkeypatch, store)
    rec = _seed_doc(store)

    canvas = np.full((40, 60, 3), 180, np.uint8)
    payload = {
        "doc_type": "DL_2020", "recognised": True,
        "quality": {"DocConf": 0.95},
        "canvas": {"width": 60, "height": 40, "is_fallback": False},
        "fields": [{"name": "Last_name_ru", "value": "ИВАНОВ"}],
        "ocr": {"Last_name_ru": "ИВАНОВ"},
    }
    monkeypatch.setattr(runtime, "recognise", lambda *a, **kw: (payload, canvas))

    asyncio.run(worker._process_document(rec.id))

    done = repo.get_by_id(store, rec.id)
    assert done.status == "done" and done.doc_type == "DL_2020"
    assert artifacts.open_artifact(store, rec.id, "canvas") is not None
    assert artifacts.open_artifact(store, rec.id, "thumb") is not None
    assert worker.get_document_progress(rec.id) is None


def test_process_document_retries_transient_until_max_retries_then_fails(tmp_path, monkeypatch):
    store = database.init_store(tmp_path / "data", wipe=True)
    _patch_store(monkeypatch, store)
    settings_repo.bulk_update(store, {"max_retries": "2"})
    rec = _seed_doc(store)

    def busy(*a, **kw):
        raise PipelineBusy("pool saturated")

    monkeypatch.setattr(runtime, "recognise", busy)

    # Attempt 1 -> requeued (retry_count=1)
    asyncio.run(worker._process_document(rec.id))
    r1 = repo.get_by_id(store, rec.id)
    assert r1.status == "queued" and r1.retry_count == 1 and r1.error_code == "BUSY"

    # Attempt 2 -> requeued (retry_count=2)
    asyncio.run(worker._process_document(rec.id))
    r2 = repo.get_by_id(store, rec.id)
    assert r2.status == "queued" and r2.retry_count == 2

    # Attempt 3 -> retries exhausted -> failed
    asyncio.run(worker._process_document(rec.id))
    r3 = repo.get_by_id(store, rec.id)
    assert r3.status == "failed" and r3.error_code == "BUSY" and r3.finished_at is not None


def test_process_document_fails_immediately_on_deterministic_error(tmp_path, monkeypatch):
    store = database.init_store(tmp_path / "data", wipe=True)
    _patch_store(monkeypatch, store)
    settings_repo.bulk_update(store, {"max_retries": "3"})
    rec = _seed_doc(store)

    def deterministic_boom(*a, **kw):
        raise ValueError("broken tensor shape")

    monkeypatch.setattr(runtime, "recognise", deterministic_boom)
    asyncio.run(worker._process_document(rec.id))

    failed = repo.get_by_id(store, rec.id)
    assert failed.status == "failed" and failed.retry_count == 0 and failed.error_code == "INTERNAL"


def test_process_document_ignores_document_deleted_mid_flight(tmp_path, monkeypatch):
    store = database.init_store(tmp_path / "data", wipe=True)
    _patch_store(monkeypatch, store)
    rec = _seed_doc(store)

    def recognise_while_deleted(*a, **kw):
        repo.delete(store, repo.get_by_id(store, rec.id))
        return {"doc_type": "DL_2020", "recognised": True, "fields": [], "quality": {}}, None

    monkeypatch.setattr(runtime, "recognise", recognise_while_deleted)
    asyncio.run(worker._process_document(rec.id))
    assert repo.get_by_id(store, rec.id) is None
