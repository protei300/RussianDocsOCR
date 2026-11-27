"""Unified repository & storage backend contract (FileStore + SqlStore).

Every shared contract test runs against BOTH ``FileStore`` (JSON files) and
``SqlStore`` (SQLite standing in for MS SQL Server / PostgreSQL) via one
parameterised ``store`` fixture — eliminating duplicate test suites while
keeping backend-specific durability, artifact, and SQL-migration tests intact.
"""
from __future__ import annotations

import pathlib
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from service.core import database  # noqa: E402
from service.core.database import FileStore  # noqa: E402
from service.core.db_sql import SqlStore  # noqa: E402
from service.core.db_sql_migrate import current_revision, head_revision, upgrade_to_head  # noqa: E402
from service.core.models import ApiKey, utcnow  # noqa: E402
from service.core.settings_schema import SettingValidationError  # noqa: E402
from service.repositories import api_keys as key_repo  # noqa: E402
from service.repositories import artifacts  # noqa: E402
from service.repositories import documents as repo  # noqa: E402
from service.repositories import settings as settings_repo  # noqa: E402


@pytest.fixture(params=["files", "sql"])
def store(request, tmp_path):
    """A store of each kind, both empty and migrated."""
    if request.param == "files":
        yield FileStore(tmp_path / "filedata")
        return

    data_dir = tmp_path / "sqldata"
    data_dir.mkdir(parents=True, exist_ok=True)
    backend = SqlStore(f"sqlite:///{tmp_path / 'test.db'}", data_dir)
    upgrade_to_head(backend.engine)
    try:
        yield backend
    finally:
        backend.dispose()


@pytest.fixture
def file_db(tmp_path):
    return database.init_store(tmp_path / "data", wipe=True)


def _make(store, filename="passport.jpg", **kw):
    return repo.create(store, filename=filename, content_type="image/jpeg",
                       size_bytes=kw.pop("size_bytes", 1024), **kw)


# ---------------------------------------------------------------------------
# Identity & CRUD (both backends)
# ---------------------------------------------------------------------------

def test_backend_reports_its_own_persistence(store):
    assert store.is_ephemeral is (store.backend == "files")


def test_create_and_read_back(store):
    created = _make(store, "a.jpg")
    fetched = repo.get_by_id(store, created.id)
    assert fetched is not None
    assert fetched.filename == "a.jpg"
    assert fetched.status == "queued"


def test_ids_increase(store):
    first, second = _make(store, "a.jpg"), _make(store, "b.jpg")
    assert second.id > first.id


def test_missing_id_is_none(store):
    assert repo.get_by_id(store, 12345) is None


def test_update_persists(store):
    record = repo.update(store, _make(store), status="processing", device="gpu")
    reloaded = repo.get_by_id(store, record.id)
    assert reloaded.status == "processing"
    assert reloaded.device == "gpu"


def test_update_status_rejects_unknown_status(store):
    with pytest.raises(ValueError):
        repo.update_status(store, _make(store), "bogus")


def test_update_status_stamps_timestamps(store):
    record = repo.update_status(store, _make(store), "processing")
    assert record.started_at is not None and record.finished_at is None
    record = repo.update_status(store, record, "done")
    assert record.finished_at is not None


def test_delete_removes_record_and_artifacts(store):
    record = _make(store)
    artifacts.save_original(store, record.id, b"\xff\xd8\xffdata", ".jpg")
    directory = store.doc_dir(record.id)
    assert directory.exists()
    repo.delete(store, record)
    assert repo.get_by_id(store, record.id) is None
    assert not directory.exists()


# ---------------------------------------------------------------------------
# Query surface (both backends)
# ---------------------------------------------------------------------------

def test_filter_by_status(store):
    _make(store, "queued.jpg")
    repo.update_status(store, _make(store, "done.jpg"), "done")
    rows, total = repo.get_all(store, status="done")
    assert total == 1 and rows[0].filename == "done.jpg"


def test_search_uses_precomputed_haystack(store):
    record = _make(store, "scan.jpg")
    repo.update(store, record, search_text="scan.jpg intpassport_2011 батурина")
    assert repo.get_all(store, search="БАТУРИНА".lower())[1] == 1
    assert repo.get_all(store, search="ivanov")[1] == 0


def test_doc_type_prefix_and_unrecognised_facets(store):
    repo.update(store, _make(store, "a.jpg"), doc_type="INTPASSPORT_2011", recognised=True)
    repo.update(store, _make(store, "b.jpg"), doc_type="NONE", recognised=False)
    assert repo.get_all(store, doc_type="INTPASSPORT")[1] == 1
    assert repo.get_all(store, doc_type="__none__")[1] == 1


def test_pagination_total_is_unpaged(store):
    for i in range(5):
        _make(store, f"{i}.jpg")
    rows, total = repo.get_all(store, page=1, page_size=2)
    assert len(rows) == 2 and total == 5
    assert len(repo.get_all(store, page=3, page_size=2)[0]) == 1


def test_sort_by_filename_both_directions(store):
    _make(store, "b.jpg")
    _make(store, "a.jpg")
    assert [r.filename for r in repo.get_all(store, sort_by="filename", sort_dir="asc")[0]] == ["a.jpg", "b.jpg"]
    assert [r.filename for r in repo.get_all(store, sort_by="filename", sort_dir="desc")[0]] == ["b.jpg", "a.jpg"]


def test_unknown_sort_column_is_ignored_not_fatal(store):
    _make(store, "a.jpg")
    assert repo.get_all(store, sort_by="1; DROP TABLE rd_documents")[1] == 1
    assert repo.get_all(store, sort_by="__import__", sort_dir="asc")[1] == 1


def test_nulls_sort_last_in_both_directions(store):
    repo.update(store, _make(store, "low.jpg"), doc_conf=0.3)
    repo.update(store, _make(store, "high.jpg"), doc_conf=0.9)
    _make(store, "unscored.jpg")
    ascending = [r.filename for r in repo.get_all(store, sort_by="doc_conf", sort_dir="asc")[0]]
    descending = [r.filename for r in repo.get_all(store, sort_by="doc_conf", sort_dir="desc")[0]]
    assert ascending == ["low.jpg", "high.jpg", "unscored.jpg"]
    assert descending == ["high.jpg", "low.jpg", "unscored.jpg"]


def test_malformed_date_filter_is_ignored(store):
    _make(store)
    assert repo.get_all(store, date_from="not-a-date")[1] == 1


def test_date_range_covers_the_whole_day(store):
    record = _make(store)
    day = record.created_at.strftime("%Y-%m-%d")
    assert repo.get_all(store, date_from=day, date_to=day)[1] == 1


# ---------------------------------------------------------------------------
# Queue & results & aggregates (both backends)
# ---------------------------------------------------------------------------

def test_next_queued_is_oldest_first(store):
    first = _make(store, "first.jpg")
    second = _make(store, "second.jpg")
    repo.update(store, second, created_at=utcnow())
    assert repo.next_queued(store) == first.id


def test_next_queued_ignores_finished(store):
    repo.update_status(store, _make(store), "done")
    assert repo.next_queued(store) is None


def test_queue_position_is_zero_based_and_none_once_claimed(store):
    first, second = _make(store, "1.jpg"), _make(store, "2.jpg")
    assert repo.queue_position(store, first.id) == 0
    assert repo.queue_position(store, second.id) == 1
    claimed = repo.update_status(store, first, "processing")
    assert repo.queue_position(store, claimed.id) is None


def test_reset_stale_processing_and_requeue(store):
    processing = repo.update_status(store, _make(store, "a.jpg"), "processing")
    assert repo.reset_stale_processing(store) == 1
    assert repo.next_queued(store) == processing.id

    failed = repo.update_status(store, _make(store, "b.jpg"), "failed", error="boom", error_code="X")
    requeued = repo.requeue(store, failed)
    assert requeued.status == "queued" and requeued.error is None and requeued.retry_count == 0


def test_result_payload_round_trips_with_cyrillic(store):
    record = _make(store)
    payload = {
        "doc_type": "INTPASSPORT_2011", "recognised": True,
        "quality": {"DocConf": 0.98, "Glare": "good"},
        "canvas": {"width": 505, "height": 701},
        "fields": [{"name": "Last_name_ru", "value": "БАТУРИНА"}],
        "boxes": [{"id": "b0", "label": "Last_name_ru"}],
    }
    record = repo.save_result(store, record, payload, search_text="батурина", processing_ms=475)
    reloaded = repo.get_by_id(store, record.id)
    assert reloaded.result["fields"][0]["value"] == "БАТУРИНА"
    assert reloaded.doc_type == "INTPASSPORT_2011"
    assert reloaded.doc_conf == 0.98
    assert reloaded.quality == {"Glare": "good"}
    assert reloaded.field_count == 1
    assert reloaded.canvas_w == 505 and reloaded.has_canvas is True
    assert reloaded.processing_ms == 475


def test_plain_update_does_not_erase_a_stored_result(store):
    record = _make(store)
    record = repo.save_result(store, record, {"doc_type": "SNILS_1996", "fields": []},
                              search_text="x", processing_ms=1)
    repo.update(store, repo.get_by_id(store, record.id), retry_count=1)
    assert repo.get_by_id(store, record.id).result is not None


def test_stats_counts_and_average(store):
    a = repo.save_result(store, _make(store, "a.jpg"),
                         {"recognised": True, "quality": {}, "fields": []},
                         search_text="", processing_ms=100)
    repo.save_result(store, _make(store, "b.jpg"),
                     {"recognised": True, "quality": {}, "fields": []},
                     search_text="", processing_ms=300)
    repo.update_status(store, _make(store, "c.jpg"), "failed", error="boom")
    stats = repo.stats(store)
    assert stats["done"] == 2 and stats["failed"] == 1
    assert stats["total"] == 3 and stats["recognised"] == 2
    assert stats["avg_processing_ms"] == 200
    assert a.id


def test_stats_average_is_none_without_completions(store):
    _make(store)
    assert repo.stats(store)["avg_processing_ms"] is None


# ---------------------------------------------------------------------------
# API keys & settings (both backends)
# ---------------------------------------------------------------------------

def test_api_keys_round_trip(store):
    key = ApiKey(id=store.next_api_key_id(), label="CI", prefix="rdk_abc",
                 key_hash="deadbeef", created_at=utcnow())
    store.put_api_key(key)
    assert [k.label for k in store.all_api_keys()] == ["CI"]
    assert store.drop_api_key(key.id) is True
    assert store.all_api_keys() == []
    assert store.drop_api_key(key.id) is False


def test_key_repo_create_verify_public_view_and_revoke(store):
    record, plaintext = key_repo.create(store, "CI pipeline")
    assert plaintext.startswith("rdk_")
    assert key_repo.verify(store, plaintext).id == record.id
    assert key_repo.verify(store, "rdk_definitely_not_valid") is None
    assert key_repo.verify(store, "") is None
    for entry in key_repo.public_list(store):
        assert "key_hash" not in entry and "•" in entry["masked"]
    assert key_repo.delete(store, record.id) is True
    assert key_repo.verify(store, plaintext) is None


def test_settings_round_trip_and_validation(store):
    assert settings_repo.get_all(store)["ocr_mode"] == "accurate"
    settings_repo.bulk_update(store, {"img_size": "1200"})
    assert settings_repo.get_value(store, "img_size") == 1200
    settings_repo.bulk_update(store, {"img_size": "900"})
    assert settings_repo.get_value(store, "img_size") == 900

    with pytest.raises(SettingValidationError):
        settings_repo.bulk_update(store, {"docconf": "5"})
    with pytest.raises(SettingValidationError):
        settings_repo.bulk_update(store, {"img_size": "banana"})
    with pytest.raises(SettingValidationError):
        settings_repo.bulk_update(store, {"ocr_mode": "legacy"})

    values, _ = settings_repo.bulk_update(store, {"not_a_setting": "1"})
    assert "not_a_setting" not in values

    _, restart = settings_repo.bulk_update(store, {"ocr_mode": "fast"})
    assert restart == ["ocr_mode"]
    _, restart = settings_repo.bulk_update(store, {"ocr_mode": "fast"})
    assert restart == []
    _, restart = settings_repo.bulk_update(store, {"docconf": "0.7"})
    assert restart == []


# ---------------------------------------------------------------------------
# Artifacts (RGB/BGR, thumbnail fallback, magic-byte sniffing)
# ---------------------------------------------------------------------------

def test_canvas_is_written_as_bgr_so_colours_survive(file_db):
    record = _make(file_db)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    artifacts.save_canvas(file_db, record.id, rgb)

    path, media = artifacts.open_artifact(file_db, record.id, "canvas")
    assert media == "image/png"
    b, g, r = cv2.imread(str(path))[0, 0]
    assert (int(b), int(g), int(r)) == (0, 0, 255), "red/blue channel swap"


def test_thumbnail_falls_back_to_canvas_when_absent(file_db):
    record = _make(file_db)
    artifacts.save_canvas(file_db, record.id, np.full((20, 40, 3), 128, dtype=np.uint8))
    _, media = artifacts.open_artifact(file_db, record.id, "thumb")
    assert media == "image/png"


@pytest.mark.parametrize("data,expected", [
    (b"\xff\xd8\xff\xe0junk", ".jpg"),
    (b"\x89PNG\r\n\x1a\njunk", ".png"),
    (b"RIFF????WEBPjunk", ".webp"),
    (b"BMjunk", ".bmp"),
])
def test_image_types_are_sniffed_from_magic_bytes(data, expected):
    assert artifacts.sniff_image(data)[0] == expected


def test_unsupported_and_pdf_are_distinguishable():
    assert artifacts.sniff_image(b"%PDF-1.7 junk") is None
    assert artifacts.is_pdf(b"%PDF-1.7 junk") is True
    assert artifacts.sniff_image(b"totally not an image") is None
    assert artifacts.decode_dimensions(b"not an image at all") is None


# ---------------------------------------------------------------------------
# FileStore durability & bootstrap key storage
# ---------------------------------------------------------------------------

def test_filestore_records_survive_restart_and_wipe_clears(file_db, tmp_path):
    record = repo.update(file_db, _make(file_db, "kept.jpg"), doc_type="SNILS_1996")
    reopened = database.init_store(tmp_path / "data", wipe=False)
    recovered = repo.get_by_id(reopened, record.id)
    assert recovered is not None and recovered.doc_type == "SNILS_1996"
    assert _make(reopened, "new.jpg").id > record.id

    wiped = database.init_store(tmp_path / "data", wipe=True)
    assert repo.get_all(wiped)[1] == 0


def test_default_key_exists_is_unpredictable_and_plaintext_never_persisted(file_db, tmp_path):
    from service.core import auth

    keys = key_repo.get_all(file_db)
    assert keys[0].is_default
    raw, generated = auth.resolve_default_key()
    assert raw.startswith("rdk_") and generated is True
    assert key_repo.verify(file_db, raw) is not None
    assert auth.resolve_default_key()[0] == raw and len(raw) > 32 and "change" not in raw.lower()

    _, plaintext = key_repo.create(file_db, "persisted")
    assert plaintext not in file_db.api_keys_path.read_text("utf-8")
    reopened = database.init_store(tmp_path / "data", wipe=False)
    assert key_repo.verify(reopened, plaintext) is not None
    assert any(k.is_default for k in key_repo.get_all(reopened))


# ---------------------------------------------------------------------------
# Migrations & reconnect (SQL only)
# ---------------------------------------------------------------------------

def test_migration_is_idempotent(tmp_path):
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    store = SqlStore(f"sqlite:///{tmp_path / 'm.db'}", data_dir)
    try:
        assert current_revision(store.engine) is None
        before, after = upgrade_to_head(store.engine)
        assert before is None and after == head_revision()
        again_before, again_after = upgrade_to_head(store.engine)
        assert again_before == again_after == head_revision()
    finally:
        store.dispose()


def test_schema_and_ids_survive_reconnect(tmp_path):
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    url = f"sqlite:///{tmp_path / 'p.db'}"

    first = SqlStore(url, data_dir)
    upgrade_to_head(first.engine)
    record = _make(first, "kept.jpg")
    repo.update(first, record, doc_type="DL_2011")
    first.dispose()

    second = SqlStore(url, data_dir)
    try:
        assert current_revision(second.engine) == head_revision()
        recovered = repo.get_by_id(second, record.id)
        assert recovered is not None and recovered.doc_type == "DL_2011"
        assert _make(second, "next.jpg").id > record.id
    finally:
        second.dispose()
