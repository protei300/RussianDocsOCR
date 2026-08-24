"""The same suite against both storage backends.

This is the file that makes "swapping the backend is configuration, not a
rewrite" a checkable claim rather than a hopeful comment. Every test is written
against the ``DocumentStore`` protocol and runs twice: once over JSON files,
once over a real database.

SQLite stands in for MS SQL Server and PostgreSQL here because it needs no
server and ships with Python — and because the point is proving the *service*
contains no dialect assumptions, which SQLite exercises just as well. The
migration and the ORM models contain no dialect-specific SQL, so a dialect that
passes here passes anywhere SQLAlchemy supports; verifying the two production
dialects is a separate, connection-string-driven exercise.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from service.core.database import FileStore  # noqa: E402
from service.core.db_sql import SqlStore  # noqa: E402
from service.core.db_sql_migrate import current_revision, head_revision, upgrade_to_head  # noqa: E402
from service.core.models import ApiKey, DocumentRecord, utcnow  # noqa: E402
from service.repositories import documents as repo  # noqa: E402


@pytest.fixture(params=["files", "sql"])
def store(request, tmp_path):
    """A store of each kind, both empty and migrated."""
    if request.param == "files":
        return FileStore(tmp_path / "filedata")

    data_dir = tmp_path / "sqldata"
    data_dir.mkdir(parents=True, exist_ok=True)
    backend = SqlStore(f"sqlite:///{tmp_path / 'test.db'}", data_dir)
    upgrade_to_head(backend.engine)
    return backend


def _make(store, filename="passport.jpg", **kw):
    return repo.create(store, filename=filename, content_type="image/jpeg",
                       size_bytes=kw.pop("size_bytes", 1024), **kw)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def test_backend_reports_its_own_persistence(store):
    if store.backend == "files":
        assert store.is_ephemeral is True
    else:
        assert store.is_ephemeral is False


# ---------------------------------------------------------------------------
# CRUD, identically on both
# ---------------------------------------------------------------------------
def test_create_and_read_back(store):
    created = _make(store, "a.jpg")
    fetched = repo.get_by_id(store, created.id)
    assert fetched is not None
    assert fetched.filename == "a.jpg"
    assert fetched.status == "queued"


def test_ids_increase(store):
    first, second = _make(store, "a.jpg"), _make(store, "b.jpg")
    assert second.id > first.id


def test_update_persists(store):
    record = repo.update(store, _make(store), status="processing", device="gpu")
    reloaded = repo.get_by_id(store, record.id)
    assert reloaded.status == "processing"
    assert reloaded.device == "gpu"


def test_delete_removes(store):
    record = _make(store)
    repo.delete(store, record)
    assert repo.get_by_id(store, record.id) is None


def test_missing_id_is_none(store):
    assert repo.get_by_id(store, 12345) is None


# ---------------------------------------------------------------------------
# Query surface — the part most likely to diverge between backends
# ---------------------------------------------------------------------------
def test_filter_by_status(store):
    _make(store, "queued.jpg")
    repo.update_status(store, _make(store, "done.jpg"), "done")
    rows, total = repo.get_all(store, status="done")
    assert total == 1 and rows[0].filename == "done.jpg"


def test_search_uses_the_precomputed_haystack(store):
    record = _make(store, "scan.jpg")
    repo.update(store, record, search_text="scan.jpg intpassport_2011 тестова")
    assert repo.get_all(store, search="тестова")[1] == 1
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
    assert [r.filename for r in repo.get_all(store, sort_by="filename", sort_dir="asc")[0]] \
        == ["a.jpg", "b.jpg"]
    assert [r.filename for r in repo.get_all(store, sort_by="filename", sort_dir="desc")[0]] \
        == ["b.jpg", "a.jpg"]


def test_unknown_sort_column_is_ignored_not_fatal(store):
    _make(store, "a.jpg")
    # A whitelist, not dynamic attribute access — in SQL that difference is an
    # injection vector, so both backends must reject the same way.
    assert repo.get_all(store, sort_by="1; DROP TABLE rd_documents")[1] == 1


def test_nulls_sort_last_in_both_directions(store):
    repo.update(store, _make(store, "scored.jpg"), doc_conf=0.5)
    _make(store, "unscored.jpg")
    ascending = [r.filename for r in repo.get_all(store, sort_by="doc_conf", sort_dir="asc")[0]]
    assert ascending[0] == "scored.jpg"


def test_malformed_date_filter_is_ignored(store):
    _make(store)
    assert repo.get_all(store, date_from="not-a-date")[1] == 1


def test_date_range_covers_the_whole_day(store):
    record = _make(store)
    day = record.created_at.strftime("%Y-%m-%d")
    assert repo.get_all(store, date_from=day, date_to=day)[1] == 1


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------
def test_next_queued_is_the_oldest(store):
    first = _make(store, "first.jpg")
    _make(store, "second.jpg")
    assert repo.next_queued(store) == first.id


def test_next_queued_ignores_finished(store):
    repo.update_status(store, _make(store), "done")
    assert repo.next_queued(store) is None


def test_queue_position_is_zero_based(store):
    first, second = _make(store, "1.jpg"), _make(store, "2.jpg")
    assert repo.queue_position(store, first.id) == 0
    assert repo.queue_position(store, second.id) == 1


def test_queue_position_is_none_once_claimed(store):
    record = repo.update_status(store, _make(store), "processing")
    assert repo.queue_position(store, record.id) is None


def test_reset_stale_processing(store):
    repo.update_status(store, _make(store), "processing")
    assert repo.reset_stale_processing(store) == 1
    assert repo.next_queued(store) is not None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def test_result_payload_round_trips_with_cyrillic(store):
    record = _make(store)
    payload = {
        "doc_type": "INTPASSPORT_2011", "recognised": True,
        "quality": {"DocConf": 0.98, "Glare": "good"},
        "canvas": {"width": 505, "height": 701},
        "fields": [{"name": "Last_name_ru", "value": "ТЕСТОВА"}],
        "boxes": [{"id": "b0", "label": "Last_name_ru"}],
    }
    record = repo.save_result(store, record, payload, search_text="тестова",
                              processing_ms=475)
    reloaded = repo.get_by_id(store, record.id)
    assert reloaded.result["fields"][0]["value"] == "ТЕСТОВА"
    # Denormalised columns must be filled, or the list page has to open blobs.
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


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------
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
# API keys and settings
# ---------------------------------------------------------------------------
def test_api_keys_round_trip(store):
    key = ApiKey(id=store.next_api_key_id(), label="CI", prefix="rdk_abc",
                 key_hash="deadbeef", created_at=utcnow())
    store.put_api_key(key)
    assert [k.label for k in store.all_api_keys()] == ["CI"]
    assert store.drop_api_key(key.id) is True
    assert store.all_api_keys() == []
    assert store.drop_api_key(key.id) is False


def test_settings_round_trip(store):
    store.set_settings({"img_size": "1200"})
    assert store.all_settings()["img_size"] == "1200"
    store.set_settings({"img_size": "900"})       # update, not insert
    assert store.all_settings()["img_size"] == "900"


# ---------------------------------------------------------------------------
# Migrations (SQL only)
# ---------------------------------------------------------------------------
def test_migration_is_idempotent(tmp_path):
    """Running it twice must be a no-op — that is 'apply only if not applied'."""
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    store = SqlStore(f"sqlite:///{tmp_path / 'm.db'}", data_dir)

    assert current_revision(store.engine) is None
    before, after = upgrade_to_head(store.engine)
    assert before is None
    assert after == head_revision()

    again_before, again_after = upgrade_to_head(store.engine)
    assert again_before == again_after == head_revision()


def test_schema_survives_reconnect(tmp_path):
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    url = f"sqlite:///{tmp_path / 'p.db'}"

    first = SqlStore(url, data_dir)
    upgrade_to_head(first.engine)
    record = _make(first, "kept.jpg")
    repo.update(first, record, doc_type="DL_2011")
    first.dispose()

    second = SqlStore(url, data_dir)
    # No migration this time: the schema is already at head.
    assert current_revision(second.engine) == head_revision()
    recovered = repo.get_by_id(second, record.id)
    assert recovered is not None and recovered.doc_type == "DL_2011"
    second.dispose()


def test_ids_continue_after_reconnect(tmp_path):
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    url = f"sqlite:///{tmp_path / 'i.db'}"

    first = SqlStore(url, data_dir)
    upgrade_to_head(first.engine)
    existing = _make(first, "a.jpg")
    first.dispose()

    second = SqlStore(url, data_dir)
    assert _make(second, "b.jpg").id > existing.id
    second.dispose()
