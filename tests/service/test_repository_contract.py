"""The repository contract — and therefore the SQL-migration specification.

Everything here is written against the *interface*, never against filesystem
details. When a SQLAlchemy backend lands, this module gets parametrised over
``[filestore, sqlite]`` and must pass unchanged for both. If a test in here
needs rewriting to accommodate the new backend, the abstraction has leaked.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from service.core import database  # noqa: E402
from service.core.models import utcnow  # noqa: E402
from service.repositories import api_keys as key_repo  # noqa: E402
from service.repositories import artifacts  # noqa: E402
from service.repositories import documents as repo  # noqa: E402
from service.repositories import settings as settings_repo  # noqa: E402
from service.core.settings_schema import SettingValidationError  # noqa: E402


@pytest.fixture
def db(tmp_path):
    return database.init_store(tmp_path / "data", wipe=True)


def _make(db, filename="passport.jpg", **kw):
    return repo.create(db, filename=filename, content_type="image/jpeg",
                       size_bytes=kw.pop("size_bytes", 1024), **kw)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------
def test_create_assigns_increasing_ids(db):
    a, b = _make(db, "a.jpg"), _make(db, "b.jpg")
    assert b.id > a.id


def test_get_by_id_returns_none_for_missing(db):
    assert repo.get_by_id(db, 999) is None


def test_update_returns_a_new_record_and_persists(db):
    record = _make(db)
    updated = repo.update(db, record, status="processing")
    assert updated.status == "processing"
    assert repo.get_by_id(db, record.id).status == "processing"


def test_update_status_rejects_unknown_status(db):
    with pytest.raises(ValueError):
        repo.update_status(db, _make(db), "bogus")


def test_update_status_stamps_timestamps(db):
    record = repo.update_status(db, _make(db), "processing")
    assert record.started_at is not None and record.finished_at is None
    record = repo.update_status(db, record, "done")
    assert record.finished_at is not None


def test_delete_removes_record_and_artifacts(db):
    record = _make(db)
    artifacts.save_original(db, record.id, b"\xff\xd8\xffdata", ".jpg")
    directory = db.doc_dir(record.id)
    assert directory.exists()
    repo.delete(db, record)
    assert repo.get_by_id(db, record.id) is None
    assert not directory.exists()


# ---------------------------------------------------------------------------
# Query surface — what the SQL version must reproduce
# ---------------------------------------------------------------------------
def test_filter_by_status(db):
    _make(db, "a.jpg")
    repo.update_status(db, _make(db, "b.jpg"), "done")
    rows, total = repo.get_all(db, status="done")
    assert total == 1 and rows[0].filename == "b.jpg"


def test_search_matches_precomputed_haystack(db):
    record = _make(db, "scan.jpg")
    repo.update(db, record, search_text="scan.jpg intpassport_2011 батурина")
    rows, total = repo.get_all(db, search="БАТУРИНА".lower())
    assert total == 1
    assert repo.get_all(db, search="nonexistent")[1] == 0


def test_doc_type_facet_and_not_recognised_facet(db):
    repo.update(db, _make(db, "a.jpg"), doc_type="INTPASSPORT_2011", recognised=True)
    repo.update(db, _make(db, "b.jpg"), doc_type="NONE", recognised=False)
    assert repo.get_all(db, doc_type="INTPASSPORT")[1] == 1
    assert repo.get_all(db, doc_type="__none__")[1] == 1


def test_pagination_reports_unpaged_total(db):
    for i in range(5):
        _make(db, f"{i}.jpg")
    rows, total = repo.get_all(db, page=1, page_size=2)
    assert len(rows) == 2 and total == 5
    assert len(repo.get_all(db, page=3, page_size=2)[0]) == 1


def test_sort_whitelist_ignores_unknown_columns(db):
    _make(db, "a.jpg")
    _make(db, "b.jpg")
    # Must not raise, must not honour the bogus column.
    rows, _ = repo.get_all(db, sort_by="__import__", sort_dir="asc")
    assert len(rows) == 2


def test_sort_by_filename_both_directions(db):
    _make(db, "b.jpg")
    _make(db, "a.jpg")
    asc = [r.filename for r in repo.get_all(db, sort_by="filename", sort_dir="asc")[0]]
    assert asc == ["a.jpg", "b.jpg"]
    desc = [r.filename for r in repo.get_all(db, sort_by="filename", sort_dir="desc")[0]]
    assert desc == ["b.jpg", "a.jpg"]


def test_nulls_sort_last_regardless_of_direction(db):
    """A queued document has no doc_conf; it must not lead an ascending sort."""
    repo.update(db, _make(db, "scored.jpg"), doc_conf=0.5)
    _make(db, "unscored.jpg")
    asc = [r.filename for r in repo.get_all(db, sort_by="doc_conf", sort_dir="asc")[0]]
    assert asc[0] == "scored.jpg"


def test_malformed_date_filter_is_ignored_not_fatal(db):
    _make(db)
    assert repo.get_all(db, date_from="not-a-date")[1] == 1


def test_date_to_is_inclusive_of_that_whole_day(db):
    record = _make(db)
    today = record.created_at.strftime("%Y-%m-%d")
    assert repo.get_all(db, date_from=today, date_to=today)[1] == 1


# ---------------------------------------------------------------------------
# Queue behaviour
# ---------------------------------------------------------------------------
def test_next_queued_is_oldest_first(db):
    first = _make(db, "first.jpg")
    second = _make(db, "second.jpg")
    repo.update(db, second, created_at=utcnow())
    assert repo.next_queued(db) == first.id


def test_next_queued_ignores_non_queued(db):
    repo.update_status(db, _make(db), "done")
    assert repo.next_queued(db) is None


def test_queue_position_is_zero_based(db):
    first, second = _make(db, "1.jpg"), _make(db, "2.jpg")
    assert repo.queue_position(db, first.id) == 0
    assert repo.queue_position(db, second.id) == 1


def test_reset_stale_processing_recovers_interrupted_jobs(db):
    repo.update_status(db, _make(db), "processing")
    assert repo.reset_stale_processing(db) == 1
    assert repo.next_queued(db) is not None


def test_requeue_clears_error_state(db):
    record = repo.update_status(db, _make(db), "failed", error="boom", error_code="X")
    record = repo.requeue(db, record)
    assert record.status == "queued" and record.error is None and record.retry_count == 0


# ---------------------------------------------------------------------------
# Durability
# ---------------------------------------------------------------------------
def test_records_survive_a_store_restart(db, tmp_path):
    """The rescan is what keeps `uvicorn --reload` from losing everything."""
    record = repo.update(db, _make(db, "kept.jpg"), doc_type="SNILS_1996")
    reopened = database.init_store(tmp_path / "data", wipe=False)
    recovered = repo.get_by_id(reopened, record.id)
    assert recovered is not None and recovered.doc_type == "SNILS_1996"


def test_wipe_on_start_actually_empties_the_store(db, tmp_path):
    _make(db)
    wiped = database.init_store(tmp_path / "data", wipe=True)
    assert repo.get_all(wiped)[1] == 0


def test_new_ids_do_not_collide_after_restart(db, tmp_path):
    existing = _make(db)
    reopened = database.init_store(tmp_path / "data", wipe=False)
    assert _make(reopened, "new.jpg").id > existing.id


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
def test_canvas_is_written_as_bgr_so_colours_survive():
    """Regression guard for the RGB/BGR trap.

    `img_with_fixed_perspective` is RGB; `cv2.imwrite` expects BGR. Getting this
    wrong swaps red and blue in every document, and on a passport the result
    still looks plausible — so only an explicit assertion catches it.
    """
    import cv2
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = database.init_store(pathlib.Path(tmp) / "data", wipe=True)
        record = _make(db)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255                       # pure RED in RGB terms
        artifacts.save_canvas(db, record.id, rgb)

        path, media = artifacts.open_artifact(db, record.id, "canvas")
        assert media == "image/png"
        # imread gives BGR; a correctly-saved red pixel is (0, 0, 255) there.
        b, g, r = cv2.imread(str(path))[0, 0]
        assert (int(b), int(g), int(r)) == (0, 0, 255), "red/blue channel swap"


def test_thumbnail_falls_back_to_canvas_when_absent(db):
    record = _make(db)
    rgb = np.full((20, 40, 3), 128, dtype=np.uint8)
    artifacts.save_canvas(db, record.id, rgb)
    path, media = artifacts.open_artifact(db, record.id, "thumb")
    assert media == "image/png"          # fell back to the canvas


@pytest.mark.parametrize("data,expected", [
    (b"\xff\xd8\xff\xe0junk", ".jpg"),
    (b"\x89PNG\r\n\x1a\njunk", ".png"),
    (b"RIFF????WEBPjunk", ".webp"),
    (b"BMjunk", ".bmp"),
])
def test_image_types_are_sniffed_from_magic_bytes(data, expected):
    assert artifacts.sniff_image(data)[0] == expected


def test_unsupported_and_pdf_are_distinguishable():
    """PDFs get their own error message — users will try them."""
    assert artifacts.sniff_image(b"%PDF-1.7 junk") is None
    assert artifacts.is_pdf(b"%PDF-1.7 junk") is True
    assert artifacts.sniff_image(b"totally not an image") is None


def test_undecodable_bytes_report_no_dimensions():
    assert artifacts.decode_dimensions(b"not an image at all") is None


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
def test_default_key_always_exists_and_verifies(db):
    """The bootstrap key exists whether or not one was configured.

    With DEFAULT_API_KEY unset it is generated at startup, so the resolver — not
    the raw setting — is the source of truth.
    """
    from service.core.auth import resolve_default_key
    keys = key_repo.get_all(db)
    assert keys[0].is_default
    raw, generated = resolve_default_key()
    assert raw.startswith("rdk_")
    assert key_repo.verify(db, raw) is not None
    # No env var is set in the test environment, so it must have been generated
    # rather than falling back to a hardcoded constant.
    assert generated is True


def test_generated_default_key_is_not_predictable(db):
    """Two resolutions in one process agree; the value is not a fixed literal."""
    from service.core import auth
    first, _ = auth.resolve_default_key()
    second, _ = auth.resolve_default_key()
    assert first == second                    # stable within the process
    assert len(first) > 32                    # not a short placeholder
    assert "change" not in first.lower()


def test_created_key_verifies_and_plaintext_is_returned_once(db):
    record, plaintext = key_repo.create(db, "CI pipeline")
    assert plaintext.startswith("rdk_")
    assert key_repo.verify(db, plaintext).id == record.id
    # Only the hash is persisted — the plaintext appears nowhere in storage.
    assert plaintext not in db.api_keys_path.read_text("utf-8")


def test_public_view_never_exposes_the_hash(db):
    key_repo.create(db, "x")
    for entry in key_repo.public_list(db):
        assert "key_hash" not in entry and "•" in entry["masked"]


def test_wrong_key_is_rejected(db):
    assert key_repo.verify(db, "rdk_definitely_not_valid") is None
    assert key_repo.verify(db, "") is None


def test_deleting_a_created_key_revokes_it(db):
    record, plaintext = key_repo.create(db, "temp")
    assert key_repo.delete(db, record.id) is True
    assert key_repo.verify(db, plaintext) is None


def test_keys_survive_restart_but_default_is_synthesised(db, tmp_path):
    _, plaintext = key_repo.create(db, "persisted")
    reopened = database.init_store(tmp_path / "data", wipe=False)
    assert key_repo.verify(reopened, plaintext) is not None
    assert any(k.is_default for k in key_repo.get_all(reopened))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def test_defaults_are_returned_when_nothing_is_stored(db):
    assert settings_repo.get_all(db)["ocr_mode"] == "accurate"


def test_update_round_trips_and_types_correctly(db):
    settings_repo.bulk_update(db, {"img_size": "1200"})
    assert settings_repo.get_value(db, "img_size") == 1200      # int, not "1200"


def test_out_of_range_value_is_rejected(db):
    with pytest.raises(SettingValidationError):
        settings_repo.bulk_update(db, {"docconf": "5"})          # max is 1.0


def test_non_numeric_value_is_rejected(db):
    """The reference project happily stores `poll_interval=banana`."""
    with pytest.raises(SettingValidationError):
        settings_repo.bulk_update(db, {"img_size": "banana"})


def test_invalid_choice_is_rejected(db):
    with pytest.raises(SettingValidationError):
        settings_repo.bulk_update(db, {"ocr_mode": "legacy"})    # removed in 3.0.0


def test_unknown_keys_are_dropped_silently(db):
    values, _ = settings_repo.bulk_update(db, {"not_a_setting": "1"})
    assert "not_a_setting" not in values


def test_restart_required_is_reported_only_on_change(db):
    _, restart = settings_repo.bulk_update(db, {"ocr_mode": "fast"})
    assert restart == ["ocr_mode"]
    _, restart = settings_repo.bulk_update(db, {"ocr_mode": "fast"})
    assert restart == []                                          # unchanged
    _, restart = settings_repo.bulk_update(db, {"docconf": "0.7"})
    assert restart == []                                          # applies live
