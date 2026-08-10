"""Populate an empty store with pre-computed sample documents.

A blank log is a bad first impression and an unhelpful one: there is nothing to
click, so nothing demonstrates what the service actually does. Seeding means the
box overlay, the field table and the timings are visible the moment the page
loads, across every supported document type.

**The results are pre-computed, not re-derived.** ``service/seed_data/`` holds
one finished recognition per document type — the view model, the rendered canvas
and a thumbnail — generated once by ``service/tools/build_seed_data.py`` and
committed. Seeding is therefore a file copy: no GPU, no model load, no minute of
startup latency, and the same rows every time regardless of the host's hardware.
That last property is what makes these files double as golden fixtures for a
port to another language (§11 of the design plan).

Three rules keep this from being a nuisance:

* Only into an **empty** store. With a database configured the first run seeds
  and later runs find the rows already there, so nothing piles up and a deleted
  document stays deleted.
* Only **anonymised repository samples** (``samples/``). Never a user upload,
  never a local personal file — everything seeded here is visible to anyone who
  can reach the UI.
* **One per document type**, in a fixed order, so the log shows the breadth of
  what the library handles rather than nineteen driving licences.

Re-run the builder after any change to the recognition library — otherwise the
seeded rows quietly describe an older version's behaviour.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from service.core.models import utcnow
from service.repositories import artifacts
from service.repositories import documents as repo

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_SAMPLES = _ROOT / "samples"
SEED_DIR = _ROOT / "service" / "seed_data"

#: Order the types appear in. Internal passport first: it is the richest example
#: (ten fields, split fields, plus photo and signature boxes).
#:
#: THIS TUPLE IS WHAT MAKES A TYPE VISIBLE TO THE TEST SUITE. The seed builder
#: walks it, and conformance derives its case list from the seed manifest
#: (conformance/cases.py), so a supported type missing here has no seed document
#: and no conformance case. That is how birth certificates reached production
#: with two of the four ports silently returning zero fields: every port was
#: "44/44 green" while not implementing the type at all. Add the type here in the
#: same change that adds its sample.
_TYPE_ORDER = (
    "INTPASSPORT_2011",
    "DL_2011",
    "SNILS_1996",
    "EXTPASSPORTBIO_2007",
    "INTPASSPORT_1997",
    "DL_2020",
    "EXTPASSPORT_2003",
    "BIRTHCERT_1998",
)

SEED_PREFIX = "sample"


def _load_manifest() -> list[dict]:
    path = SEED_DIR / "manifest.json"
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        log.exception("[SEED] unreadable manifest at %s", path)
        return []


def seed_if_empty(db, *, limit: int | None = None) -> int:
    """Insert the pre-computed samples when the store is empty.

    Returns how many were added. Never raises: a service that cannot seed its
    demo data must still start and accept real uploads.
    """
    if sum(db.count_by_status().values()) > 0:
        return 0

    entries = _load_manifest()
    if not entries:
        log.warning("[SEED] no pre-computed data in %s — the log starts empty. "
                    "Run: python service/tools/build_seed_data.py", SEED_DIR)
        return 0
    if limit:
        entries = entries[:limit]

    added = 0
    for entry in entries:
        try:
            added += _seed_one(db, entry)
        except Exception:
            # One bad fixture must not stop the service from starting.
            log.exception("[SEED] skipping %s", entry.get("slug"))

    log.info("[SEED] inserted %d pre-computed sample document(s)", added)
    return added


def _seed_one(db, entry: dict) -> int:
    entry_dir = SEED_DIR / entry["slug"]
    payload = json.loads((entry_dir / "result.json").read_text("utf-8"))

    # The original is not duplicated into the fixture set — it is the repository
    # sample the result was computed from.
    sample = _ROOT / entry["sample"]
    if not sample.is_file():
        log.warning("[SEED] %s: sample %s is gone", entry["slug"], entry["sample"])
        return 0
    data = sample.read_bytes()
    dimensions = artifacts.decode_dimensions(data)

    now = utcnow()
    # Same bytes-before-row ordering as an upload. Safe either way here (seeding
    # finishes before the worker starts), but two orderings for one invariant is
    # how the unsafe one survives a refactor.
    doc_id = repo.reserve_id(db)
    artifacts.save_original(db, doc_id, data, entry["original_ext"])
    record = repo.create(
        db,
        doc_id=doc_id,
        filename=entry["filename"],
        content_type=entry["content_type"],
        size_bytes=entry["size_bytes"],
        original_ext=entry["original_ext"],
        original_w=dimensions[0] if dimensions else None,
        original_h=dimensions[1] if dimensions else None,
        search_text=entry["search_text"],
        # Seeded rows arrive finished. Timestamps are "now" rather than the
        # build time so the log's relative dates ("2 minutes ago") stay sane.
        status="queued", created_at=now, started_at=now,
    )
    directory = artifacts.doc_dir(db, record.id)
    for name in ("canvas.jpg", "thumb.jpg"):
        source = entry_dir / name
        if source.is_file():
            shutil.copyfile(source, directory / name)

    # `timings` is the library's own dict, in **seconds** (see transform.py).
    total_sec = (payload.get("timings") or {}).get("total") or 0
    repo.save_result(db, record, payload, search_text=entry["search_text"],
                     processing_ms=round(total_sec * 1000))
    return 1
