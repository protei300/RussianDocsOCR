"""The case list, derived from the service's seed manifest.

Deliberately NOT a second hand-maintained list. ``service/seed_data/manifest.json``
already names one representative document per type, and it is regenerated whenever
recognition changes (``service/tools/build_seed_data.py``). Deriving from it means
the conformance cases and the service's demo data cannot drift apart — and if
somebody adds a document type, both gain a case at once.

Pure: no numpy, no cv2, no document_processing. The checker imports this.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from conformance.paths import REPO, SEED_MANIFEST


@dataclass(frozen=True)
class Case:
    """One document to be recognised identically by every implementation."""

    slug: str
    #: Repo-relative POSIX path, exactly as the manifest stores it.
    sample: str
    doc_type: str

    @property
    def image(self) -> Path:
        return REPO / self.sample

    def exists(self) -> bool:
        return self.image.is_file()


def _doc_type_from_slug(slug: str) -> str:
    """Seed slugs are ``<DOCTYPE>-<source stem>``; the type is the first part."""
    return slug.split("-", 1)[0]


def load_cases(limit: int | None = None) -> list[Case]:
    """Read the case list. Raises if the manifest is missing — that is a setup
    error worth stopping for, not something to paper over with an empty list."""
    if not SEED_MANIFEST.is_file():
        raise FileNotFoundError(
            f"{SEED_MANIFEST} is missing. Generate it with:\n"
            f"    python service/tools/build_seed_data.py")

    entries = json.loads(SEED_MANIFEST.read_text(encoding="utf-8"))
    cases = [Case(slug=e["slug"], sample=e["sample"],
                  doc_type=_doc_type_from_slug(e["slug"]))
             for e in entries]
    # Stable order so reports are diffable run to run.
    cases.sort(key=lambda c: c.slug)
    return cases[:limit] if limit else cases


def select(names: list[str] | None = None, limit: int | None = None) -> list[Case]:
    """Cases filtered by slug substring, for `--case` on the command line."""
    cases = load_cases(limit=limit)
    if not names:
        return cases
    wanted = []
    for c in cases:
        if any(n.lower() in c.slug.lower() for n in names):
            wanted.append(c)
    return wanted
