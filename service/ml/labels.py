"""Human-facing names, ordering and grouping for the library's raw labels.

Why this lives on the server, not in the frontend: the API should be
self-describing. A client — the bundled SPA, a future Go rewrite, or a
third-party integration — must not have to hardcode knowledge like
"``Sex_ru`` means Sex" or "surname comes before given name on a passport".

The label list is the ``TextFields`` detector's 22 classes, from
``document_processing/models/TextFields/ONNX/model.json``. The per-document
field sets come from the ``OCROptions*`` classes in
``document_processing/pipeline/pipeline.py`` (lines ~80-135). If the library
adds a field, this module is the one place that needs updating — and
``UNKNOWN_FIELDS_ARE_TOLERATED`` below explains what happens until it is.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------
# The UI is English (a product decision) while the values are Cyrillic. The
# ``_ru``/``_en`` suffix is a *script* marker, not part of the field's meaning,
# so it is surfaced separately (see ``field_script``) rather than baked into
# the display string.
FIELD_LABELS: dict[str, str] = {
    "Last_name_ru": "Last name",
    "Last_name_en": "Last name",
    "First_name_ru": "First name",
    "First_name_en": "First name",
    "Middle_name_ru": "Middle name",
    "Middle_name_en": "Middle name",
    "Birth_date": "Date of birth",
    "Birth_place_ru": "Place of birth",
    "Birth_place_en": "Place of birth",
    "Sex_ru": "Sex",
    "Sex_en": "Sex",
    "Licence_number": "Document number",
    "Issue_date": "Date of issue",
    "Expiration_date": "Valid until",
    "Issue_organization_ru": "Issuing authority",
    "Issue_organization_en": "Issuing authority",
    "Issue_organisation_code": "Authority code",
    "Living_region_ru": "Place of residence",
    "Living_region_en": "Place of residence",
    "Driver_class": "Categories",
    "Face": "Photo",
    "Signature": "Signature",
    # Synthetic key produced by the pipeline for the registration page — it is
    # not a TextFields class, it comes from the address-line branch.
    "Address": "Registration address",
    "Address_has_handwritten": "Contains handwriting",
}

# ---------------------------------------------------------------------------
# Detected but never OCR'd
# ---------------------------------------------------------------------------
# The detector finds these and the pipeline never sends them to OCR. They are
# still worth drawing — "did it find the photo?" is a real diagnostic question —
# but they need a different visual treatment because they have no text, and a
# UI that expects every box to carry a value would render them as broken rows.
NON_TEXT_LABELS: frozenset[str] = frozenset({"Face", "Signature"})

# ---------------------------------------------------------------------------
# Monospace rendering
# ---------------------------------------------------------------------------
# Monospace earns its keep on digit runs (column alignment, tabular figures).
# On ALL-CAPS Cyrillic it does the opposite: Ш, Щ, Ж and Ы get squeezed and
# legibility drops. So this is an explicit allowlist of digit/Latin-only fields
# rather than "monospace everything that looks like data".
MONOSPACE_FIELDS: frozenset[str] = frozenset({
    "Licence_number",
    "Issue_date",
    "Expiration_date",
    "Birth_date",
    "Issue_organisation_code",
})

# ---------------------------------------------------------------------------
# Reading order
# ---------------------------------------------------------------------------
# A dict has no meaningful order, and even the library's insertion order is not
# the order a human reads the document in. Boxes could be sorted geometrically,
# but that breaks down for fields with no box and for split fields whose parts
# are far apart. An explicit per-document order is boring and correct.
#
# Derived from the OCROptions* field sets, re-sequenced into document reading
# order. Fields absent from a given document simply don't appear.
_PASSPORT_ORDER = [
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
    "Birth_date", "Birth_place_ru",
    "Licence_number", "Issue_date", "Expiration_date",
    "Issue_organization_ru", "Issue_organisation_code",
    "Living_region_ru",
]

_EXT_PASSPORT_ORDER = [
    "Last_name_ru", "Last_name_en", "First_name_ru", "First_name_en",
    "Middle_name_ru", "Middle_name_en", "Sex_ru", "Sex_en",
    "Birth_date", "Birth_place_ru", "Birth_place_en",
    "Licence_number", "Issue_date", "Expiration_date",
    "Issue_organization_ru", "Issue_organization_en", "Issue_organisation_code",
    "Living_region_ru", "Living_region_en",
]

_DL_ORDER = [
    "Last_name_ru", "Last_name_en", "First_name_ru", "First_name_en",
    "Middle_name_ru", "Middle_name_en",
    "Birth_date", "Birth_place_ru", "Birth_place_en",
    "Licence_number", "Issue_date", "Expiration_date",
    "Issue_organization_ru", "Issue_organization_en", "Issue_organisation_code",
    "Living_region_ru", "Living_region_en",
    "Driver_class",
]

_SNILS_ORDER = [
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru",
    "Birth_date", "Birth_place_ru",
    "Licence_number", "Issue_date",
]

_ADDR_ORDER = ["Address", "Address_has_handwritten"]

# Keyed by the *base* document type — the part before the trailing year, which
# is how the pipeline itself dispatches (``doc_type.rsplit('_', maxsplit=1)``).
FIELD_ORDER: dict[str, list[str]] = {
    "INTPASSPORT": _PASSPORT_ORDER,
    "INTPASSPORTADDR": _ADDR_ORDER,
    "EXTPASSPORT": _EXT_PASSPORT_ORDER,
    "EXTPASSPORTBIO": _EXT_PASSPORT_ORDER,
    "DL": _DL_ORDER,
    "SNILS": _SNILS_ORDER,
}

#: Fields the library returns that this module doesn't know about are appended
#: alphabetically after the known ones, with their raw name as the display name.
#: The API therefore degrades gracefully instead of dropping data when the
#: library gains a field before this file is updated.
UNKNOWN_FIELDS_ARE_TOLERATED = True


def base_doc_type(doc_type: str | None) -> str:
    """Strip the era suffix: ``'INTPASSPORT_2011'`` -> ``'INTPASSPORT'``.

    Mirrors the pipeline's own dispatch (``pipeline.py`` line ~463) but without
    its ``ValueError`` on a label that has no underscore — here an unexpected
    label just means "no known ordering", which is survivable.
    """
    if not doc_type:
        return ""
    return doc_type.rsplit("_", maxsplit=1)[0] if "_" in doc_type else doc_type


def doc_type_era(doc_type: str | None) -> str | None:
    """The era suffix on its own: ``'INTPASSPORT_2011'`` -> ``'2011'``.

    ``'INTPASSPORTADDR_ALL'`` yields ``'ALL'``, which the UI renders as a chip
    like any other era — that is intentional, it is what the model reports.
    """
    if not doc_type or "_" not in doc_type:
        return None
    return doc_type.rsplit("_", maxsplit=1)[1]


def field_display(name: str) -> str:
    """English UI label for a raw field name, falling back to the raw name."""
    return FIELD_LABELS.get(name, name)


def field_script(name: str) -> str:
    """``'ru'``, ``'en'`` or ``'num'`` — what the *value* of this field is.

    Drives two client decisions that would otherwise be hardcoded per-language:
    the ``lang`` attribute (font matching, spell-check, screen readers) and
    whether to render the value monospace.
    """
    if name in MONOSPACE_FIELDS:
        return "num"
    if name.endswith("_ru"):
        return "ru"
    if name.endswith("_en"):
        return "en"
    return "ru"


def order_fields(doc_type: str | None, names: list[str]) -> list[str]:
    """Sort field names into document reading order.

    Known fields first in their canonical order, then anything unrecognised
    alphabetically (see ``UNKNOWN_FIELDS_ARE_TOLERATED``).
    """
    canonical = FIELD_ORDER.get(base_doc_type(doc_type), [])
    rank = {name: i for i, name in enumerate(canonical)}
    known = sorted((n for n in names if n in rank), key=lambda n: rank[n])
    unknown = sorted(n for n in names if n not in rank)
    return known + unknown
