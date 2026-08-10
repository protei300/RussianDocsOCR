"""OCR alphabet-masking config.

Loads ``ocr_alphabets.json`` (vendored next to this module) and resolves the set
of characters a decode step is allowed to emit for a given script/country. The
model's *full* alphabet lives in each ``model.json``; this module only says which
of those characters are permitted for a particular document (RU passports use
RUS Cyrillic + digits/punctuation; the English lines use USA Latin A-Z + digits).

Only RU documents are supported today, so the country resolves to the per-script
default (cyrillic->RUS, latin->USA). The table is data-driven so new countries
can be added to ``ocr_alphabets.json`` without touching code.
"""
import json
from functools import lru_cache
from pathlib import Path

_CFG_PATH = Path(__file__).resolve().parent / "ocr_alphabets.json"


@lru_cache(maxsize=1)
def _config() -> dict:
    return json.loads(_CFG_PATH.read_text(encoding="utf-8"))


def default_country(script: str) -> str:
    """Default ISO-3 country for a script (cyrillic->RUS, latin->USA)."""
    return _config()["default_country"][script]


@lru_cache(maxsize=None)
def allowed_charset(script: str, country: str | None = None) -> frozenset:
    """Characters a decode step may emit for this script/country.

    Returns letters for the country plus the always-allowed SPECIALS (digits and
    punctuation). ``country=None`` uses the per-script default.

    Raises KeyError for an unknown script/country so misconfiguration fails loud
    rather than silently masking every character away.
    """
    cfg = _config()
    if country is None:
        country = default_country(script)
    letters = cfg["letters_per_country"][script][country]
    return frozenset(letters) | frozenset(cfg["specials"])
