"""Canonical ``dd.mm.yyyy`` view of a recognised date.

The pipeline returns dates AS PRINTED - «15 ОКТЯБРЯ 2020 Г.» on a 2018 birth
certificate, «10 ДЕКАБРЯ 1999 ГОДА» on a SNILS, «03.АВГУСТ.1989» on a 1997
internal passport - because that is what the ground truth describes and what the
accuracy measurement compares against. A consumer usually wants a machine form
instead, so the canonical view is built ALONGSIDE the reading, never in place of
it (``PipelineResults.ocr`` keeps the reading; ``ocr_normalized`` holds this).

Two rules shape everything here:

* **Never guess.** No year in the text -> no canonical value. A month word that
  does not match -> no canonical value. A date outside the calendar (31.02) ->
  no canonical value. The caller falls back to the reading, which is honest,
  instead of receiving a plausible invention.
* **Never touch the reading.** Trailing «Г.» / «ГОДА» stay in the reading; they
  are printed on the document. They simply have no place in ``dd.mm.yyyy``.

Pure functions of a string: no image, no model, no configuration. That is what
makes them testable on their own, separately from the question of whether the
string was read correctly.
"""
import re
from datetime import date

#: Month names as the documents print them, in the nominative and the genitive:
#: a birth certificate writes «15 ОКТЯБРЯ», a SNILS «10 ДЕКАБРЯ», and the 1997
#: internal passport prints the nominative «03.АВГУСТ.1989».
_MONTHS = {
    'ЯНВАРЬ': 1, 'ЯНВАРЯ': 1,
    'ФЕВРАЛЬ': 2, 'ФЕВРАЛЯ': 2,
    'МАРТ': 3, 'МАРТА': 3,
    'АПРЕЛЬ': 4, 'АПРЕЛЯ': 4,
    'МАЙ': 5, 'МАЯ': 5,
    'ИЮНЬ': 6, 'ИЮНЯ': 6,
    'ИЮЛЬ': 7, 'ИЮЛЯ': 7,
    'АВГУСТ': 8, 'АВГУСТА': 8,
    'СЕНТЯБРЬ': 9, 'СЕНТЯБРЯ': 9,
    'ОКТЯБРЬ': 10, 'ОКТЯБРЯ': 10,
    'НОЯБРЬ': 11, 'НОЯБРЯ': 11,
    'ДЕКАБРЬ': 12, 'ДЕКАБРЯ': 12,
}

#: Words a document prints next to a date that carry no date information.
_NOISE = {'Г', 'Г.', 'ГОД', 'ГОДА', 'ГОДУ'}

_TOKEN = re.compile(r'[^\W\d_]+|\d+', re.UNICODE)


def _as_date(day, month, year):
    """dd.mm.yyyy for a real calendar date, else None (31.02 is not a date)."""
    if not (1 <= month <= 12) or year < 1900 or year > 2100:
        return None
    try:
        date(year, month, day)
    except ValueError:
        return None
    return f'{day:02d}.{month:02d}.{year:04d}'


def to_ddmmyyyy(text: str):
    """Canonical ``dd.mm.yyyy``, or None when the text does not yield one.

    Handles what the documents actually print:

    * ``'22.06.2010'``            -> ``'22.06.2010'`` (already canonical)
    * ``'15 ОКТЯБРЯ 2020 Г.'``    -> ``'15.10.2020'``
    * ``'10 ДЕКАБРЯ 1999 ГОДА'``  -> ``'10.12.1999'``
    * ``'03.АВГУСТ.1989'``        -> ``'03.08.1989'``
    * ``'5 МАЯ'``                 -> None (no year: guessing one would invent data)
    * ``'31.02.2020'``            -> None (not a calendar date)
    """
    if not text:
        return None

    tokens = [t.upper() for t in _TOKEN.findall(text)]
    tokens = [t for t in tokens if t not in _NOISE and t != 'Г']
    if not tokens:
        return None

    day = month = year = None
    for token in tokens:
        if token.isdigit():
            value = int(token)
            if len(token) == 4 and year is None:
                year = value
            elif day is None and 1 <= value <= 31:
                day = value
            elif month is None and 1 <= value <= 12:
                month = value
            elif year is None and len(token) <= 2:
                # a two-digit year is ambiguous (26 -> 1926 or 2026?) and this
                # module does not guess, so it is left unresolved
                return None
        else:
            resolved = _MONTHS.get(token)
            if resolved is None or month is not None:
                return None
            month = resolved

    if day is None or month is None or year is None:
        return None
    return _as_date(day, month, year)


def canonical_dates(ocr: dict, fields) -> dict:
    """Canonical view of every date field that yields one.

    Returns a NEW dict holding only the fields that converted - a field that did
    not convert is simply absent, so the consumer can tell "no canonical form"
    from "canonical form equals the reading". Never mutates ``ocr``: the reading
    is what the accuracy measurement compares against, and a canonical value
    written over it would quietly change what is being measured.
    """
    if not ocr:
        return {}
    out = {}
    for name in fields:
        value = ocr.get(name)
        if not isinstance(value, str):
            continue
        canonical = to_ddmmyyyy(value)
        if canonical:
            out[name] = canonical
    return out
