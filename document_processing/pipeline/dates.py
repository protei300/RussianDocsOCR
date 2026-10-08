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
#: «месяца» and «числа» belong to the 1998 birth certificate's record date,
#: printed in reverse order around the values: «2010 года июня месяца 15 числа».
_NOISE = {'Г', 'Г.', 'ГОД', 'ГОДА', 'ГОДУ', 'МЕСЯЦ', 'МЕСЯЦА', 'ЧИСЛО', 'ЧИСЛА'}

_TOKEN = re.compile(r'[^\W\d_]+|\d+', re.UNICODE)


def _drop_quote_letters(tokens):
    """Drop a lone letter standing right next to the day number.

    The 1998 birth certificate prints the issue date as «10» ЯНВАРЯ 2013 г.,
    and the field box starts on the opening quote. «» are not in the Cyrillic
    engine's alphabet, so the engine reads the quote as the nearest letter it
    knows: «И 10 ЯНВАРЯ 2013» (issue #23). The letter carries no date
    information, but as an unknown word it made the whole date refuse.

    Only a SINGLE letter and only ADJACENT to a one- or two-digit number (the
    day, on either side - the closing quote sits after it) is dropped. Anything
    else - a longer word, a letter elsewhere - still refuses: this reads a
    known misreading of printed punctuation, it does not guess.
    """
    def is_day(i):
        return 0 <= i < len(tokens) and tokens[i].isdigit() and len(tokens[i]) <= 2

    return [t for i, t in enumerate(tokens)
            if not (len(t) == 1 and not t.isdigit() and t not in _MONTHS
                    and (is_day(i - 1) or is_day(i + 1)))]


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
    * ``'И 10 ЯНВАРЯ 2013'``      -> ``'10.01.2013'`` (the quote « read as a letter)
    * ``'2010 ГОДА ИЮНЯ МЕСЯЦА 15 ЧИСЛА'`` -> ``'15.06.2010'`` (record date, 1998 form)
    * ``'5 МАЯ'``                 -> None (no year: guessing one would invent data)
    * ``'31.02.2020'``            -> None (not a calendar date)
    """
    if not text:
        return None

    tokens = [t.upper() for t in _TOKEN.findall(text)]
    tokens = [t for t in tokens if t not in _NOISE and t != 'Г']
    tokens = _drop_quote_letters(tokens)
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


#: Fields printed as a civil-registry record date: year, month, day in a FIXED
#: order with printed words between them - «2010 года июня месяца 15 числа» on
#: the 1998 birth certificate. The box spans the printed words, the word split
#: often loses the gaps («2015ГОДАИЮНЯМЕСЯЦА16») and the printed words come back
#: misread («ИЕСЯЦА», «ТОДА»), so the general converter refuses most of them.
RECORD_DATE_FIELDS = ('Act_date',)

#: Month names in the genitive - the only case a record date prints.
_GENITIVE = {name: n for name, n in _MONTHS.items() if name.endswith(('Я', 'А'))}


def record_date_to_ddmmyyyy(text: str):
    """Canonical ``dd.mm.yyyy`` of a civil-registry record date, or None.

    Whatever the general converter accepts is taken as is. Otherwise the parts
    are found by their FORM, which is what the fixed layout allows: exactly one
    four-digit year, exactly one one- or two-digit day, and exactly one
    genitive month name found INSIDE the letters (glued or not), whatever the
    printed words around it were read as. Any ambiguity - two days, two months,
    no year - refuses, as everywhere in this module:

    * ``'2015ГОДАИЮНЯИЕСЯЦА16'``  -> ``'16.06.2015'``
    * ``'2010 ГОДА ЦЮЛЯ МЕСЯЦА 17'`` -> None (the month itself is misread)
    * ``'2020 ГОДА ИЮЛЯ МЕСЯЦА'`` -> None (no day)
    """
    canonical = to_ddmmyyyy(text)
    if canonical or not text:
        return canonical
    runs = _TOKEN.findall(text.upper())
    years = [r for r in runs if r.isdigit() and len(r) == 4]
    days = [r for r in runs if r.isdigit() and len(r) <= 2]
    stray = [r for r in runs if r.isdigit() and len(r) not in (1, 2, 4)]
    if len(years) != 1 or len(days) != 1 or stray:
        return None
    letters = ''.join(r for r in runs if not r.isdigit())
    months = {n for name, n in _GENITIVE.items() if name in letters}
    if len(months) != 1:
        return None
    return _as_date(int(days[0]), months.pop(), int(years[0]))


def canonical_date(field: str, text: str):
    """The canonical view of one field: by its printed layout."""
    if field in RECORD_DATE_FIELDS:
        return record_date_to_ddmmyyyy(text)
    return to_ddmmyyyy(text)


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
        canonical = canonical_date(name, value)
        if canonical:
            out[name] = canonical
    return out
