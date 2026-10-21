"""A verdict for every recognised field: verified, mismatch or unchecked.

Decision No. 152; format and "mark only, never correct" chosen 2026-10-19.
Check ids are the rows of ``docs/validation-checks.md``.

Three verdicts, one per field of ``results.ocr``:

* ``verified`` - at least one CONFIRMING check agreed and no hard check
  disagreed;
* ``mismatch`` - a hard check disagreed: almost certainly a reading error;
* ``unchecked`` - nothing applicable, or nothing to lean on.

Every check declares what its agreement is worth:

* **confirming** - two independent readings agree, or a check digit holds.
  Any single changed character breaks it, so agreement verifies the field.
* **refuting** - a plausibility rule (birth before issue, a real calendar
  date). Almost any misreading still satisfies it, so agreement proves
  nothing; only disagreement counts.
* **soft** - a usual but not guaranteed rule. It never changes a verdict and
  shows up as a note.

A disagreement between two fields marks BOTH: the check knows that one of
them is wrong, not which one. The reading itself is never changed - the value
the field was compared with travels in the check as ``reference``, and what to
do with it is the integrator's call.

Pure functions of the OCR dict and the document type: no image, no model.
That keeps "is the reading right" separate from "is the check right", and is
what a port reimplements.
"""
import re
from datetime import date

from ..pipeline.dates import canonical_date
from .mrz import numbers_agree, parse_mrz, printed_number, validate_mrz

VERIFIED = 'verified'
MISMATCH = 'mismatch'
UNCHECKED = 'unchecked'

CONFIRMING = 'confirming'
REFUTING = 'refuting'

OK = 'ok'

#: Cyrillic -> zone characters of the internal passport's top MRZ line, FMS
#: order No. 279 (2011-06-30), the "modernised clear" table: letters with no
#: Latin counterpart become digits.
_ZONE_LETTERS = str.maketrans({
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': '2',
    'Ж': 'J', 'З': 'Z', 'И': 'I', 'Й': 'Q', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': '3', 'Ш': '4', 'Щ': 'W', 'Ъ': 'X',
    'Ы': 'Y', 'Ь': '9', 'Э': '6', 'Ю': '7', 'Я': '8',
})

#: Sex as printed -> the zone's letter. The page prints «МУЖ.»/«ЖЕН.» (the
#: ground truth keeps the first letter), the external passport also «M»/«F».
_SEX = {'М': 'M', 'Ж': 'F', 'M': 'M', 'F': 'F'}

#: Years between the issue date and the expiry date. Driving licences: 10
#: (35 of 35 references in samples/). External passports: 10 for the
#: biometric form, 5 for the 2003 one (23 of 23 and 19 of 19) - either is
#: accepted for both, so a form the classifier confused costs a little
#: strength instead of a false alarm on two dates.
_VALIDITY_YEARS = {'DL': (10,), 'EXTPASSPORT': (5, 10)}

#: Minimum age at issue. 14 for the internal passport, 16 for a licence
#: (category M/A1). SNILS is issued to children (4 of 13 references) and the
#: external passport too, so neither has an age rule.
_MIN_AGE = {'INTPASSPORT': 14, 'DL': 16}

#: Soft: the second pair of the internal passport's series is the year the
#: blank was printed, usually within a few years of the issue date. The
#: references differ by -2..+1 in 23 of 24 and by +5 in one - so a note, never
#: a verdict, and a window of three years.
_SERIES_YEAR_WINDOW = 3

_DATE_FIELDS = ('Birth_date', 'Issue_date', 'Expiration_date')

#: The top MRZ line: its fixed start, and the characters its name area can
#: hold. No check digit covers this line, but its form is strict: an external
#: passport writes names in Latin letters only; the internal one also uses the
#: digits of the order No. 279 table (2 3 4 6 7 8 9) - never 0, 1 or 5. A
#: character outside that set is a misreading of the zone (in samples/ the
#: letter O read as the digit 0), and such a line is no reference for the page.
#: The printed form of numbers, whitespace collapsed. Every reference of
#: samples/ has it (internal passport 24 of 24, its code 24 of 24, external
#: passport 42 of 42, SNILS 14 of 14). The comparisons with the zone look at
#: digits only, so without the form a reading with a stray character - a glare
#: read as quotes in the number, a doubled hyphen in the code (Damir set) - was
#: 'verified' while not being what is printed.
_NUMBER_FORMS = {
    'INTPASSPORT': (('Licence_number', 'PASS-1', r'\d{2} \d{2} \d{6}'),
                    ('Issue_organisation_code', 'PASS-4', r'\d{3}-\d{3}')),
    'EXTPASSPORT': (('Licence_number', 'EXT-1', r'\d{2} \d{7}'),),
    'SNILS': (('Licence_number', 'SNILS-1', r'\d{3}-\d{3}-\d{3} \d{2}'),),
}

_LINE1_PREFIX = {'INTPASSPORT': 'PNRUS', 'EXTPASSPORT': 'P<RUS'}
_LATIN = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ<')
_NAME_CHARS = {'INTPASSPORT': _LATIN | set('2346789'), 'EXTPASSPORT': _LATIN}


def family(doc_type) -> str:
    """The document family the checks are defined for, or None.

    Order matters as in ``OCROptionsClass.make_options``: the registration page
    and the licence back side contain the names of the families they are not.
    Everything outside the four families gets ``unchecked`` throughout - their
    rules were not measured.
    """
    label = (doc_type or '').upper()
    if 'INTPASSPORTADDR' in label or 'DLBACK' in label:
        return None
    for name in ('INTPASSPORT', 'EXTPASSPORT', 'SNILS', 'DL'):
        if label.startswith(name):
            return name
    return None


class _Verdicts:
    """Collects check results per field and turns them into verdicts."""

    def __init__(self, ocr: dict):
        self.ocr = ocr
        self.entries = {name: {'checks': [], 'notes': []} for name in ocr}

    def value(self, name):
        """The field's reading, stripped, or None when absent or empty."""
        value = self.ocr.get(name)
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip()

    def add(self, fields, check_id, kind, agrees, reference=None):
        """Record one check over one or several fields (only those in ocr)."""
        if isinstance(fields, str):
            fields = (fields,)
        for name in fields:
            if name not in self.entries:
                continue
            check = {'id': check_id, 'kind': kind, 'result': OK if agrees else MISMATCH}
            ref = reference.get(name) if isinstance(reference, dict) else reference
            if ref is not None:
                check['reference'] = ref
            self.entries[name]['checks'].append(check)

    def note(self, fields, check_id, text):
        for name in fields:
            if name in self.entries:
                self.entries[name]['notes'].append({'id': check_id, 'text': text})

    def result(self) -> dict:
        out = {}
        for name, entry in self.entries.items():
            checks = entry['checks']
            if any(c['result'] == MISMATCH for c in checks):
                status = MISMATCH
            elif any(c['kind'] == CONFIRMING for c in checks):
                status = VERIFIED
            else:
                status = UNCHECKED
            out[name] = {'status': status, 'checks': checks, 'notes': entry['notes']}
        return out


def _as_date(name, text):
    canonical = canonical_date(name, text) if text else None
    if not canonical:
        return None
    day, month, year = map(int, canonical.split('.'))
    return date(year, month, day)


def _add_years(day: date, years: int):
    """The same day ``years`` later; 29.02 maps to both 28.02 and 01.03."""
    try:
        return (day.replace(year=day.year + years),)
    except ValueError:
        return (date(day.year + years, 2, 28), date(day.year + years, 3, 1))


def _age(birth: date, at: date) -> int:
    return at.year - birth.year - ((at.month, at.day) < (birth.month, birth.day))


def _fmt(day: date) -> str:
    return day.strftime('%d.%m.%Y')


def _digits(text):
    return re.sub(r'\D', '', text or '')


def _check_dates(v: _Verdicts, fam: str, today: date):
    """GEN-1, GEN-2, PASS-7 / DL-4, DL-2 / EXT-3."""
    dates = {}
    for name in _DATE_FIELDS:
        text = v.value(name)
        if text is None:
            continue
        parsed = _as_date(name, text)
        # GEN-1: every reference date in samples/ converts (309 of 309), so a
        # reading that does not is not a date.
        v.add(name, 'GEN-1', REFUTING, parsed is not None)
        if parsed is not None:
            dates[name] = parsed

    birth, issue, expiry = (dates.get(n) for n in _DATE_FIELDS)
    for name, day in (('Birth_date', birth), ('Issue_date', issue)):
        if day is not None:
            v.add(name, 'GEN-2', REFUTING, day <= today)
    if birth and issue:
        v.add(('Birth_date', 'Issue_date'), 'GEN-2', REFUTING, birth < issue)
        min_age = _MIN_AGE.get(fam)
        if min_age is not None:
            check_id = 'PASS-7' if fam == 'INTPASSPORT' else 'DL-4'
            v.add(('Birth_date', 'Issue_date'), check_id, REFUTING,
                  _age(birth, issue) >= min_age)
    years = _VALIDITY_YEARS.get(fam)
    if years and issue and expiry:
        expected = [d for n in years for d in _add_years(issue, n)]
        reference = None
        if len(years) == 1 and len(expected) == 1:
            back = _add_years(expiry, -years[0])
            reference = {'Expiration_date': _fmt(expected[0]),
                         'Issue_date': _fmt(back[0]) if len(back) == 1 else None}
        check_id = 'DL-2' if fam == 'DL' else 'EXT-3'
        v.add(('Issue_date', 'Expiration_date'), check_id, CONFIRMING,
              expiry in expected, reference)


def _zone_names(text: str) -> str:
    """A name for comparison with the zone: no spaces, hyphens or fillers."""
    return re.sub(r'[\s\-<]', '', text.upper())


def _cyrillic_to_zone(text: str) -> str:
    """A Cyrillic name as the internal passport's zone writes it.

    Ё and Е are made equal on both sides (Ё is '2' in the zone): whether a
    name was typed with Ё is the clerk's choice, not something the page and
    the zone are guaranteed to share.
    """
    return _zone_names(text).replace('Ё', 'Е').translate(_ZONE_LETTERS)


def _names_agree(zone: str, page: str, truncated: bool) -> bool:
    """Equal, or - when the zone's name area is full - the page starts with it."""
    if not zone or not page:
        return False
    if zone == page:
        return True
    return truncated and page.startswith(zone)


def _check_mrz(v: _Verdicts, fam: str):
    """MRZ-1 .. MRZ-9 (docs/validation-checks.md, section MRZ)."""
    mrz_text = v.value('MRZ')
    if mrz_text is None:
        return
    checked = validate_mrz(mrz_text)
    # The check digits cover line 2 only, so for the zone as a whole they can
    # refute but not confirm; the confirmation is the names of line 1 agreeing
    # with the page (MRZ-8 below).
    v.add('MRZ', 'MRZ-1', REFUTING, checked['ok'])
    digits_ok = checked.get('checks') or {}
    internal = fam == 'INTPASSPORT'
    # The layout is chosen by the classifier's type, not by the zone's own
    # first characters, which may be misread.
    zone = parse_mrz(mrz_text, internal=internal)

    def trusted(*parts):
        return all(digits_ok.get(p) == OK for p in parts)

    # MRZ-2: the number, in the form the page prints it.
    page_number = _digits(v.value('Licence_number'))
    number_parts = ('document_number', 'personal_number') if internal else ('document_number',)
    zone_number = printed_number(zone, internal=internal)
    if page_number and zone_number and trusted(*number_parts):
        v.add('Licence_number', 'MRZ-2', CONFIRMING,
              numbers_agree(zone_number, page_number), zone_number)

    # MRZ-3..MRZ-5: dates, each behind its own check digit.
    date_parts = [('Birth_date', 'birth_date', 'birth_date', 'MRZ-3')]
    if internal:
        date_parts.append(('Issue_date', 'issue_date', 'personal_number', 'MRZ-5'))
    else:
        date_parts.append(('Expiration_date', 'expiry_date', 'expiry_date', 'MRZ-4'))
    for page_name, zone_name, part, check_id in date_parts:
        text, zone_value = v.value(page_name), zone.get(zone_name)
        if text is None or not zone_value or not trusted(part):
            continue
        page_value = _as_date(page_name, text)
        v.add(page_name, check_id, CONFIRMING,
              page_value is not None and _fmt(page_value) == zone_value, zone_value)

    # MRZ-6: the subdivision code (internal passport only).
    code_page, code_zone = _digits(v.value('Issue_organisation_code')), zone.get('subdivision_code')
    if internal and code_page and code_zone and trusted('personal_number'):
        v.add('Issue_organisation_code', 'MRZ-6', CONFIRMING, code_page == code_zone,
              f'{code_zone[:3]}-{code_zone[3:]}')

    # MRZ-7: sex. No check digit covers it, but two independent readings
    # agreeing is a confirmation all the same.
    sex_zone = zone.get('sex')
    if sex_zone:
        for name in ('Sex_ru', 'Sex_en'):
            text = v.value(name)
            if text is not None:
                v.add(name, 'MRZ-7', CONFIRMING, _SEX.get(text[0].upper()) == sex_zone, sex_zone)

    # MRZ-8: names from the top line. No check digit covers that line, so a
    # disagreement says "one of the two readings is wrong", not which - and a
    # line of the wrong length may have lost part of a name: not compared.
    lines = [ln.strip() for ln in mrz_text.strip().splitlines() if ln.strip()]
    line1 = lines[0] if lines else ''
    well_formed = (len(line1) == 44 and line1.startswith(_LINE1_PREFIX[fam])
                   and '<<' in line1[5:] and set(line1[5:]) <= _NAME_CHARS[fam])
    v.add('MRZ', 'MRZ-9', REFUTING, well_formed)
    if not well_formed:
        return
    truncated = not lines[0].endswith('<')
    surname = (zone.get('surname') or '').replace(' ', '')
    given = (zone.get('given_names') or '').split(' ', 1)
    zone_first = given[0]
    zone_rest = given[1].replace(' ', '') if len(given) > 1 else ''
    if internal:
        # Ё is '2' in the zone; made equal to Е on both sides (_cyrillic_to_zone).
        surname, zone_first, zone_rest = (s.replace('2', 'E')
                                          for s in (surname, zone_first, zone_rest))
        convert = _cyrillic_to_zone
        names = ('Last_name_ru', 'First_name_ru', 'Middle_name_ru')
    else:
        convert = _zone_names
        names = ('Last_name_en', 'First_name_en', 'Middle_name_en')
    pairs = zip(names, (surname, zone_first, zone_rest))
    agreed = []
    for name, zone_value in pairs:
        text = v.value(name)
        if text is None or not zone_value:
            continue
        agreed.append(_names_agree(zone_value, convert(text), truncated))
        v.add(name, 'MRZ-8', CONFIRMING, agreed[-1], zone_value)
    # The zone itself is confirmed only when every compared name agrees. A
    # disagreement does not mark it: the zone's own misreadings are caught for
    # certain by MRZ-1 and MRZ-9, and which of the two readings of a name is
    # wrong is unknown - the page field already carries that mismatch.
    if agreed and all(agreed):
        v.add('MRZ', 'MRZ-8', CONFIRMING, True)


def _snils_control(body: str) -> int:
    """SNILS check number over nine digits (weights 9..1, mod 101)."""
    total = sum(int(c) * w for c, w in zip(body, range(9, 0, -1)))
    while total > 101:
        total %= 101
    return 0 if total in (100, 101) else total


def _check_forms(v: _Verdicts, fam: str):
    """PASS-1, PASS-4, EXT-1, SNILS-1: the printed form of numbers (refuting)."""
    for name, check_id, pattern in _NUMBER_FORMS.get(fam, ()):
        text = v.value(name)
        if text is not None:
            v.add(name, check_id, REFUTING, re.fullmatch(pattern, ' '.join(text.split())) is not None)


def _check_snils(v: _Verdicts):
    """SNILS-2 (the check number); its form is SNILS-1 in _check_forms."""
    text = v.value('Licence_number')
    if text is None:
        return
    digits = _digits(text)
    if len(digits) != 11:
        return
    body = digits[:9]
    # Numbers up to 001-001-998 are issued without a valid check number.
    if int(body) <= 1001998:
        return
    expected = _snils_control(body)
    v.add('Licence_number', 'SNILS-2', CONFIRMING, int(digits[9:]) == expected, f'{expected:02d}')


def _note_series_year(v: _Verdicts):
    """PASS-3 (soft): the blank's year in the series against the issue year."""
    digits = _digits(v.value('Licence_number'))
    issue = _as_date('Issue_date', v.value('Issue_date'))
    if len(digits) != 10 or issue is None:
        return
    yy = int(digits[2:4])
    blank_year = 2000 + yy if 2000 + yy <= issue.year + _SERIES_YEAR_WINDOW else 1900 + yy
    if abs(issue.year - blank_year) > _SERIES_YEAR_WINDOW:
        v.note(('Licence_number', 'Issue_date'), 'PASS-3',
               f'series printed in {blank_year}, issued in {issue.year}')


def field_checks(ocr: dict, doc_type, today: date = None) -> dict:
    """Verdict per field of ``ocr``: ``{name: {'status', 'checks', 'notes'}}``.

    Every key of ``ocr`` gets an entry, so "no entry" never has to be told
    apart from "unchecked". ``checks`` lists ``{'id', 'kind', 'result'}`` plus
    ``reference`` where the check has a value to compare with; ``notes`` holds
    soft remarks ``{'id', 'text'}`` that do not affect ``status``.
    """
    ocr = ocr or {}
    v = _Verdicts(ocr)
    fam = family(doc_type)
    if fam is None:
        return v.result()
    today = today or date.today()
    _check_dates(v, fam, today)
    _check_forms(v, fam)
    if fam in ('INTPASSPORT', 'EXTPASSPORT'):
        _check_mrz(v, fam)
    if fam == 'SNILS':
        _check_snils(v)
    if fam == 'INTPASSPORT':
        _note_series_year(v)
    return v.result()
