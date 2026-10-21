"""ICAO 9303 machine-readable-zone checks (EXT-4) and MRZ/visual cross-checks (EXT-5).

The MRZ is the only field on these documents whose recognition can be verified
by arithmetic: its second line carries check digits over the document number,
the dates and the personal number, plus a composite digit over all of them. A
failing digit means an OCR error with near-certainty, which makes this the
strongest quality signal available for passports.

Two document families use a TD3-shaped (2x44) zone here and they are NOT the
same:

* external passport - a full ICAO TD3. Type ``P<``, expiry date present.
* internal passport (2011+) - type ``PN``, and **no expiry date**: positions
  21-27 are filler with ``<`` where the check digit would be. Treating that as
  a failed check digit would reject every correctly read internal passport, so
  it is reported as ``not_applicable``.

Scope is recognition quality, not document authenticity - see
``docs/validation-checks.md``. Names in the internal-passport MRZ are also not
plain transliteration (Cyrillic letters with no Latin counterpart come out as
digits: Ч->3, Ш->4), so the name cross-check is limited to documents that
actually print a Latin name in the visual zone.
"""
import re
from datetime import date

FILLER = '<'
LINE_LEN = 44

#: TD3 line-2 layout: (name, payload slice, check-digit position).
_LINE2_FIELDS = (
    ('document_number', slice(0, 9), 9),
    ('birth_date', slice(13, 19), 19),
    ('expiry_date', slice(21, 27), 27),
    ('personal_number', slice(28, 42), 42),
)
#: The composite digit covers the number, the birth date and everything from
#: the expiry date to the personal-number check digit, each WITH its own digit.
_COMPOSITE_PARTS = (slice(0, 10), slice(13, 20), slice(21, 43))
_COMPOSITE_POS = 43

_MRZ_CHARS = re.compile(r'^[A-Z0-9<]+$')


def check_digit(payload: str):
    """ICAO 9303 check digit: weights cycle 7-3-1, ``A``-``Z`` are 10-35, ``<`` is 0.

    Returns None if the payload holds a character outside the MRZ alphabet -
    an OCR artifact, for which no meaningful digit exists.
    """
    weights = (7, 3, 1)
    total = 0
    for i, ch in enumerate(payload):
        if ch == FILLER:
            value = 0
        elif ch.isdigit():
            value = int(ch)
        elif 'A' <= ch <= 'Z':
            value = ord(ch) - 55
        else:
            return None
        total += value * weights[i % 3]
    return total % 10


def split_lines(mrz_text: str):
    """MRZ string -> list of lines. The pipeline joins detected lines with '\\n'."""
    if not mrz_text:
        return []
    return [ln.strip() for ln in mrz_text.splitlines() if ln.strip()]


def _is_empty_field(payload: str, digit: str) -> bool:
    """A field the document does not carry: all filler, filler in place of the digit.

    This is what an internal passport does with the expiry date.
    """
    return payload == FILLER * len(payload) and digit == FILLER


def validate_mrz(mrz_text: str) -> dict:
    """EXT-4: verify every check digit of the MRZ.

    Returns a dict with a per-field verdict (``ok`` / ``failed`` /
    ``not_applicable``), the overall ``ok`` flag and, when the zone is
    well-formed, the parsed field values. ``ok`` is False for a malformed zone:
    a 43-character line means characters were lost, which is itself a finding.
    """
    lines = split_lines(mrz_text)
    result = {'ok': False, 'checks': {}, 'lines': len(lines)}

    if len(lines) < 2:
        result['error'] = 'expected 2 lines'
        return result
    line2 = lines[1]
    if len(line2) != LINE_LEN or not _MRZ_CHARS.match(line2):
        result['error'] = f'line 2 malformed (len={len(line2)})'
        return result

    checks = result['checks']
    for name, payload_slice, digit_pos in _LINE2_FIELDS:
        payload, digit = line2[payload_slice], line2[digit_pos]
        if _is_empty_field(payload, digit):
            checks[name] = 'not_applicable'
            continue
        expected = check_digit(payload)
        checks[name] = 'ok' if expected is not None and str(expected) == digit else 'failed'

    composite = ''.join(line2[part] for part in _COMPOSITE_PARTS)
    expected = check_digit(composite)
    checks['composite'] = ('ok' if expected is not None
                           and str(expected) == line2[_COMPOSITE_POS] else 'failed')

    result['ok'] = all(v != 'failed' for v in checks.values())
    result['fields'] = parse_mrz(mrz_text)
    return result


def _mrz_date(raw: str, kind: str = 'past'):
    """YYMMDD -> ``dd.mm.yyyy``, or None if it is not a real calendar date.

    The MRZ carries no century, so it has to come from what the field means -
    and the two directions are opposite. A birth date is in the past: a
    two-digit year later than the current one belongs to the previous century.
    An expiry date is in the future and is always 20xx - reading it by the
    birth-date rule turns a passport valid until 2029 into one that expired in
    1929.
    """
    if len(raw) != 6 or not raw.isdigit():
        return None
    yy, mm, dd = int(raw[0:2]), int(raw[2:4]), int(raw[4:6])
    if kind == 'future':
        century = 2000
    else:
        century = 2000 if yy <= date.today().year % 100 else 1900
    try:
        return date(century + yy, mm, dd).strftime('%d.%m.%Y')
    except ValueError:
        return None


def parse_mrz(mrz_text: str, internal: bool = None) -> dict:
    """Field values carried by the zone. Missing/malformed parts come back None.

    ``internal`` says whether this is an internal passport's zone, whose
    optional data holds three printed fields (``_internal_optional_data``).
    None takes it from the zone's own type characters - a caller that knows the
    document type from the classifier should pass it, because those two
    characters can be misread like any other.
    """
    lines = split_lines(mrz_text)
    fields = {}
    if lines:
        line1 = lines[0]
        fields['document_type'] = line1[0:2].replace(FILLER, '') or None
        fields['issuing_state'] = line1[2:5] or None
        names = line1[5:].split(FILLER * 2, 1)
        fields['surname'] = names[0].replace(FILLER, ' ').strip() or None
        given = names[1] if len(names) > 1 else ''
        fields['given_names'] = given.replace(FILLER, ' ').strip() or None
    if len(lines) > 1 and len(lines[1]) == LINE_LEN:
        line2 = lines[1]
        fields['document_number'] = line2[0:9].replace(FILLER, '') or None
        fields['nationality'] = line2[10:13] or None
        fields['birth_date'] = _mrz_date(line2[13:19], 'past')
        fields['sex'] = line2[20] if line2[20] in ('M', 'F') else None
        fields['expiry_date'] = _mrz_date(line2[21:27], 'future')
        fields['personal_number'] = line2[28:42].replace(FILLER, '') or None
        if internal is None:
            internal = fields.get('document_type') == INTERNAL_TYPE
        if internal:
            fields.update(_internal_optional_data(line2[28:42]))
    return fields


#: Document type of the internal passport's zone ("Passport National").
INTERNAL_TYPE = 'PN'


def _internal_optional_data(optional: str) -> dict:
    """The internal passport's optional data: three printed fields in one run.

    FMS order No. 279 (2011-06-30), table 2, positions 29-42: the LAST digit of
    the series, the issue date as YYMMDD, the issuing subdivision's code as six
    digits without the hyphen, then a filler. The zone's own number (positions
    1-9) holds only the first three digits of the series, so the fourth one has
    to be taken from here to rebuild the printed series and number.
    """
    digit = optional[0:1]
    code = optional[7:13]
    return {
        'series_last_digit': digit if digit.isdigit() else None,
        'issue_date': _mrz_date(optional[1:7], 'past'),
        'subdivision_code': code if len(code) == 6 and code.isdigit() else None,
    }


def printed_number(fields: dict, internal: bool = None):
    """The zone's document number in the form the page prints it, digits only.

    External passport: the number field as is. Internal passport: the first
    three digits of the series + the fourth one from the optional data + the
    six digits of the number - the zone splits the ten printed digits in two
    places, and comparing its nine-digit field with the page's ten digits
    reported every correctly read internal passport as a mismatch.
    """
    number = fields.get('document_number')
    if not number:
        return None
    if internal is None:
        internal = fields.get('document_type') == INTERNAL_TYPE
    if not internal:
        return number
    last = fields.get('series_last_digit')
    if not last or len(number) != 9:
        return None
    return number[:3] + last + number[3:]


def numbers_agree(zone_number: str, page_digits: str) -> bool:
    """Zone number against the page's digits.

    Five of the 19 external passports of the 2003 form in samples/ write the
    number WITHOUT its two-digit series (seven digits); the rest, and every
    biometric one, write all nine. A seven-digit zone number is therefore
    compared with the end of the printed number.
    """
    if not zone_number or not page_digits:
        return False
    if len(zone_number) == 7 and len(page_digits) == 9:
        return page_digits.endswith(zone_number)
    return zone_number == page_digits


def _digits(text):
    return re.sub(r'\D', '', text or '')


def cross_check(mrz_text: str, ocr: dict) -> dict:
    """EXT-5: MRZ against the visually printed fields of the same document.

    Only fields present on both sides are compared; anything missing is
    skipped rather than reported as a mismatch, because a field the detector
    did not find says nothing about agreement. Names are compared only when the
    document prints a Latin name of its own - the internal passport substitutes
    digits for Cyrillic-only letters in the MRZ, so its zone can never match a
    transliteration character for character (the per-field verdicts in
    ``fields.py`` compare those through the order's own table instead).

    The internal passport's number is rebuilt from two places of the zone
    (``printed_number``), and its issue date and subdivision code - which only
    that zone carries - are compared too.
    """
    fields = parse_mrz(mrz_text)
    ocr = ocr or {}
    checks = {}

    def compare(name, mrz_value, ocr_value, normalize=lambda v: v):
        if not mrz_value or not ocr_value:
            return
        checks[name] = 'ok' if normalize(mrz_value) == normalize(ocr_value) else 'mismatch'

    # The visual zone prints the number with a space ("76 4820154"); the MRZ
    # does not. Compare digits only.
    zone_number = printed_number(fields)
    page_number = _digits(ocr.get('Licence_number'))
    if zone_number and page_number:
        checks['document_number'] = ('ok' if numbers_agree(zone_number, page_number)
                                     else 'mismatch')
    compare('birth_date', fields.get('birth_date'), ocr.get('Birth_date'), _digits)
    compare('issue_date', fields.get('issue_date'), ocr.get('Issue_date'), _digits)
    compare('subdivision_code', fields.get('subdivision_code'),
            ocr.get('Issue_organisation_code'), _digits)
    compare('expiry_date', fields.get('expiry_date'), ocr.get('Expiration_date'), _digits)
    compare('sex', fields.get('sex'), ocr.get('Sex_en'), lambda v: v.strip().upper())
    compare('surname', fields.get('surname'), ocr.get('Last_name_en'),
            lambda v: v.replace(' ', '').upper())
    compare('given_names', fields.get('given_names'), ocr.get('First_name_en'),
            lambda v: v.replace(' ', '').upper())

    return {'ok': all(v == 'ok' for v in checks.values()) if checks else None,
            'checks': checks}


def validate(ocr: dict) -> dict:
    """Both MRZ checks for one document's OCR dict; empty when it carries no MRZ."""
    mrz_text = (ocr or {}).get('MRZ')
    if not mrz_text:
        return {}
    return {'mrz_checksum': validate_mrz(mrz_text),
            'mrz_vs_visual': cross_check(mrz_text, ocr)}
