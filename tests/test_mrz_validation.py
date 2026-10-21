"""Tests for the MRZ checks (EXT-4 / EXT-5 in docs/validation-checks.md).

The fixtures are FICTITIOUS documents (user's decision 2026-10-21): the zones
used to be the readings of two sample images whose data may belong to real
people, so they were replaced by invented ones built with every check digit
computed and the same layout as those readings (internal passport: type PN,
the series split between the number and the optional data, Cyrillic-only
letters written as digits). The tie to what the pipeline actually produces is
kept by the conformance cases and tests/test_quality.py, not by these strings.
"""
from document_processing.validation import (check_digit, cross_check, parse_mrz,
                                            validate, validate_mrz)

# A fictitious biometric passport - full TD3.
EXT_MRZ = ('P<RUSROGOVA<<IRINA<<<<<<<<<<<<<<<<<<<<<<<<<<\n'
           '7648201572RUS8803148F3105205<<<<<<<<<<<<<<04')
# Its visual zone. NOTE the number: the document is deliberately inconsistent,
# as the anonymised samples/ passports are. The page prints "76 4820154" while
# the MRZ carries 764820157 - and the MRZ check digit agrees with the MRZ value,
# i.e. the zone is valid and the printed number differs in its last digit. It is
# the exact disagreement EXT-5 exists to surface.
EXT_OCR = {
    'MRZ': EXT_MRZ,
    'Licence_number': '76 4820154',
    'Birth_date': '14.03.1988',
    'Expiration_date': '20.05.2031',
    'Last_name_en': 'ROGOVA',
    'First_name_en': 'IRINA',
    'Sex_en': 'F',
}
#: The same document as it would read if the printed number matched the zone.
EXT_OCR_CONSISTENT = dict(EXT_OCR, Licence_number='76 4820157')

# A fictitious internal passport (2011 form):
# type PN, no expiry date, Cyrillic-only letters written as digits (PETROVI3).
INT_MRZ = ('PNRUSLE4KIN<<PAVEL<PETROVI3<<<<<<<<<<<<<<<<<\n'
           '4503182940RUS8704234M<<<<<<<7070515770021<46')


def test_check_digit_matches_icao_weights():
    # 7-3-1 over "764820157": digits only, so the weights alone decide.
    assert check_digit('764820157') == 2
    # Letters count as 10..35; fillers as 0.
    assert check_digit('<<<<<<<<<<<<<<') == 0
    assert check_digit('AB') == 3          # 10*7 + 11*3 = 103


def test_check_digit_rejects_characters_outside_the_alphabet():
    assert check_digit('76482015!') is None


def test_external_passport_passes_every_check():
    result = validate_mrz(EXT_MRZ)
    assert result['ok']
    # The personal-number field is empty on a Russian passport, but unlike the
    # internal passport's expiry date it still carries a real check digit
    # (zero, over all-filler) - so it verifies rather than being skipped.
    assert result['checks'] == {
        'document_number': 'ok',
        'birth_date': 'ok',
        'expiry_date': 'ok',
        'personal_number': 'ok',
        'composite': 'ok',
    }


def test_internal_passport_has_no_expiry_date():
    """The zone carries filler where the expiry date and its digit would be.

    Scoring that as a failed check digit would reject every correctly read
    internal passport, so it must be reported as not applicable.
    """
    result = validate_mrz(INT_MRZ)
    assert result['checks']['expiry_date'] == 'not_applicable'
    assert result['checks']['composite'] == 'ok'
    assert result['ok']


def test_single_wrong_character_is_caught():
    corrupted = EXT_MRZ.replace('7648201572', '7648201582')
    result = validate_mrz(corrupted)
    assert not result['ok']
    assert result['checks']['document_number'] == 'failed'


def test_malformed_line_is_a_finding_not_a_pass():
    """A short line means characters were lost; it must not read as success."""
    short = EXT_MRZ[:-1]
    result = validate_mrz(short)
    assert not result['ok']
    assert 'malformed' in result['error']


def test_missing_second_line():
    result = validate_mrz(EXT_MRZ.splitlines()[0])
    assert not result['ok']
    assert result['error'] == 'expected 2 lines'


def test_parse_external_passport_fields():
    fields = parse_mrz(EXT_MRZ)
    assert fields['document_type'] == 'P'
    assert fields['issuing_state'] == 'RUS'
    assert fields['surname'] == 'ROGOVA'
    assert fields['given_names'] == 'IRINA'
    assert fields['document_number'] == '764820157'   # see EXT_OCR note
    assert fields['birth_date'] == '14.03.1988'
    assert fields['expiry_date'] == '20.05.2031'
    assert fields['sex'] == 'F'


def test_expiry_date_is_read_into_this_century():
    """An expiry year above the current one is future, not 19xx."""
    mrz = ('P<RUSBELOV<<OLEG<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n'
           '7654321097RUS8904034M2910024<<<<<<<<<<<<<<02')
    fields = parse_mrz(mrz)
    assert fields['expiry_date'] == '02.10.2029'
    assert fields['birth_date'] == '03.04.1989'


def test_impossible_calendar_date_is_rejected():
    broken = EXT_MRZ.replace('RUS8803148', 'RUS8813148')  # month 13
    assert parse_mrz(broken)['birth_date'] is None


def test_cross_check_against_the_visual_zone():
    result = cross_check(EXT_MRZ, EXT_OCR_CONSISTENT)
    assert result['ok']
    assert result['checks'] == {
        'document_number': 'ok',
        'birth_date': 'ok',
        'expiry_date': 'ok',
        'sex': 'ok',
        'surname': 'ok',
        'given_names': 'ok',
    }


def test_cross_check_catches_the_specimens_altered_number():
    """The shipped sample really does disagree with its own MRZ - see EXT_OCR."""
    result = cross_check(EXT_MRZ, EXT_OCR)
    assert not result['ok']
    assert result['checks']['document_number'] == 'mismatch'
    assert result['checks']['birth_date'] == 'ok'


def test_cross_check_reports_a_disagreeing_field():
    ocr = dict(EXT_OCR_CONSISTENT, Birth_date='15.03.1988')
    result = cross_check(EXT_MRZ, ocr)
    assert not result['ok']
    assert result['checks']['birth_date'] == 'mismatch'
    assert result['checks']['document_number'] == 'ok'


def test_cross_check_skips_fields_the_detector_did_not_find():
    """A missing field says nothing about agreement and must not be a mismatch."""
    ocr = {k: v for k, v in EXT_OCR_CONSISTENT.items() if k != 'Last_name_en'}
    result = cross_check(EXT_MRZ, ocr)
    assert 'surname' not in result['checks']
    assert result['ok']


def test_internal_passport_names_are_not_cross_checked():
    """Its MRZ writes Cyrillic-only letters as digits, so no transliteration
    can match it character for character - and it prints no Latin name at all."""
    result = cross_check(INT_MRZ, {'MRZ': INT_MRZ, 'Birth_date': '23.04.1987'})
    assert result['checks'] == {'birth_date': 'ok'}


def test_validate_is_empty_without_an_mrz():
    assert validate({'Last_name_ru': 'РОГОВА'}) == {}
    assert validate({}) == {}


def test_validate_returns_both_checks():
    result = validate(EXT_OCR_CONSISTENT)
    assert result['mrz_checksum']['ok']
    assert result['mrz_vs_visual']['ok']
