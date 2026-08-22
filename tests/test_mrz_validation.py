"""Tests for the MRZ checks (EXT-4 / EXT-5 in docs/validation-checks.md).

The fixtures are not invented: both zones were read by the shipped Latin OCR
off the sample images named beside them, and the expected visual-zone values
come from those samples' ground-truth JSON. That is deliberate - a hand-typed
MRZ can be made to satisfy any implementation, while these strings tie the
checks to what the pipeline actually produces.
"""
from document_processing.validation import (check_digit, cross_check, parse_mrz,
                                            validate, validate_mrz)

# samples/EXTPASSPORTBIO_2007/1_CR_EXTPASSPORTBIO_2010.jpg - full TD3.
EXT_MRZ = ('P<RUSPOPOVA<<ALENA<<<<<<<<<<<<<<<<<<<<<<<<<<\n'
           '7515919230RUS8410102F2506090<<<<<<<<<<<<<<02')
# The same sample's ground truth (samples/.../1_CR_EXTPASSPORTBIO_2010.json).
# NOTE the number: this specimen is internally inconsistent. Its visual zone
# prints "75 1591921" while its MRZ carries 751591923 - and the MRZ check digit
# agrees with the MRZ value, so the zone was generated valid and the printed
# number was altered afterwards (or the reverse). Both were re-read off the
# image at high zoom; neither is an OCR error. It is the exact disagreement
# EXT-5 exists to surface, so it is kept as a fixture rather than "fixed".
EXT_OCR = {
    'MRZ': EXT_MRZ,
    'Licence_number': '75 1591921',
    'Birth_date': '10.10.1984',
    'Expiration_date': '09.06.2025',
    'Last_name_en': 'POPOVA',
    'First_name_en': 'ALENA',
    'Sex_en': 'F',
}
#: The same document as it would read if the printed number matched the zone.
EXT_OCR_CONSISTENT = dict(EXT_OCR, Licence_number='75 1591923')

# samples/INTPASSPORT_2011/1_CR_INTPASSPORT_2011.jpg - internal passport:
# type PN, no expiry date, Cyrillic-only letters written as digits (SERGEEVI3).
INT_MRZ = ('PNRUSBEKE4EV<<STEPAN<SERGEEVI3<<<<<<<<<<<<<<\n'
           '8811064367RUS9210077M<<<<<<<2121212120012<76')


def test_check_digit_matches_icao_weights():
    # 7-3-1 over "751591923": digits only, so the weights alone decide.
    assert check_digit('751591923') == 0
    # Letters count as 10..35; fillers as 0.
    assert check_digit('<<<<<<<<<<<<<<') == 0
    assert check_digit('AB') == 3          # 10*7 + 11*3 = 103


def test_check_digit_rejects_characters_outside_the_alphabet():
    assert check_digit('75159192!') is None


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
    corrupted = EXT_MRZ.replace('7515919230', '7515919240')
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
    assert fields['surname'] == 'POPOVA'
    assert fields['given_names'] == 'ALENA'
    assert fields['document_number'] == '751591923'   # see EXT_OCR note
    assert fields['birth_date'] == '10.10.1984'
    assert fields['expiry_date'] == '09.06.2025'
    assert fields['sex'] == 'F'


def test_expiry_date_is_read_into_this_century():
    """An expiry year above the current one is future, not 19xx."""
    mrz = ('P<RUSTSEITLIN<<FELIKS<<<<<<<<<<<<<<<<<<<<<<<\n'
           '7616733795RUS8904034M2910024<<<<<<<<<<<<<<06')
    fields = parse_mrz(mrz)
    assert fields['expiry_date'] == '02.10.2029'
    assert fields['birth_date'] == '03.04.1989'


def test_impossible_calendar_date_is_rejected():
    broken = EXT_MRZ.replace('RUS8410102', 'RUS8413102')  # month 13
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
    ocr = dict(EXT_OCR_CONSISTENT, Birth_date='11.10.1984')
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
    result = cross_check(INT_MRZ, {'MRZ': INT_MRZ, 'Birth_date': '07.10.1992'})
    assert result['checks'] == {'birth_date': 'ok'}


def test_validate_is_empty_without_an_mrz():
    assert validate({'Last_name_ru': 'ПОПОВА'}) == {}
    assert validate({}) == {}


def test_validate_returns_both_checks():
    result = validate(EXT_OCR_CONSISTENT)
    assert result['mrz_checksum']['ok']
    assert result['mrz_vs_visual']['ok']
