"""Per-field verdicts (document_processing/validation/fields.py).

Every check has a negative control here: a deliberately wrong value must come
out as ``mismatch``. A check that cannot show red guards nothing.

The zones are the fictitious documents of tests/test_mrz_validation.py (with
every check digit computed); the page values next to them are
what those zones say, so that a consistent document is the starting point and
each test changes exactly one thing.
"""
from datetime import date

from document_processing.pipeline.results import PipelineResults
from document_processing.validation import cross_check, field_checks
from document_processing.validation.fields import MISMATCH, UNCHECKED, VERIFIED, family

from document_processing.validation.mrz import check_digit
from test_mrz_validation import EXT_MRZ, EXT_OCR, EXT_OCR_CONSISTENT, INT_MRZ

# What INT_MRZ says, in the form the page prints it. The number: the zone holds
# the first three digits of the series and the six of the number; the fourth
# digit of the series opens the optional data.
INT_OCR = {
    'MRZ': INT_MRZ,
    'Licence_number': '45 07 318294',
    'Birth_date': '23.04.1987',
    'Issue_date': '15.05.2007',
    'Issue_organisation_code': '770-021',
    'Sex_ru': 'МУЖ.',
    'Last_name_ru': 'ЛЕШКИН',
    'First_name_ru': 'ПАВЕЛ',
    'Middle_name_ru': 'ПЕТРОВИЧ',
    'Birth_place_ru': 'ГОР. МОСКВА',
}

TODAY = date(2026, 10, 19)


def statuses(ocr, doc_type):
    return {k: v['status'] for k, v in field_checks(ocr, doc_type, today=TODAY).items()}


def check_ids(verdict):
    return {(c['id'], c['result']) for c in verdict['checks']}


# --- shape -------------------------------------------------------------------

def test_every_field_gets_an_entry_even_empty_ones():
    ocr = dict(INT_OCR, Living_region_ru='')
    result = field_checks(ocr, 'INTPASSPORT_2011', today=TODAY)
    assert set(result) == set(ocr)
    assert result['Living_region_ru'] == {'status': UNCHECKED, 'checks': [], 'notes': []}


def test_reading_is_never_changed():
    ocr = dict(INT_OCR, Licence_number='45 07 318295')
    before = dict(ocr)
    field_checks(ocr, 'INTPASSPORT_2011', today=TODAY)
    assert ocr == before


def test_unmeasured_families_are_unchecked_throughout():
    for doc_type in ('INTPASSPORTADDR_ALL', 'DLBACK_ALL', 'BIRTHCERT_2018', 'STS_2019', 'NONE', None):
        result = statuses({'Birth_date': '31.02.2020', 'MRZ': INT_MRZ}, doc_type)
        assert set(result.values()) == {UNCHECKED}, doc_type


def test_family_does_not_confuse_the_registration_page_or_the_licence_back():
    assert family('INTPASSPORT_2011') == 'INTPASSPORT'
    assert family('INTPASSPORTADDR_ALL') is None
    assert family('DL_2020') == 'DL'
    assert family('DLBACK_ALL') is None
    assert family('EXTPASSPORTBIO_2007') == 'EXTPASSPORT'


# --- internal passport zone --------------------------------------------------

def test_consistent_internal_passport_is_verified_field_by_field():
    result = statuses(INT_OCR, 'INTPASSPORT_2011')
    assert result.pop('Birth_place_ru') == UNCHECKED   # nothing to compare it with
    assert set(result.values()) == {VERIFIED}


def test_internal_number_is_rebuilt_from_two_places_of_the_zone():
    """The defect this work found: 10 printed digits against the zone's 9."""
    verdict = field_checks(INT_OCR, 'INTPASSPORT_2011', today=TODAY)['Licence_number']
    assert ('MRZ-2', 'ok') in check_ids(verdict)
    assert cross_check(INT_MRZ, INT_OCR)['checks']['document_number'] == 'ok'


def test_internal_number_negative_controls():
    # a digit of the number, and the series' fourth digit (lives in the optional data)
    for wrong in ('45 07 318295', '45 08 318294'):
        verdict = field_checks(dict(INT_OCR, Licence_number=wrong), 'INTPASSPORT_2011',
                               today=TODAY)['Licence_number']
        assert verdict['status'] == MISMATCH, wrong
        mrz2 = [c for c in verdict['checks'] if c['id'] == 'MRZ-2'][0]
        assert mrz2['reference'] == '4507318294'


def test_internal_issue_date_and_code_negative_controls():
    result = statuses(dict(INT_OCR, Issue_date='16.05.2007', Issue_organisation_code='770-022'),
                      'INTPASSPORT_2011')
    assert result['Issue_date'] == MISMATCH
    assert result['Issue_organisation_code'] == MISMATCH
    checks = cross_check(INT_MRZ, dict(INT_OCR, Issue_organisation_code='770-022'))['checks']
    assert checks['issue_date'] == 'ok'
    assert checks['subdivision_code'] == 'mismatch'


def test_internal_names_through_the_order_279_table():
    # Ч -> 3, Ш -> 4 follow the order's table; one changed letter must break it
    for name, wrong in (('Last_name_ru', 'ЛЕЩКИН'), ('First_name_ru', 'ПАВЕК'),
                        ('Middle_name_ru', 'ПЕТРОВИ')):
        assert statuses(dict(INT_OCR, **{name: wrong}), 'INTPASSPORT_2011')[name] == MISMATCH


def test_yo_and_ye_are_the_same_letter_for_the_zone():
    """Whether a clerk typed Ё is not shared between page and zone."""
    zone = INT_MRZ.replace('LE4KIN', 'L24KIN')
    ocr = dict(INT_OCR, MRZ=zone)
    assert statuses(ocr, 'INTPASSPORT_2011')['Last_name_ru'] == VERIFIED
    assert statuses(dict(ocr, Last_name_ru='ЛЁШКИН'), 'INTPASSPORT_2011')['Last_name_ru'] == VERIFIED


def test_sex_negative_control():
    assert statuses(dict(INT_OCR, Sex_ru='ЖЕН.'), 'INTPASSPORT_2011')['Sex_ru'] == MISMATCH


def test_failed_check_digit_takes_the_zone_part_out_of_play():
    """A zone part whose own digit fails is not a reference: the page field
    loses that comparison (unchecked), it does not become a mismatch."""
    zone = INT_MRZ.replace('RUS8704234', 'RUS8704244')     # birth date, digit kept
    result = field_checks(dict(INT_OCR, MRZ=zone), 'INTPASSPORT_2011', today=TODAY)
    assert result['MRZ']['status'] == MISMATCH
    assert 'MRZ-3' not in {c['id'] for c in result['Birth_date']['checks']}
    assert result['Birth_date']['status'] == UNCHECKED
    assert result['Licence_number']['status'] == VERIFIED   # other parts still hold


def test_malformed_zone_is_a_mismatch_and_no_reference():
    zone = INT_MRZ[:-1]
    result = field_checks(dict(INT_OCR, MRZ=zone), 'INTPASSPORT_2011', today=TODAY)
    assert result['MRZ']['status'] == MISMATCH
    assert result['Licence_number']['status'] == UNCHECKED


def test_line1_with_an_impossible_character_is_no_reference():
    """Seen on samples/: the letter O of a surname read as the digit 0 in the zone. The
    page reading is right; it must not be blamed for the zone's misreading."""
    zone = EXT_MRZ.replace('ROGOVA', 'R0GOVA')
    result = field_checks(dict(EXT_OCR_CONSISTENT, MRZ=zone), 'EXTPASSPORTBIO_2007', today=TODAY)
    assert result['MRZ']['status'] == MISMATCH
    assert ('MRZ-9', 'mismatch') in check_ids(result['MRZ'])
    assert result['Last_name_en']['status'] == UNCHECKED
    assert result['Licence_number']['status'] == VERIFIED   # line 2 still holds


def test_internal_zone_digits_outside_the_table_are_impossible():
    for wrong in ('LE0KIN', 'LE1KIN', 'LE5KIN'):
        zone = INT_MRZ.replace('LE4KIN', wrong)
        result = field_checks(dict(INT_OCR, MRZ=zone), 'INTPASSPORT_2011', today=TODAY)
        assert result['MRZ']['status'] == MISMATCH, wrong
        assert result['Last_name_ru']['status'] == UNCHECKED, wrong


def test_zone_is_verified_only_through_its_names():
    """Line-2 check digits say nothing about line 1."""
    names = ('Last_name_ru', 'First_name_ru', 'Middle_name_ru')
    no_names = {k: v for k, v in INT_OCR.items() if k not in names}
    assert statuses(no_names, 'INTPASSPORT_2011')['MRZ'] == UNCHECKED
    assert statuses(INT_OCR, 'INTPASSPORT_2011')['MRZ'] == VERIFIED
    # a page name that disagrees withholds the confirmation, it does not blame the zone
    result = statuses(dict(INT_OCR, First_name_ru='ПАВЕК'), 'INTPASSPORT_2011')
    assert result['MRZ'] == UNCHECKED
    assert result['First_name_ru'] == MISMATCH


def test_zone_layout_follows_the_classifier_not_the_zone_type_characters():
    # 'PH' also fails the line-1 form (MRZ-9), so names are not compared; the
    # optional data of line 2 is still read with the internal layout
    zone = 'PH' + INT_MRZ[2:]          # type characters misread
    assert statuses(dict(INT_OCR, MRZ=zone), 'INTPASSPORT_2011')['Issue_date'] == VERIFIED


def test_digits_agreeing_do_not_hide_stray_characters():
    """Damir set: a glare read as quotes inside the number, a doubled hyphen in
    the code. The zone comparisons look at digits only; the form catches it."""
    result = statuses(dict(INT_OCR, Licence_number='88 12 106"436""',
                           Issue_organisation_code='770--021'), 'INTPASSPORT_2011')
    assert result['Licence_number'] == MISMATCH
    assert result['Issue_organisation_code'] == MISMATCH
    # whitespace is not part of the form
    assert statuses(dict(INT_OCR, Licence_number='45  07 318294'),
                    'INTPASSPORT_2011')['Licence_number'] == VERIFIED


def test_external_number_form():
    assert statuses(dict(EXT_OCR_CONSISTENT, Licence_number='764820157'),
                    'EXTPASSPORTBIO_2007')['Licence_number'] == MISMATCH


# --- external passport zone --------------------------------------------------

def test_consistent_external_passport():
    ocr = dict(EXT_OCR_CONSISTENT, Issue_date='20.05.2021')
    result = statuses(ocr, 'EXTPASSPORTBIO_2007')
    assert set(result.values()) == {VERIFIED}


def test_external_negative_controls():
    ocr = dict(EXT_OCR_CONSISTENT, Expiration_date='20.05.2032', Last_name_en='ROGOVA',
               First_name_en='IRENA')
    result = statuses(ocr, 'EXTPASSPORTBIO_2007')
    assert result['Expiration_date'] == MISMATCH
    assert result['First_name_en'] == MISMATCH


def test_the_specimens_altered_number_is_a_mismatch():
    """As on the anonymised samples/: the printed number differs, the zone is valid."""
    assert statuses(EXT_OCR, 'EXTPASSPORTBIO_2007')['Licence_number'] == MISMATCH


def _td3_line2(number, birth, sex, expiry, personal=''):
    """A TD3 second line with every check digit computed, for building cases."""
    def digit(s):
        return str(check_digit(s))
    number, personal = number.ljust(9, '<'), personal.ljust(14, '<')
    line = (number + digit(number) + 'RUS' + birth + digit(birth) + sex
            + expiry + digit(expiry) + personal + digit(personal))
    composite = line[0:10] + line[13:20] + line[21:43]
    return line + digit(composite)


def test_td3_builder_reproduces_a_real_zone():
    assert _td3_line2('764820157', '880314', 'F', '310520') == EXT_MRZ.splitlines()[1]


def test_seven_digit_zone_number_of_the_2003_form():
    """Five of 19 references of the 2003 form write the number without its series."""
    zone = EXT_MRZ.splitlines()[0] + '\n' + _td3_line2('4820157', '880314', 'F', '310520')
    ocr = dict(EXT_OCR_CONSISTENT, MRZ=zone)
    result = field_checks(ocr, 'EXTPASSPORT_2003', today=TODAY)
    assert result['MRZ']['status'] == VERIFIED
    assert ('MRZ-2', 'ok') in check_ids(result['Licence_number'])
    bad = field_checks(dict(ocr, Licence_number='76 4820158'), 'EXTPASSPORT_2003',
                       today=TODAY)['Licence_number']
    assert bad['status'] == MISMATCH


def test_truncated_name_compares_by_prefix():
    long_name = 'KONSTANTINOPOLSKAYA' * 3
    zone_line1 = ('P<RUS' + 'ROGOVA<<' + long_name)[:44]
    zone = zone_line1 + '\n' + EXT_MRZ.splitlines()[1]
    ocr = dict(EXT_OCR_CONSISTENT, MRZ=zone, First_name_en=long_name)
    assert statuses(ocr, 'EXTPASSPORTBIO_2007')['First_name_en'] == VERIFIED


# --- dates -------------------------------------------------------------------

DL_OCR = {'Birth_date': '01.02.1990', 'Issue_date': '15.03.2015',
          'Expiration_date': '15.03.2025', 'Licence_number': '99 12 345678'}


def test_licence_validity_confirms_both_dates():
    result = statuses(DL_OCR, 'DL_2011')
    assert result['Issue_date'] == VERIFIED
    assert result['Expiration_date'] == VERIFIED
    assert result['Birth_date'] == UNCHECKED      # only refuting checks reach it
    assert result['Licence_number'] == UNCHECKED


def test_licence_validity_negative_control_marks_both_with_references():
    result = field_checks(dict(DL_OCR, Expiration_date='16.03.2025'), 'DL_2020', today=TODAY)
    assert result['Issue_date']['status'] == MISMATCH
    assert result['Expiration_date']['status'] == MISMATCH
    dl2 = [c for c in result['Expiration_date']['checks'] if c['id'] == 'DL-2'][0]
    assert dl2['reference'] == '15.03.2025'
    dl2 = [c for c in result['Issue_date']['checks'] if c['id'] == 'DL-2'][0]
    assert dl2['reference'] == '16.03.2015'


def test_external_validity_accepts_five_or_ten_years():
    for years in (5, 10):
        ocr = {'Issue_date': '01.06.2010', 'Expiration_date': f'01.06.{2010 + years}'}
        assert statuses(ocr, 'EXTPASSPORT_2003')['Issue_date'] == VERIFIED
    ocr = {'Issue_date': '01.06.2010', 'Expiration_date': '01.06.2017'}
    assert statuses(ocr, 'EXTPASSPORTBIO_2007')['Issue_date'] == MISMATCH


def test_leap_day_validity():
    ocr = {'Issue_date': '29.02.2016', 'Expiration_date': '28.02.2026'}
    assert statuses(ocr, 'DL_2011')['Expiration_date'] == VERIFIED


def test_not_a_calendar_date():
    assert statuses({'Birth_date': '31.02.1990'}, 'SNILS_1996')['Birth_date'] == MISMATCH
    assert statuses({'Birth_date': '10 ДЕКАБРЯ 1999 ГОДА'}, 'SNILS_1996')['Birth_date'] == UNCHECKED


def test_issue_before_birth_marks_both():
    result = statuses({'Birth_date': '01.01.2000', 'Issue_date': '01.01.1999'}, 'SNILS_1996')
    assert result == {'Birth_date': MISMATCH, 'Issue_date': MISMATCH}


def test_future_date():
    assert statuses({'Issue_date': '01.01.2030'}, 'SNILS_1996')['Issue_date'] == MISMATCH


def test_minimum_age_at_issue():
    young = {'Birth_date': '01.01.2000', 'Issue_date': '31.12.2013'}
    assert statuses(young, 'INTPASSPORT_1997')['Issue_date'] == MISMATCH
    assert statuses(dict(young, Issue_date='01.01.2014'), 'INTPASSPORT_1997')['Issue_date'] == UNCHECKED
    assert statuses({'Birth_date': '01.01.2000', 'Issue_date': '01.06.2015'}, 'DL_2020')['Birth_date'] == MISMATCH
    # SNILS is issued to children: no age rule
    assert statuses(young, 'SNILS_1996')['Issue_date'] == UNCHECKED


# --- SNILS -------------------------------------------------------------------

def test_snils_check_number():
    # the worked example of docs/validation-checks.md
    assert statuses({'Licence_number': '112-233-445 95'}, 'SNILS_1996')['Licence_number'] == VERIFIED
    verdict = field_checks({'Licence_number': '112-233-445 96'}, 'SNILS_1996', today=TODAY)
    assert verdict['Licence_number']['status'] == MISMATCH
    assert verdict['Licence_number']['checks'][-1]['reference'] == '95'
    # a changed digit in the body
    assert statuses({'Licence_number': '112-233-446 95'}, 'SNILS_1996')['Licence_number'] == MISMATCH


def test_snils_check_number_wraps_mod_101():
    # all nines: 9 * (9+8+...+1) = 405 -> 405 mod 101 = 1
    assert statuses({'Licence_number': '999-999-999 01'}, 'SNILS_1996')['Licence_number'] == VERIFIED
    # 9*9 + 2*8 + 3*1 = 100 -> 00, not 100
    assert statuses({'Licence_number': '920-000-003 00'}, 'SNILS_1996')['Licence_number'] == VERIFIED
    assert statuses({'Licence_number': '920-000-003 01'}, 'SNILS_1996')['Licence_number'] == MISMATCH


def test_snils_length_and_small_numbers():
    assert statuses({'Licence_number': '112-233-445 9'}, 'SNILS_1996')['Licence_number'] == MISMATCH
    assert statuses({'Licence_number': '112-233-445-95'}, 'SNILS_1996')['Licence_number'] == MISMATCH
    # up to 001-001-998 there is no valid check number to verify
    assert statuses({'Licence_number': '001-001-998 00'}, 'SNILS_1996')['Licence_number'] == UNCHECKED


# --- soft --------------------------------------------------------------------

def test_series_year_is_a_note_not_a_verdict():
    ocr = dict(INT_OCR, Licence_number='45 02 318294')
    result = field_checks(ocr, 'INTPASSPORT_1997', today=TODAY)
    notes = result['Licence_number']['notes']
    assert notes and notes[0]['id'] == 'PASS-3'
    # the note does not change any verdict by itself
    plain = field_checks(dict(ocr, MRZ=''), 'INTPASSPORT_1997', today=TODAY)
    assert plain['Licence_number']['status'] == UNCHECKED
    assert plain['Licence_number']['notes'][0]['id'] == 'PASS-3'


# --- results -----------------------------------------------------------------

def test_results_expose_the_verdicts_outside_the_full_report():
    results = PipelineResults()
    results._meta_results['OCR'] = dict(INT_OCR)
    results._meta_results['FieldChecks'] = field_checks(INT_OCR, 'INTPASSPORT_2011', today=TODAY)
    assert results.field_checks['Licence_number']['status'] == VERIFIED
    assert 'FieldChecks' not in results.full_report
    # a copy: the caller cannot edit the stored verdicts
    results.field_checks['Licence_number']['status'] = MISMATCH
    assert results.field_checks['Licence_number']['status'] == VERIFIED


def test_results_without_reading_have_no_verdicts():
    assert PipelineResults().field_checks == {}
