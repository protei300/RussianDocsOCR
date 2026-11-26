"""Regression tests for the confirmed Wave 3 fixes (2026-11-24).

All fixtures are 100% fictitious (no data from real samples).
"""
from datetime import date

import numpy as np

from document_processing.pipeline.dates import to_ddmmyyyy
from document_processing.pipeline.mrz_reading import MrzReadingStage
from document_processing.pipeline.stamps import StampsStage
from document_processing.pipeline_modules.doc_detector.image_transformation import order_points
from document_processing.pipeline_modules.ocr_corrections import check_en_sex, check_rus_sex
from document_processing.validation import cross_check, field_checks, parse_mrz, validate_mrz
from document_processing.validation.fields import MISMATCH, UNCHECKED, VERIFIED, family

from test_mrz_validation import EXT_MRZ, EXT_OCR_CONSISTENT, INT_MRZ
from test_field_checks import INT_OCR

TODAY = date(2026, 11, 24)


def _statuses(ocr, doc_type):
    return {k: v['status'] for k, v in field_checks(ocr, doc_type, today=TODAY).items()}


# --- W3-1: MRZ & field validation --------------------------------------------

def test_extpassport_compound_given_name_is_verified():
    line1 = ('P<RUSROGOVA<<ANNA<MARIA').ljust(44, '<')
    zone = line1 + '\n' + EXT_MRZ.splitlines()[1]
    ocr = dict(EXT_OCR_CONSISTENT, MRZ=zone, First_name_en='ANNA-MARIA')
    res = _statuses(ocr, 'EXTPASSPORTBIO_2007')
    assert res['First_name_en'] == VERIFIED
    assert res['MRZ'] == VERIFIED


def test_truncated_line1_only_prefixes_last_non_empty_segment():
    # Line 1 is full (no trailing '<'): given name is truncated at char 44,
    # surname is bounded by '<<' and must NOT match by prefix.
    long_given = 'KONSTANTINOPOLSKAYA' * 3
    line1 = ('P<RUSROGOVA<<' + long_given)[:44]
    assert not line1.endswith('<')
    zone = line1 + '\n' + EXT_MRZ.splitlines()[1]
    ocr = dict(EXT_OCR_CONSISTENT, MRZ=zone, Last_name_en='ROGOVAEXTRA', First_name_en=long_given)
    res = _statuses(ocr, 'EXTPASSPORTBIO_2007')
    assert res['First_name_en'] == VERIFIED
    assert res['Last_name_en'] == MISMATCH


def test_mrz7_sex_requires_exact_token():
    # Garbage starting with 'М'/'Ж' must not verify as sex.
    assert _statuses(dict(INT_OCR, Sex_ru='МОСКВА'), 'INTPASSPORT_2011')['Sex_ru'] == MISMATCH
    assert _statuses(dict(EXT_OCR_CONSISTENT, Sex_en='FEMALE'), 'EXTPASSPORTBIO_2007')['Sex_en'] == MISMATCH
    # Exact valid tokens still verify.
    for valid in ('М', 'МУЖ', 'МУЖ.'):
        assert _statuses(dict(INT_OCR, Sex_ru=valid), 'INTPASSPORT_2011')['Sex_ru'] == VERIFIED
    for valid in ('F', 'Ж', 'ЖЕН', 'ЖЕН.'):
        assert _statuses(dict(EXT_OCR_CONSISTENT, Sex_en=valid), 'EXTPASSPORTBIO_2007')['Sex_en'] == VERIFIED


def test_three_line_mrz_is_rejected():
    three_lines = EXT_MRZ + '\nEXTRA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
    v = validate_mrz(three_lines, today=TODAY)
    assert not v['ok']
    assert v['error'] == 'expected 2 lines'
    res = field_checks(dict(EXT_OCR_CONSISTENT, MRZ=three_lines), 'EXTPASSPORTBIO_2007', today=TODAY)
    assert res['MRZ']['status'] == MISMATCH
    ids = {(c['id'], c['result']) for c in res['MRZ']['checks']}
    assert ('MRZ-1', MISMATCH) in ids
    assert ('MRZ-9', MISMATCH) in ids


def test_parse_mrz_respects_today_parameter():
    # Birth year '25': in 2024 it is 1925, in 2026 it is 2025.
    line2 = EXT_MRZ.splitlines()[1].replace('RUS8803148', 'RUS2503140')
    zone = EXT_MRZ.splitlines()[0] + '\n' + line2
    assert parse_mrz(zone, today=date(2024, 1, 1))['birth_date'] == '14.03.1925'
    assert parse_mrz(zone, today=date(2026, 1, 1))['birth_date'] == '14.03.2025'


def test_cross_check_handles_non_string_and_russian_month():
    ocr = dict(INT_OCR, Birth_date='23 АПРЕЛЯ 1987', Issue_date=None, Licence_number=123)
    res = cross_check(INT_MRZ, ocr, internal=True, today=TODAY)
    assert res['checks']['birth_date'] == 'ok'
    assert res['checks']['sex'] == 'ok'
    assert 'issue_date' not in res['checks']
    assert 'document_number' not in res['checks']


# --- W3-2: MIGRATIONCARD validation ------------------------------------------

MIG_OCR = {
    'Card_series': '4501',
    'Card_number': '1234567',
    'Birth_date': '15.05.1990',
    'Stay_from': '01.11.2026',
    'Stay_to': '30.01.2027',
}


def test_migration_card_family_and_valid_card():
    assert family('MIGRATIONCARD_ALL') == 'MIGRATIONCARD'
    res = field_checks(MIG_OCR, 'MIGRATIONCARD_ALL', today=TODAY)
    # All migration checks are refuting -> valid card stays unchecked with ok checks.
    for key in MIG_OCR:
        assert res[key]['status'] == UNCHECKED
        assert all(c['result'] == 'ok' for c in res[key]['checks'])
    assert {c['id'] for c in res['Card_series']['checks']} == {'MIG-1'}
    assert {c['id'] for c in res['Card_number']['checks']} == {'MIG-2'}
    assert {c['id'] for c in res['Stay_to']['checks']} == {'GEN-1', 'MIG-3'}
    # Spaced series/number and compact DDMMYYYY stay dates also pass.
    spaced = dict(MIG_OCR, Card_series='45 01', Card_number='123 4567',
                  Stay_from='01112026', Stay_to='30012027')
    assert set(_statuses(spaced, 'MIGRATIONCARD_ALL').values()) == {UNCHECKED}


def test_migration_card_negative_controls():
    assert _statuses(dict(MIG_OCR, Card_series='450'), 'MIGRATIONCARD_ALL')['Card_series'] == MISMATCH
    assert _statuses(dict(MIG_OCR, Card_series='45A1'), 'MIGRATIONCARD_ALL')['Card_series'] == MISMATCH
    assert _statuses(dict(MIG_OCR, Card_number='123456'), 'MIGRATIONCARD_ALL')['Card_number'] == MISMATCH
    assert _statuses(dict(MIG_OCR, Birth_date='31.02.1990'), 'MIGRATIONCARD_ALL')['Birth_date'] == MISMATCH
    # Stay_from > Stay_to marks both via MIG-3.
    rev = _statuses(dict(MIG_OCR, Stay_from='10.12.2026', Stay_to='01.12.2026'), 'MIGRATIONCARD_ALL')
    assert rev['Stay_from'] == MISMATCH
    assert rev['Stay_to'] == MISMATCH
    # Misread future Birth_date must flag Birth_date ONLY, NOT false-alarm valid Stay_from.
    fut = _statuses(dict(MIG_OCR, Birth_date='15.05.2076'), 'MIGRATIONCARD_ALL')
    assert fut['Birth_date'] == MISMATCH
    assert fut['Stay_from'] == UNCHECKED
    # Valid past Birth_date >= Stay_from marks both via GEN-2.
    bad_order = _statuses(dict(MIG_OCR, Birth_date='01.01.2025', Stay_from='01.01.2024', Stay_to='01.03.2024'),
                          'MIGRATIONCARD_ALL')
    assert bad_order['Birth_date'] == MISMATCH
    assert bad_order['Stay_from'] == MISMATCH


# --- W3-3: registration stamp cancellation pairing ---------------------------

def test_tie_cancellations_does_not_re_cancel_or_promote_older():
    # Freshest first: [dereg, reg_new, reg_old]
    # reg_new is cancelled -> nobody is current (must NOT promote reg_old).
    stamps = [
        {'kind': 'deregistered', 'cancels': None, 'current': False},
        {'kind': 'registered', 'cancels': None, 'current': False},
        {'kind': 'registered', 'cancels': None, 'current': False},
    ]
    StampsStage._tie_cancellations(stamps)
    assert stamps[0]['cancels'] == 1
    assert not any(s['current'] for s in stamps)

    # Two deregistered stamps before two registrations must cancel distinct ones.
    stamps2 = [
        {'kind': 'deregistered', 'cancels': None, 'current': False},
        {'kind': 'deregistered', 'cancels': None, 'current': False},
        {'kind': 'registered', 'cancels': None, 'current': False},
        {'kind': 'registered', 'cancels': None, 'current': False},
    ]
    StampsStage._tie_cancellations(stamps2)
    assert stamps2[0]['cancels'] == 2
    assert stamps2[1]['cancels'] == 3
    assert not any(s['current'] for s in stamps2)

    # Active newest registration with older cancelled one stays current.
    stamps3 = [
        {'kind': 'registered', 'cancels': None, 'current': False},
        {'kind': 'deregistered', 'cancels': None, 'current': False},
        {'kind': 'registered', 'cancels': None, 'current': False},
    ]
    StampsStage._tie_cancellations(stamps3)
    assert stamps3[1]['cancels'] == 2
    assert stamps3[0]['current'] is True
    assert stamps3[2]['current'] is False


# --- W3-4: strict to_ddmmyyyy ------------------------------------------------

def test_to_ddmmyyyy_rejects_stray_numbers_and_iso():
    assert to_ddmmyyyy('2020-05-12') is None
    assert to_ddmmyyyy('12.05.2020 99') is None
    assert to_ddmmyyyy('01.02.1990 15.03.2015') is None
    assert to_ddmmyyyy('123.05.2020') is None
    # Legitimate formats remain intact.
    assert to_ddmmyyyy('12.05.2020') == '12.05.2020'
    assert to_ddmmyyyy('2010 ГОДА ИЮНЯ МЕСЯЦА 15 ЧИСЛА') == '15.06.2010'


# --- W3-5: _read_mrz does not retry on >= 44 chars ---------------------------

def test_read_mrz_skips_retries_when_already_overlong():
    class _SpyMrz(MrzReadingStage):
        def __init__(self):
            self.calls = 0
            self._mrz_zone = {
                'canvas': np.zeros((100, 500, 3), dtype=np.uint8),
                'boxes': [[10, 10, 490, 40]],
                'span': None,
            }
            self.ocr_lat = self

        @property
        def model_name(self):
            return 'm'

        def predict(self, crop):
            self.calls += 1
            return {'m': {'ocr_output': 'A' * 50}}

        def fix_errors(self, field_type, text):
            return text

    spy = _SpyMrz()
    overlong = 'P<RUSROGOVA<<IRINA<<<<<<<<<<<<<<<<<<<<<<<<<<<'  # 45 chars
    assert spy._read_mrz(0, overlong) == overlong
    assert spy.calls == 0


# --- W3-6: empty sex stays empty ---------------------------------------------

def test_empty_sex_correction_returns_empty():
    assert check_en_sex('') == ''
    assert check_en_sex('...') == ''
    assert check_en_sex('  ') == ''
    assert check_rus_sex('') == ''
    assert check_rus_sex('...') == ''
    assert check_rus_sex('  ') == ''
    assert check_en_sex('M') == 'M'
    assert check_en_sex('F') == 'F'
    assert check_rus_sex('МУЖ.') == 'М'
    assert check_rus_sex('ЖЕН.') == 'Ж'


# --- W3-8: order_points returns 4 distinct clockwise vertices ----------------

def test_order_points_preserves_four_distinct_vertices_at_45_deg():
    diamond = np.array([[10, 0], [20, 10], [10, 20], [0, 10]], dtype=np.float32)
    ordered = order_points(diamond)
    assert ordered.shape == (4, 2)
    assert len({tuple(p) for p in ordered.tolist()}) == 4
    # Clockwise from TL=(10,0): (10,0) -> (20,10) -> (10,20) -> (0,10)
    assert ordered.tolist() == [[10.0, 0.0], [20.0, 10.0], [10.0, 20.0], [0.0, 10.0]]
    # Axis-aligned rectangle gives standard [TL, TR, BR, BL].
    rect = np.array([[100, 200], [10, 20], [100, 20], [10, 200]], dtype=np.float32)
    assert order_points(rect).tolist() == [[10.0, 20.0], [100.0, 20.0], [100.0, 200.0], [10.0, 200.0]]
