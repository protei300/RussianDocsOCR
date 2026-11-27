"""OCR options dispatch, field routing, and join contracts across document types.

Covers BIRTHCERT (1998/2018), DL & DLBACK, STS/STSBACK (old & new forms),
MIGRATIONCARD, passports (MRZ & Licence_number engine routing), and VIN fixes.
No ONNX models are loaded here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from document_processing.pipeline.options import (
    OCROptionsBIRTHCERT,
    OCROptionsClass,
    OCROptionsDL,
    OCROptionsEXTPassport,
    OCROptionsINTPassport,
    OCROptionsMigrationCard,
    OCROptionsSTS,
)
from document_processing.pipeline.pipeline import Pipeline
from document_processing.pipeline_modules.ocr_corrections import check_vin
from document_processing.pipeline_modules.ocr_latin.ocr_latin import OCRLatin

REPO_ROOT = Path(__file__).resolve().parent.parent

FIELDS_BIRTHCERT_2018 = (
    'Last_name_ru', 'First_name_ru', 'Birth_date', 'Birth_place_ru',
    'Issue_date', 'Issue_organization_ru', 'Issue_place_ru', 'Licence_number',
    'Father_last_name_ru', 'Father_first_middle_ru', 'Father_birth_date',
    'Mother_last_name_ru', 'Mother_first_middle_ru', 'Mother_birth_date',
    'Act_number',
)

FIELDS_STS_FRONT = (
    'Reg_number', 'VIN', 'Vehicle_make_ru', 'Vehicle_make_en', 'Vehicle_type',
    'Vehicle_category', 'Vehicle_year', 'Chassis_number', 'Body_number',
    'Vehicle_color', 'Engine_power', 'Eco_class', 'Max_mass', 'Curb_mass',
    'Expiration_date', 'PTS_number', 'Licence_number',
)
FIELDS_STS_BACK = (
    'Licence_number', 'Last_name_ru', 'Last_name_en', 'First_name_ru',
    'First_name_en', 'Middle_name_ru', 'Living_region_ru', 'House_number',
    'Apartment_number', 'Special_marks', 'Issue_organisation_code', 'Issue_date',
)
FIELDS_STS_NEW_FORM = ('Type_approval', 'Building_number')
FIELDS_STS_OLD_2010 = (
    'Engine_model', 'Engine_number', 'Engine_volume',
    'Issue_organization_ru', 'Issue_date', 'Building_number',
)

MRZ_LINE1 = 'P<RUSROGOVA<<IRINA<<<<<<<<<<<<<<<<<<<<<<<<<<'
MRZ_LINE2 = '7648201572RUS8803148F3105205<<<<<<<<<<<<<<04'


# ---------------------------------------------------------------------------
# BIRTHCERT options & date join
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('doc_type', ['BIRTHCERT', 'BIRTHCERT_1998', 'BIRTHCERT_2018'])
def test_birthcert_every_era_reaches_the_same_options(doc_type):
    assert isinstance(OCROptionsClass.make_options(doc_type), OCROptionsBIRTHCERT)


def test_birthcert_all_2018_fields_are_read():
    options = OCROptionsBIRTHCERT()
    missing = [f for f in FIELDS_BIRTHCERT_2018 if f not in options.ru_fields + options.en_fields]
    assert not missing, f'not routed to any engine: {missing}'


def test_birthcert_nothing_is_routed_to_the_latin_engine():
    assert OCROptionsBIRTHCERT().en_fields == []


def test_birthcert_sample_ground_truth_only_names_fields_the_options_read():
    options = OCROptionsBIRTHCERT()
    readable = set(options.ru_fields) | set(options.en_fields)
    gt_files = sorted((REPO_ROOT / 'samples').glob('BIRTHCERT_*/*.json'))
    assert gt_files, 'no BIRTHCERT ground truth under samples/'
    for gt_file in gt_files:
        gt = json.loads(gt_file.read_text(encoding='utf-8'))
        assert set(gt) <= readable, f'{gt_file.name}: {sorted(set(gt) - readable)}'


def test_worded_date_joins_with_spaces():
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'Birth_date', 'BIRTHCERT', ['15', 'ОКТЯБРЯ', '2020', 'Г.'])
    assert ocr_dict['Birth_date'] == '15 ОКТЯБРЯ 2020 Г.'


def test_digit_date_keeps_its_dots():
    for doc_type in ('BIRTHCERT', 'DL', 'INTPASSPORT'):
        ocr_dict = {}
        Pipeline._join_field(ocr_dict, 'Birth_date', doc_type, ['22', '06', '2010'])
        assert ocr_dict['Birth_date'] == '22.06.2010', doc_type


def test_snils_dates_are_unaffected():
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'Birth_date', 'SNILS', ['26', 'СЕНТЯБРЯ', '1997'])
    assert ocr_dict['Birth_date'] == '26 СЕНТЯБРЯ 1997'


# ---------------------------------------------------------------------------
# DL / DLBACK & MIGRATIONCARD dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('doc_type', ['DLBACK', 'DLBACK_ALL'])
def test_dlback_does_not_get_front_side_options(doc_type):
    options = OCROptionsClass.make_options(doc_type)
    assert not isinstance(options, OCROptionsDL)
    assert type(options) is OCROptionsClass
    assert not options.ru_fields and not options.en_fields


@pytest.mark.parametrize('doc_type', ['DL', 'DL_2011', 'DL_2020'])
def test_dl_front_side_reaches_dl_options(doc_type):
    assert isinstance(OCROptionsClass.make_options(doc_type), OCROptionsDL)


@pytest.mark.parametrize('doc_type', ['MIGRATIONCARD', 'MIGRATIONCARD_ALL'])
def test_migration_card_make_options_accepts_full_doctype(doc_type):
    assert isinstance(OCROptionsClass.make_options(doc_type), OCROptionsMigrationCard)


# ---------------------------------------------------------------------------
# STS / STSBACK options & VIN correction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('doc_type', ['STS', 'STS_1996', 'STSBACK', 'STSBACK_1996',
                                      'STS_2019', 'STSBACK_2019'])
def test_sts_both_sides_reach_the_same_options(doc_type):
    assert isinstance(OCROptionsClass.make_options(doc_type), OCROptionsSTS)


@pytest.mark.parametrize('doc_type', ['INTPASSPORT_2011', 'INTPASSPORTADDR_ALL', 'EXTPASSPORT_2003',
                                      'DL_2011', 'SNILS_1996', 'BIRTHCERT_2018'])
def test_no_other_type_is_caught_by_the_sts_branch(doc_type):
    assert not isinstance(OCROptionsClass.make_options(doc_type), OCROptionsSTS)


@pytest.mark.parametrize('fields', [FIELDS_STS_FRONT, FIELDS_STS_BACK, FIELDS_STS_NEW_FORM, FIELDS_STS_OLD_2010],
                         ids=['front', 'back', 'new-form', 'old-2010'])
def test_sts_every_field_of_each_side_is_routed_to_an_engine(fields):
    options = OCROptionsSTS()
    missing = [f for f in fields if f not in options.ru_fields + options.en_fields]
    assert not missing, f'detected but never read: {missing}'


def test_sts_no_field_is_routed_to_both_engines():
    options = OCROptionsSTS()
    assert not (set(options.ru_fields) & set(options.en_fields))


def test_sts_reg_number_and_vin_are_latin():
    options = OCROptionsSTS()
    assert 'Reg_number' in options.en_fields
    assert 'VIN' in options.en_fields


def test_sts_series_keeps_the_passport_precedent():
    assert 'Licence_number' in OCROptionsSTS().ru_fields


@pytest.mark.parametrize('read,expected', [
    ('WFODXXGAJD1A00001', 'WF0DXXGAJD1A00001'),
    ('XTAOOOOOOOO123456', 'XTA00000000123456'),
    ('WF0DXXGAJD1A00001', 'WF0DXXGAJD1A00001'),
])
def test_vin_letter_o_becomes_zero(read, expected):
    assert check_vin(read) == expected


def test_vin_fix_leaves_other_fields_alone():
    fix = OCRLatin.fix_errors
    assert fix(None, 'VIN', 'WFODX') == 'WF0DX'
    assert fix(None, 'Last_name_en', 'SOKOLOV') == 'SOKOLOV'
    assert fix(None, 'Body_number', 'WFODX') == 'WFODX'


def test_sts_there_is_no_model_field():
    options = OCROptionsSTS()
    assert 'Vehicle_model_en' not in options.ru_fields + options.en_fields
    assert 'Vehicle_model_en' not in options.needed_split


# ---------------------------------------------------------------------------
# Passport MRZ wiring & Licence_number engine routing
# ---------------------------------------------------------------------------

def test_mrz_lines_are_joined_with_a_newline():
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'MRZ', 'EXTPASSPORT', [MRZ_LINE1, MRZ_LINE2])
    assert ocr_dict['MRZ'] == f'{MRZ_LINE1}\n{MRZ_LINE2}'


def test_mrz_join_does_not_insert_spaces():
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'MRZ', 'INTPASSPORT', [MRZ_LINE1, MRZ_LINE2])
    assert ' ' not in ocr_dict['MRZ']
    assert len(ocr_dict['MRZ'].splitlines()) == 2


def test_mrz_single_recognised_line_has_no_trailing_newline():
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'MRZ', 'EXTPASSPORT', [MRZ_LINE2, ''])
    assert ocr_dict['MRZ'] == MRZ_LINE2


def test_other_fields_keep_the_space_join():
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'Birth_place_ru', 'INTPASSPORT', ['ГОР', 'МОСКВА'])
    assert ocr_dict['Birth_place_ru'] == 'ГОР МОСКВА'


@pytest.mark.parametrize('options', [OCROptionsINTPassport, OCROptionsEXTPassport])
def test_passports_declare_mrz_latin_unsplit_and_licence_number_cyrillic(options):
    opts = options()
    assert 'MRZ' in opts.en_fields
    assert 'MRZ' not in opts.ru_fields
    assert 'MRZ' not in opts.needed_split
    assert 'Licence_number' in opts.ru_fields
    assert 'Licence_number' not in opts.en_fields


def test_documents_without_a_zone_do_not_declare_mrz():
    for doc_type in ('DL_2011', 'SNILS_1996', 'BIRTHCERT_1998', 'INTPASSPORTADDR_ALL'):
        options = OCROptionsClass.make_options(doc_type)
        assert 'MRZ' not in options.en_fields, doc_type
        assert 'MRZ' not in options.ru_fields, doc_type
