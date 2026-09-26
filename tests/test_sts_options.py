"""OCR options for the vehicle registration certificate (issue #17).

One options class serves both sides of the document (STS_<year> is the vehicle
side, STSBACK_<year> the owner side): `make_options` is given the type with the
year suffix stripped, and it dispatches on the substring 'sts', which both
share. Every field the field detector can produce for either side has to be
routed to an engine, or the pipeline detects it and silently drops it - the
failure mode that put this file's sibling (test_birthcert_options.py) here.
"""
import pytest

from document_processing.pipeline.pipeline import OCROptionsClass, OCROptionsSTS
from document_processing.pipeline_modules.ocr_corrections import check_vin

#: Every TextFields class the two sides can produce (the field map of the
#: synthetic-data generator the detector was trained with; not part of this tree).
FIELDS_FRONT = ('Reg_number', 'VIN', 'Vehicle_make_ru', 'Vehicle_make_en', 'Vehicle_type',
                'Vehicle_category', 'Vehicle_year', 'Chassis_number', 'Body_number',
                'Vehicle_color', 'Engine_power', 'Eco_class', 'Max_mass', 'Curb_mass',
                'Expiration_date', 'PTS_number', 'Licence_number')
FIELDS_BACK = ('Licence_number', 'Last_name_ru', 'Last_name_en', 'First_name_ru',
               'First_name_en', 'Middle_name_ru', 'Living_region_ru', 'House_number',
               'Apartment_number', 'Special_marks', 'Issue_organisation_code', 'Issue_date')
#: Added by the new form (order 267/2019).
FIELDS_NEW_FORM = ('Vehicle_model_en', 'Type_approval', 'Building_number')
#: Added by the original edition of the old form (order 1001/2008, ~2010).
FIELDS_OLD_2010 = ('Engine_model', 'Engine_number', 'Engine_volume',
                   'Issue_organization_ru', 'Issue_date', 'Building_number')


@pytest.mark.parametrize('doc_type', ['STS', 'STS_1996', 'STSBACK', 'STSBACK_1996',
                                      'STS_2019', 'STSBACK_2019'])
def test_both_sides_reach_the_same_options(doc_type):
    assert isinstance(OCROptionsClass.make_options(doc_type), OCROptionsSTS)


@pytest.mark.parametrize('doc_type', ['INTPASSPORT_2011', 'INTPASSPORTADDR_ALL', 'EXTPASSPORT_2003',
                                      'DL_2011', 'SNILS_1996', 'BIRTHCERT_2018'])
def test_no_other_type_is_caught_by_the_sts_branch(doc_type):
    """The dispatcher matches substrings in order; a new branch must not shadow
    an old type (the intpassportaddr/intpassport trap, documented on make_options)."""
    assert not isinstance(OCROptionsClass.make_options(doc_type), OCROptionsSTS)


def test_the_year_suffix_splits_off_cleanly():
    assert 'STSBACK_1996'.rsplit('_', maxsplit=1) == ['STSBACK', '1996']


@pytest.mark.parametrize('fields', [FIELDS_FRONT, FIELDS_BACK, FIELDS_NEW_FORM, FIELDS_OLD_2010],
                         ids=['front', 'back', 'new-form', 'old-2010'])
def test_every_field_of_each_side_is_routed_to_an_engine(fields):
    options = OCROptionsSTS()
    missing = [f for f in fields if f not in options.ru_fields + options.en_fields]
    assert not missing, f'detected but never read: {missing}'


def test_no_field_is_routed_to_both_engines():
    options = OCROptionsSTS()
    both = sorted(set(options.ru_fields) & set(options.en_fields))
    assert not both, both


def test_reg_number_and_vin_are_latin():
    """Decision of 2026-09-05: plate letters and VIN are Latin in the ground truth
    and on the output, so they must reach the Latin engine."""
    options = OCROptionsSTS()
    assert 'Reg_number' in options.en_fields
    assert 'VIN' in options.en_fields


def test_the_series_keeps_the_passport_precedent():
    """Digits read better on the Cyrillic engine (issue #12), and old blanks
    carry Cyrillic letters in the series («77 УХ»)."""
    assert 'Licence_number' in OCROptionsSTS().ru_fields


@pytest.mark.parametrize('read, expected', [
    ('WFODXXGAJD1A00001', 'WF0DXXGAJD1A00001'),   # the issue #17 case (a made-up VIN)
    ('XTAOOOOOOOO123456', 'XTA00000000123456'),
    ('WF0DXXGAJD1A00001', 'WF0DXXGAJD1A00001'),   # already right: untouched
])
def test_vin_letter_o_becomes_zero(read, expected):
    """A VIN never contains O (ISO 3779); the Latin engine reads 0 as O."""
    assert check_vin(read) == expected


def test_vin_fix_leaves_other_fields_alone():
    """Only VIN is rewritten: the O->0 map must not reach names or plates."""
    from document_processing.pipeline_modules.ocr_latin.ocr_latin import OCRLatin
    fix = OCRLatin.fix_errors  # does not touch the loaded model
    assert fix(None, 'VIN', 'WFODX') == 'WF0DX'
    assert fix(None, 'Last_name_en', 'SOKOLOV') == 'SOKOLOV'
    assert fix(None, 'Body_number', 'WFODX') == 'WFODX'
