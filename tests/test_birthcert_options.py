"""OCR options and the date join for birth certificates (both blanks).

One options class serves BIRTHCERT_1998 and BIRTHCERT_2018: `make_options` is
given the type with the `_<year>` suffix already stripped, so it cannot tell the
two apart. Everything the 2018 blank added (the parents' birth dates, the place
of issue, the 21-digit act number) therefore has to live in the same lists as
the 1998 fields, and every date on both blanks has to survive a join rule that
was written for digit dates.
"""
import json
from pathlib import Path

import pytest

from document_processing.pipeline.pipeline import OCROptionsBIRTHCERT, OCROptionsClass, Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Every TextFields class the 2018 blank can produce (see the field map in
#: Melnikov/Text_fields/gen_birthcert_canvases.py).
FIELDS_2018 = ('Last_name_ru', 'First_name_ru', 'Birth_date', 'Birth_place_ru',
               'Issue_date', 'Issue_organization_ru', 'Issue_place_ru', 'Licence_number',
               'Father_last_name_ru', 'Father_first_middle_ru', 'Father_birth_date',
               'Mother_last_name_ru', 'Mother_first_middle_ru', 'Mother_birth_date',
               'Act_number')


@pytest.mark.parametrize('doc_type', ['BIRTHCERT', 'BIRTHCERT_1998', 'BIRTHCERT_2018'])
def test_every_era_reaches_the_same_options(doc_type):
    assert isinstance(OCROptionsClass.make_options(doc_type), OCROptionsBIRTHCERT)


def test_the_year_suffix_splits_off_cleanly():
    """process_img does this before make_options; a label without '_' would raise."""
    assert 'BIRTHCERT_2018'.rsplit('_', maxsplit=1) == ['BIRTHCERT', '2018']


def test_all_2018_fields_are_read():
    """A class the options do not list is detected and then silently dropped."""
    options = OCROptionsBIRTHCERT()
    missing = [f for f in FIELDS_2018 if f not in options.ru_fields + options.en_fields]
    assert not missing, f'not routed to any engine: {missing}'


def test_nothing_is_routed_to_the_latin_engine():
    """Both blanks spell their dates out in Cyrillic («15 октября 2020 г.»), and the
    Cyrillic engine reads the 1998 digit-only birth date just as well - the same
    precedent as the passport Licence_number (issue #12)."""
    assert OCROptionsBIRTHCERT().en_fields == []


def test_the_sample_ground_truth_only_names_fields_the_options_read():
    options = OCROptionsBIRTHCERT()
    readable = set(options.ru_fields) | set(options.en_fields)
    gt_files = sorted((REPO_ROOT / 'samples').glob('BIRTHCERT_*/*.json'))
    # Without this the loop over an empty samples/ passes silently.
    assert gt_files, 'no BIRTHCERT ground truth under samples/'
    for gt_file in gt_files:
        gt = json.loads(gt_file.read_text(encoding='utf-8'))
        assert set(gt) <= readable, f'{gt_file.name}: {sorted(set(gt) - readable)}'


def test_a_worded_date_joins_with_spaces():
    """The 2018 blank writes every date in words. Joining those with '.' gave
    «15.ОКТЯБРЯ.2020.Г.» - the 1998 blank only hid it because its ruler dots merged
    with the join dot into a run that _clean_ruler_artifacts wiped out."""
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'Birth_date', 'BIRTHCERT', ['15', 'ОКТЯБРЯ', '2020', 'Г.'])
    assert ocr_dict['Birth_date'] == '15 ОКТЯБРЯ 2020 Г.'


def test_a_digit_date_keeps_its_dots():
    """The 1998 blank's Birth_date is DD.MM.YYYY, and passports and licences are
    digit-only throughout - the separator follows the content, not the doc type."""
    for doc_type in ('BIRTHCERT', 'DL', 'INTPASSPORT'):
        ocr_dict = {}
        Pipeline._join_field(ocr_dict, 'Birth_date', doc_type, ['22', '06', '2010'])
        assert ocr_dict['Birth_date'] == '22.06.2010', doc_type


def test_snils_dates_are_unaffected():
    """SNILS was the original exception and keeps its explicit rule."""
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'Birth_date', 'SNILS', ['26', 'СЕНТЯБРЯ', '1997'])
    assert ocr_dict['Birth_date'] == '26 СЕНТЯБРЯ 1997'
