"""Wiring of the MRZ field: how its two lines become one OCR value.

The MRZ is detected one box per line and read line by line, so the pipeline
has to assemble the field itself. The layout is fixed-offset - every check
digit sits at a known position in line 2 - which makes the join rule part of
the contract rather than cosmetics.
"""
from document_processing.pipeline.pipeline import (OCROptionsClass, OCROptionsEXTPassport,
                                                   OCROptionsINTPassport, Pipeline)

LINE1 = 'P<RUSPOPOVA<<ALENA<<<<<<<<<<<<<<<<<<<<<<<<<<'
LINE2 = '7515919230RUS8410102F2506090<<<<<<<<<<<<<<02'


def test_lines_are_joined_with_a_newline():
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'MRZ', 'EXTPASSPORT', [LINE1, LINE2])
    assert ocr_dict['MRZ'] == f'{LINE1}\n{LINE2}'


def test_join_does_not_insert_spaces_into_the_zone():
    """A space is outside the MRZ alphabet and would shift every check-digit
    offset in line 2, so the default word join must not apply here."""
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'MRZ', 'INTPASSPORT', [LINE1, LINE2])
    assert ' ' not in ocr_dict['MRZ']
    assert len(ocr_dict['MRZ'].splitlines()) == 2


def test_a_single_recognised_line_does_not_become_a_trailing_newline():
    """Canvases cropped through the zone yield one line; it must stay one line."""
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'MRZ', 'EXTPASSPORT', [LINE2, ''])
    assert ocr_dict['MRZ'] == LINE2


def test_other_fields_keep_the_space_join():
    ocr_dict = {}
    Pipeline._join_field(ocr_dict, 'Birth_place_ru', 'INTPASSPORT', ['ГОР', 'МОСКВА'])
    assert ocr_dict['Birth_place_ru'] == 'ГОР МОСКВА'


def test_passports_declare_mrz_on_the_latin_engine():
    for options in (OCROptionsINTPassport(), OCROptionsEXTPassport()):
        assert 'MRZ' in options.en_fields
        assert 'MRZ' not in options.ru_fields


def test_mrz_is_never_word_split():
    """Splitting the zone at its filler runs would destroy the fixed layout."""
    for options in (OCROptionsINTPassport(), OCROptionsEXTPassport()):
        assert 'MRZ' not in options.needed_split


def test_documents_without_a_zone_do_not_declare_mrz():
    for doc_type in ('DL_2011', 'SNILS_1996', 'BIRTHCERT_1998', 'INTPASSPORTADDR_ALL'):
        options = OCROptionsClass.make_options(doc_type)
        assert 'MRZ' not in options.en_fields, doc_type
        assert 'MRZ' not in options.ru_fields, doc_type
