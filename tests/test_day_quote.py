"""The closing quote after the day of a worded date, read as a comma or a dot.

The 1998 birth certificate prints «28» ИЮЛЯ 2010 г.; a field box with a margin
around the letters takes in », and the Cyrillic engine reads it as '28,'.
"""
import pytest

from document_processing.pipeline_modules.ocr_corrections import strip_day_quote
from document_processing.pipeline_modules.ocr_cyrillic.ocr_cyrillic import OCRCyrillic


@pytest.mark.parametrize('word, expected', [
    ('28,', '28'), ('28.', '28'), ('5,', '5'),
    ('28', '28'), ('ИЮЛЯ', 'ИЮЛЯ'), ('2010', '2010'), ('Г.', 'Г.'),
    ('22.06.2010', '22.06.2010'),      # a digit date is one word: untouched
    ('2010,', '2010,'),                # not a day: four digits
    ('28,,', '28,,'), (',', ','), ('', ''),
])
def test_only_a_day_with_one_mark_loses_it(word, expected):
    assert strip_day_quote(word) == expected


def test_the_engine_applies_it_to_date_fields_only():
    fix = OCRCyrillic.fix_errors  # does not touch the loaded model
    assert fix(None, 'Issue_date', '28,') == '28'
    assert fix(None, 'Birth_date', '22.06.2010') == '22.06.2010'
    assert fix(None, 'Last_name_ru', '28,') == '28,'
