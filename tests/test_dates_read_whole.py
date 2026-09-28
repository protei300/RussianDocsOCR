"""A date field re-read whole when the word-by-word reading is not a date.

The word splitter can drop a word without leaving a hole the gap guard sees: on
a real 1998 birth certificate the issue date came out as month and year only -
the day was not taken for a word - while the same crop read whole gave the day
glued to the rest. The values below are made up («14» МАРТА 2019). The rule
replaces a reading only when it does NOT convert to dd.mm.yyyy and the
whole-line reading DOES.
"""
from types import SimpleNamespace

import numpy as np

from document_processing.pipeline.pipeline import Pipeline


class Engine:
    """Stand-in OCR engine: returns a fixed text for any crop, counts calls."""
    model_name = 'engine'

    def __init__(self, text):
        self.text, self.calls = text, 0

    def predict(self, _img):
        self.calls += 1
        return {self.model_name: {'ocr_output': self.text}}

    @staticmethod
    def fix_errors(field_type, text):
        return text


def run(ocr, date_lines, cyr_text='14МАРТА2019', lat_text='14.03.2019',
        ru_fields=('Issue_date',)):
    stub = SimpleNamespace(
        results=SimpleNamespace(_meta_results={'OCR': dict(ocr)}),
        ocr_options=SimpleNamespace(ru_fields=list(ru_fields)),
        ocr_cyr=Engine(cyr_text), ocr_lat=Engine(lat_text),
        _date_lines=date_lines,
    )
    Pipeline._reread_dates_whole(stub)
    return stub


LINE = np.zeros((22, 355, 3), np.uint8)


def test_a_date_that_lost_its_day_is_read_whole():
    s = run({'Issue_date': 'МАРТА 2019'}, {'Issue_date': [LINE]})
    assert s.results._meta_results['OCR']['Issue_date'] == '14МАРТА2019'
    assert s.results._meta_results['DatesReadWhole'] == [
        {'field': 'Issue_date', 'split': 'МАРТА 2019', 'whole': '14МАРТА2019'}]


def test_a_date_that_already_converts_is_never_re_read():
    s = run({'Issue_date': '28 ИЮЛЯ 2010'}, {'Issue_date': [LINE]})
    assert s.results._meta_results['OCR']['Issue_date'] == '28 ИЮЛЯ 2010'
    assert s.ocr_cyr.calls == 0 and 'DatesReadWhole' not in s.results._meta_results


def test_a_whole_reading_that_is_no_date_either_changes_nothing():
    s = run({'Issue_date': 'МАРТА 2019'}, {'Issue_date': [LINE]}, cyr_text='МАРТА2019')
    assert s.results._meta_results['OCR']['Issue_date'] == 'МАРТА 2019'
    assert 'DatesReadWhole' not in s.results._meta_results


def test_the_field_keeps_its_engine():
    # a date routed to Latin (not in ru_fields) is re-read by the Latin engine
    s = run({'Issue_date': '01.2011'}, {'Issue_date': [LINE]}, ru_fields=())
    assert s.ocr_lat.calls == 1 and s.ocr_cyr.calls == 0
    assert s.results._meta_results['OCR']['Issue_date'] == '14.03.2019'


def test_nothing_remembered_means_nothing_done():
    s = run({'Issue_date': 'МАРТА 2019'}, {})
    assert s.results._meta_results['OCR']['Issue_date'] == 'МАРТА 2019'
    assert s.ocr_cyr.calls == 0
