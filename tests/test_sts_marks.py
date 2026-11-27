"""STS special marks: words torn by a line break, and the leasing record.

The tears below are the four found on real STS backs (2026-10-07); the leasing
texts follow the two editions the generator and the real samples print.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from document_processing.geometry import Offset
from document_processing.pipeline.pipeline import OCROptionsClass, OCROptionsSTS, Pipeline
from document_processing.pipeline.sts_marks import glue_torn_words, parse_leasing


@pytest.mark.parametrize('lines,glued', [
    ([['ПО', 'ДЛ', 'ЛИЗИ'], ['НГА', 'ОТ']], ['ПО', 'ДЛ', 'ЛИЗИНГА', 'ОТ']),
    ([['ЛИЗИН'], ['ГОДАТЕЛЬ', 'АО']], ['ЛИЗИНГОДАТЕЛЬ', 'АО']),
    ([['ЛИЗИНГОВАЯ', 'КОМ'], ['ПАНИЯ']], ['ЛИЗИНГОВАЯ', 'КОМПАНИЯ']),
    ([['ГАЗПРОМБАНК', 'АВТОЛ'], ['ИЗИНГ']], ['ГАЗПРОМБАНК', 'АВТОЛИЗИНГ']),
])
def test_a_known_word_torn_by_the_line_break_is_glued(lines, glued):
    assert glue_torn_words(lines) == glued


@pytest.mark.parametrize('lines', [
    [['ЛИЗИНГ'], ['ДОГОВОР']],              # two whole words
    [['ЛИЗИНГОДАТЕЛЬ'], ['АО', 'ВТБ']],     # a whole word, then a name
    [['ООО', 'КАРКА'], ['ПЕТРОВ']],          # joined is not a known word
    [['№АЛ2682'], ['26/01-26']],            # numbers are never glued
])
def test_words_that_are_not_a_torn_known_word_stay_apart(lines):
    flat = [w for line in lines for w in line]
    assert glue_torn_words(lines) == flat


def test_empty_lines_are_skipped():
    assert glue_torn_words([[], ['ЛИЗИ'], [''], ['НГ']]) == ['ЛИЗИНГ']


def test_the_full_record_with_the_lessor_on_the_next_line():
    got = parse_leasing('ПО ДЛ №АЛ268226/01-26 ОТ 12.03.2024 ЛИЗИНГОДАТЕЛЬ АО ВТБ ЛИЗИНГ')
    assert got == {'leasing': True, 'role': 'lessor_named', 'lessor': 'АО ВТБ ЛИЗИНГ',
                   'contract_number': 'АЛ268226/01-26', 'contract_date': '12.03.2024',
                   'contract_date_normalized': '12.03.2024',
                   'until': None, 'until_normalized': None}


def test_real_abbreviations():
    """Forms seen on real backs (2026-10-07), values invented."""
    got = parse_leasing('ЛИЗИНГОПОЛУЧАТЕЛЬ ИВАНОВ ИВАН ЛИЗИНГОДАТЕЛЬ АО ЛИЗИНГОВАЯ КОМПАНИЯ '
                        'КАМАЗ ЛИЗИНГ ВРЕМ. УЧЕТ ДО 01.01.2027')
    assert got['role'] == 'lessor_named'
    assert got['lessor'] == 'АО ЛИЗИНГОВАЯ КОМПАНИЯ КАМАЗ ЛИЗИНГ'
    got = parse_leasing('ЛИЗИНГ ДО 31.12.2027. ДОГ ЛИЗ АХ ЭЛ/УЛН-123456/ДЛ ООО ЭЛЕМЕНТ')
    assert got['until_normalized'] == '31.12.2027'
    assert got['contract_number'] == 'АХ ЭЛ/УЛН-123456/ДЛ'
    assert got['contract_date'] is None                # «ДО» is the term, not the contract
    got = parse_leasing('ЛИЗИНГОДАТЕЛЬ ООО АВТОЛИЗИНГ, ДЕЙСТВИТЕЛЬНО ДО 01.02.2026')
    assert got['lessor'] == 'ООО АВТОЛИЗИНГ'


def test_the_lessor_ends_where_the_contract_begins():
    got = parse_leasing('В ЛИЗИНГЕ ЛИЗИНГОДАТЕЛЬ ООО "КАРКАДЕ" ДОГОВОР ЛИЗИНГА №45120 ОТ 05.11.2016')
    assert got['lessor'] == 'ООО "КАРКАДЕ"'
    assert got['contract_number'] == '45120' and got['contract_date_normalized'] == '05.11.2016'


def test_the_owner_as_lessee_names_no_lessor():
    got = parse_leasing('ЛИЗИНГОПОЛУЧАТЕЛЬ ДОГОВОР №1234-Л ОТ 03.04.2015')
    assert got['role'] == 'lessee' and got['lessor'] is None
    assert got['contract_number'] == '1234-Л'


def test_the_2010_short_form():
    got = parse_leasing('77 1234 Л.Д 14.02.2013')
    assert got['leasing'] and got['contract_date_normalized'] == '14.02.2013'
    assert got['role'] is None and got['lessor'] is None


@pytest.mark.parametrize('text', ['ДУБЛИКАТ', 'СМЕНА СОБСТВЕННИКА', '', None,
                                  'ЛИЗИ НГ'])   # a tear left unglued is not found
def test_no_leasing_record(text):
    assert parse_leasing(text) is None


def test_the_pipeline_glues_only_the_fields_the_options_name():
    me = SimpleNamespace(ocr_options=OCROptionsSTS,
                         _field_lines={'Special_marks': [2, 2], 'Last_name_ru': [1, 1]})
    assert Pipeline._glue_torn(me, 'Special_marks', ['ПО', 'ЛИЗИ', 'НГА', 'ОТ']) == \
        ['ПО', 'ЛИЗИНГА', 'ОТ']
    assert Pipeline._glue_torn(me, 'Last_name_ru', ['ЛИЗИ', 'НГА']) == ['ЛИЗИ', 'НГА']
    # line lengths that do not add up to the words: left alone
    assert Pipeline._glue_torn(me, 'Special_marks', ['ЛИЗИ', 'НГА']) == ['ЛИЗИ', 'НГА']
    assert OCROptionsClass.glue_torn == []


def test_only_the_flag_reaches_the_results():
    """The parts are parsed but not reported until the reading can carry them."""
    meta = {'OCR': {'Special_marks': 'ПО ДЛ №АЛ1/01-26 ОТ 12.03.2024 ЛИЗИНГОДАТЕЛЬ АО ВТБ ЛИЗИНГ'}}
    emitted = {}
    me = SimpleNamespace(results=SimpleNamespace(_meta_results=meta),
                         LEASING_REPORTED=Pipeline.LEASING_REPORTED,
                         _emit=lambda name, payload: emitted.__setitem__(name, payload))
    Pipeline._read_leasing(me, 'STSBACK')
    assert meta['Leasing'] == {'leasing': True}
    assert emitted == {'leasing': {'leasing': True}}, 'the conformance stage carries it'
    emitted.clear()
    meta3 = {'OCR': {'Special_marks': 'ДУБЛИКАТ'}}
    Pipeline._read_leasing(SimpleNamespace(results=SimpleNamespace(_meta_results=meta3),
                                           LEASING_REPORTED=Pipeline.LEASING_REPORTED,
                                           _emit=lambda n, p: emitted.__setitem__(n, p)), 'STSBACK')
    assert emitted == {'leasing': None}, 'an STS without leasing still emits the stage, as null'
    meta2 = {'OCR': {'Special_marks': 'ЛИЗИНГ'}}
    Pipeline._read_leasing(SimpleNamespace(results=SimpleNamespace(_meta_results=meta2),
                                           LEASING_REPORTED=Pipeline.LEASING_REPORTED,
                                           _emit=lambda n, p: emitted.__setitem__(n, p)), 'DL')
    assert 'Leasing' not in meta2, 'only an STS carries special marks'


# ---------------------------------------------------------------------------
# STS reading margins & year-based make engine (merged from test_sts_reading)
# ---------------------------------------------------------------------------

def _canvas():
    img = np.zeros((200, 300, 3), np.uint8)
    img[:, :, 0] = np.arange(200)[:, None]
    return img


def _run_margins(boxes, margin):
    img = _canvas()
    fields = {'bbox': [list(b) for b in boxes],
              'warped_img': [img[b[1]:b[3], b[0]:b[2]] for b in boxes]}
    frames = [Offset(-float(b[0]), -float(b[1])) for b in boxes]
    me = SimpleNamespace(ocr_options=SimpleNamespace(read_margin=margin))
    Pipeline._read_margins(me, fields, frames, img)
    return fields, frames


def test_the_read_crop_grows_and_the_box_does_not():
    box = [20, 100, 280, 120, 0.9, 0, 'Special_marks']
    fields, frames = _run_margins([box], {'Special_marks': 0.25})
    patch = fields['warped_img'][0]
    assert fields['bbox'][0] == box
    assert patch.shape[0] == 30
    assert patch[0, 0, 0] == 95 and patch[-1, 0, 0] == 124
    assert frames[0].to_input(np.float32([[0, 0]]))[0, 1] == 95


def test_the_margin_stops_halfway_to_the_next_line():
    upper = [20, 80, 280, 96, 0.9, 0, 'Special_marks']
    lower = [20, 100, 280, 116, 0.9, 0, 'Special_marks']
    fields, _ = _run_margins([upper, lower], {'Special_marks': 0.5})
    rows_upper = fields['warped_img'][0][:, 0, 0]
    rows_lower = fields['warped_img'][1][:, 0, 0]
    assert rows_upper.max() < rows_lower.min(), 'the two crops overlap'
    assert rows_upper.max() == 97 and rows_lower.min() == 98


def test_a_box_beside_it_does_not_limit_the_margin():
    marks = [20, 100, 140, 120, 0.9, 0, 'Special_marks']
    beside = [160, 80, 280, 98, 0.9, 0, 'Reg_number']
    fields, _ = _run_margins([marks, beside], {'Special_marks': 0.25})
    assert fields['warped_img'][0].shape[0] == 30


def test_other_fields_and_other_types_are_untouched():
    box = [20, 100, 280, 120, 0.9, 0, 'Vehicle_color']
    fields, _ = _run_margins([box], {'Special_marks': 0.25})
    assert fields['warped_img'][0].shape[0] == 20
    assert OCROptionsClass.read_margin == {} and OCROptionsClass.engine_by_year == {}


def test_the_make_engine_follows_the_sts_form():
    assert 'Vehicle_make_ru' in OCROptionsSTS.ru_fields
    assert set(OCROptionsSTS.read_margin) == {'Special_marks'}
    engine = lambda year, field='Vehicle_make_ru': Pipeline._engine_by_year(
        SimpleNamespace(ocr_options=OCROptionsSTS, _doc_year=year), field)
    assert engine('2019') == 'lat', 'the new form prints the make in Latin'
    assert engine('1996') is None, 'the old form keeps the Cyrillic route'
    assert engine('2019', 'Vehicle_color') is None
