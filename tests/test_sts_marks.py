"""STS special marks: words torn by a line break, and the leasing record.

The tears below are the four found on real STS backs (2026-10-07); the leasing
texts follow the two editions the generator and the real samples print.
"""
from types import SimpleNamespace

import pytest

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
