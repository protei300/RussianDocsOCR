"""The post-processing on matrices built from strings (one-hot rows with blanks between characters):
no OCR model needed, the dictionaries are the ones shipped in service/ml/registration/assets."""
from __future__ import annotations

import numpy as np
import pytest

from ocr_postprocessing import CTCAlphabet, OCRPostprocessor, greedy
from ocr_postprocessing.postprocess import References, address_pieces

ABC = CTCAlphabet(' "(),-./:№0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')


def matrix(text: str, p: float = 0.9) -> np.ndarray:
    """A CTC matrix that reads ``text``: every character on its own row, a blank row after it."""
    n = len(ABC.alphabet) + 1
    rows = [np.eye(n)[0]]
    for ch in text:
        r = np.full(n, (1 - p) / (n - 1))
        r[ABC.index[ch]] = p
        rows += [r, np.eye(n)[0] * p + (1 - p) / n]
    return np.array(rows, np.float32)


@pytest.fixture(scope='module')
def pp():
    refs = References()
    if not refs.addressru.exists() or not refs.fms_csv.exists():
        pytest.skip('dictionaries are not installed')
    return OCRPostprocessor(ABC, refs)


def test_matrix_reads_back():
    assert greedy(matrix('Г. ИРКУТСК'), ABC) == 'Г. ИРКУТСК'


def test_same_fields_out(pp):
    fields = [{'label': 'date', 'probs': matrix('18 ОКТЯБРЯ 2007 Г.'), 'id': 7},
              {'label': 'house', 'probs': matrix('11-А')}]
    out = pp.process_crops(fields)
    assert [f['label'] for f in out['fields']] == ['date', 'house']
    assert out['fields'][0]['id'] == 7 and 'probs' not in out['fields'][0]
    assert out['fields'][0]['text'] == '18 ОКТЯБРЯ 2007' and out['fields'][0]['value'] == '18.10.2007'
    assert out['fields'][0]['raw'] == '18 ОКТЯБРЯ 2007 Г.'
    assert out['fields'][1]['text'] == '11-А'                  # the stamp's own form of the number stays
    assert out['stamp']['date'] == '18.10.2007'


def test_misread_city_fixed_by_dictionary(pp):
    m = matrix('Г. ИРКУОСК')
    row = 1 + 2 * 'Г. ИРКУОСК'.index('О')
    m[row, ABC.index['О']], m[row, ABC.index['Т']] = 0.5, 0.4     # the network hesitated between О and Т
    out = pp.process_crops([{'label': 'addr_line', 'probs': matrix('ИРКУТСКАЯ ОБЛ.')},
                      {'label': 'addr_line', 'probs': m}])
    assert out['fields'][1]['text'] == 'Г. ИРКУТСК' and out['fields'][1]['changed']
    assert out['fields'][0]['changed'] is False
    assert out['stamp']['locality'] == 'ИРКУТСК'


def test_confident_reading_kept(pp):
    out = pp.process_crops([{'label': 'addr_line', 'probs': matrix('Г. ИРКУОСК', 0.9999)}])  # sure of «О»
    assert out['fields'][0]['text'] == 'Г. ИРКУОСК'


def test_word_of_same_root_not_replaced(pp):
    out = pp.process_crops([{'label': 'addr_line', 'probs': matrix('САМАРСКАЯ ОБЛ., Г. ЧАПАЕВСК')},
                      {'label': 'addr_line', 'probs': matrix('УЛ. ЧАПАЕВА')}])
    assert out['fields'][1]['text'] == 'УЛ. ЧАПАЕВА'


def test_code_gives_region(pp):
    out = pp.process_crops([{'label': 'code', 'probs': matrix('380-050')},
                      {'label': 'addr_line', 'probs': matrix('Г. ИРКУТСК')}])
    assert out['stamp']['code'] == '380-050'
    assert out['stamp']['region'] == 'ИРКУТСКАЯ'


def test_address_pieces():
    assert address_pieces('Г. ЛИПЕЦК, УЛ. МИРА, Д. 5, КВ. 7') == [
        ('place', 'Г ЛИПЕЦК'), ('street', 'УЛ МИРА'), ('house', '5'), ('flat', '7')]
    assert address_pieces('Д. ИВАНОВКА')[0][0] == 'place'          # «Д.» before a name is a village


@pytest.fixture(scope='module')
def imp():
    from ocr_postprocessing.crop import CropImprover
    refs = References()
    if not refs.addressru.exists() or not refs.fms_csv.exists():
        pytest.skip('dictionaries are not installed')
    return CropImprover(ABC, refs)


def test_crop_alone(imp):
    """One crop, no stamp: the format and the matrix are enough."""
    r_date = imp.improve('date', matrix('18 09 2018 Г.'))
    assert r_date['value'] == '18.09.2018' and r_date['text'] == '18 09 2018'
    assert imp.improve('house', matrix('80 А'))['text'] == '80 А'          # the number's own form stays
    assert imp.improve('code', matrix('380-050'))['text'] == '380-050'
    m = matrix('ОТДЕЛ ПО РОПОСАМ МИГРАЦИИ')
    row = 1 + 2 * 'ОТДЕЛ ПО РОПОСАМ МИГРАЦИИ'.index('Р', 9)
    m[row, ABC.index['Р']], m[row, ABC.index['В']] = 0.5, 0.4
    r = imp.improve('issuer_line', m)
    assert r['text'] == 'ОТДЕЛ ПО ВОПРОСАМ МИГРАЦИИ' and r['changed']


def test_model_output_shape_accepted(pp, imp):
    """The OCR's output as it comes, [1, T, C], is taken as well as [T, C]."""
    m = matrix('Г. ИРКУТСК')
    assert pp.process_crops([{'label': 'addr_line', 'probs': m[None]}])['fields'][0]['raw'] == 'Г. ИРКУТСК'
    assert imp.improve('addr_line', m[None])['raw'] == 'Г. ИРКУТСК'
