"""Fast isolated unit tests for ocr_postprocessing (no external 15 MB dictionaries needed).

Uses a tiny synthetic AddressRU fixture (both directory and .zip) and a tiny FMS
CSV so every module in ``ocr_postprocessing`` is verified in < 1 s on any machine
or CI runner, even when production dictionaries are absent.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ocr_postprocessing import rules
from ocr_postprocessing.crop import CropImprover
from ocr_postprocessing.ctc import CTCAlphabet, MatrixJoiner, as_matrix, greedy, join
from ocr_postprocessing.decode import best_number, code_candidates, code_scores, street_choice
from ocr_postprocessing.dictionaries import AddressBook, DepartmentCodes, best as fuzzy_best
from ocr_postprocessing.postprocess import (
    OCRPostprocessor,
    References,
    _dedupe,
    _replace_words,
    address_pieces,
    date_init,
    date_text,
    strict_date,
    strip_date_year_suffix,
)
from ocr_postprocessing.textnorm import digits_fixed, fix_digits_in_number, norm
from ocr_postprocessing.wildctc import NEG, WildCTC, best_date

ABC = CTCAlphabet(' "(),-./:№0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')


def _mat(text: str, p: float = 0.92) -> np.ndarray:
    n = len(ABC.alphabet) + 1
    rows = [np.eye(n)[0]]
    for ch in text:
        r = np.full(n, (1 - p) / (n - 1))
        r[ABC.index[ch]] = p
        rows += [r, np.eye(n)[0] * p + (1 - p) / n]
    return np.array(rows, np.float32)


def _write_mini_refs(base: Path, *, as_zip: bool = False) -> References:
    """Tiny KLADR/AddressRU + FMS + common-streets bundle."""
    base.mkdir(parents=True, exist_ok=True)
    files = {
        'socrbase.json': {
            '105': 'ОБЛАСТЬ',
            '103': 'ГОРОД',
            '106': 'РЕСПУБЛИКА',
            '201': 'РАЙОН',
            '301': 'ГОРОД',
            '305': 'ПОСЕЛОК',
            '501': 'УЛИЦА',
        },
        '00.json': [
            ['38', 'ИРКУТСКАЯ', '105'],
            ['77', 'МОСКВА', '103'],
            ['07', 'КАБАРДИНО-БАЛКАРСКАЯ', '106'],
            ['63', 'САМАРСКАЯ', '105'],
        ],
        '38.json': [
            ['000001', 'ИРКУТСК', '301'],
            ['010', 'ИРКУТСКИЙ', '201'],
        ],
        '38010.json': [
            ['000001', 'ХОМУТОВО', '305'],
        ],
        '38000001.json': [
            ['0001', 'ЛЕНИНА', '501'],
            ['0002', 'БАЙКАЛЬСКАЯ', '501'],
        ],
        '77.json': [
            ['0001', 'ТВЕРСКАЯ', '501'],
            ['0002', 'АРБАТ', '501'],
        ],
        '63.json': [
            ['000001', 'ТОЛЬЯТТИ', '301'],
            ['000002', 'ЧАПАЕВСК', '301'],
        ],
        '63000002.json': [
            ['0001', 'ЧАПАЕВА', '501'],
        ],
        '07.json': [
            ['000001', 'НАЛЬЧИК', '301'],
        ],
    }

    if as_zip:
        addr_path = base / 'addressru.zip'
        with zipfile.ZipFile(addr_path, 'w') as zf:
            for name, obj in files.items():
                zf.writestr(name, json.dumps(obj, ensure_ascii=False))
    else:
        addr_path = base / 'addressru'
        addr_path.mkdir(parents=True, exist_ok=True)
        for name, obj in files.items():
            (addr_path / name).write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')

    fms_path = base / 'fms.csv'
    fms_path.write_text(
        'code,name\n'
        '380-050,ОТДЕЛ ПО ВОПРОСАМ МИГРАЦИИ В КИРОВСКОМ РАЙОНЕ Г. ИРКУТСК\n'
        '380-051,ОТДЕЛЕНИЕ ПО РАЙОНУ СВЕРДЛОВСКИЙ Г. ИРКУТСК\n'
        '630-010,ТП УФМС В Г. ТОЛЬЯТТИ\n'
        '770-001,ОТДЕЛЕНИЕ ПО РАЙОНУ АРБАТ\n',
        encoding='utf-8',
    )
    streets_path = base / 'streets.json'
    streets_path.write_text(json.dumps(['ЛЕНИНА', 'МИРА', 'СОВЕТСКАЯ'], ensure_ascii=False), encoding='utf-8')
    return References(addressru=addr_path, fms_csv=fms_path, street_names=streets_path)


@pytest.fixture
def mini_refs(tmp_path):
    return _write_mini_refs(tmp_path, as_zip=True)


@pytest.fixture
def mini_dicts(mini_refs):
    return rules.Dicts(AddressBook(mini_refs.addressru), DepartmentCodes(mini_refs.fms_csv))


# ---------------------------------------------------------------------------
# textnorm & ctc
# ---------------------------------------------------------------------------

def test_textnorm_upper_lat2cyr_punct_and_digits():
    assert norm('  ул. «Ленина» — 12a  ') == 'УЛ. "ЛЕНИНА" - 12А'
    assert norm('Ёлка') == 'ЕЛКА'
    assert norm('') == '' and norm(None) == ''
    assert fix_digits_in_number('200Б') == '2006'
    assert fix_digits_in_number('З80') == '380'
    assert fix_digits_in_number('ОБЛАСТЬ') == 'ОБЛАСТЬ'
    assert digits_fixed('200Б Г. ДОМ О8') == '2006 Г. ДОМ 08'


def test_ctc_alphabet_as_matrix_greedy_and_join(tmp_path):
    cfg = tmp_path / 'model.json'
    cfg.write_text(json.dumps({'Outputs': [{'Alphabet': 'АБ '}]}), encoding='utf-8')
    abc = CTCAlphabet.from_model_json(tmp_path)
    assert abc.alphabet == 'АБ ' and abc.index['А'] == 1

    m1 = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]], np.float32)
    assert greedy(m1, abc) == 'АБ'
    assert greedy(m1[None], abc) == 'АБ'
    assert as_matrix(m1[None]).shape == (4, 4)

    joiner = MatrixJoiner(abc)
    assert joiner.join([m1]).shape == m1.shape
    joined = join([m1, m1], abc)
    assert greedy(joined, abc) == 'АБ АБ'


# ---------------------------------------------------------------------------
# wildctc
# ---------------------------------------------------------------------------

def test_wildctc_score_matches_score_many_and_handles_oov_and_lookalike():
    m = _mat('18 ОКТЯБРЯ 2007')
    wc = WildCTC(m, ABC.index, gamma=0.1)
    s_single = wc.score(['18', 'ОКТЯБРЯ', '2007'])
    wc2 = WildCTC(m, ABC.index, gamma=0.1)
    s_batch = wc2.score_many([['18', 'ОКТЯБРЯ', '2007'], ['19', 'ЯНВАРЯ', '2001'], ['?']])
    assert s_single == pytest.approx(s_batch[0], rel=1e-5)
    assert s_batch[0] > s_batch[1]
    assert s_batch[2] == NEG

    # Lookalike 'О' in matrix contributes to digit '0'
    m_letter_o = _mat('2ОО7')
    w_look = WildCTC(m_letter_o, ABC.index, gamma=0.1)
    assert w_look.score(['2007']) > w_look.score(['2117'])


def test_best_date_recovers_date_from_noisy_worded_and_multi_crop_matrices():
    m = _mat('"08" АВГУСТА 2015 Г.')
    got, score = best_date(m, ABC.index, init=(8, 8, 2015))
    assert got == '08.08.2015' and score > NEG

    # Multi-crop scoring sums both matrices
    m1 = _mat('15 03 2019')
    m2 = _mat('15 МАРТА 2019')
    got2, _ = best_date([m1, m2], ABC.index, init=(15, 3, 2019))
    assert got2 == '15.03.2019'


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

def test_decode_best_number_code_candidates_and_street_choice():
    assert best_number([], ABC.index, '10') is None
    val, _ = best_number([_mat('ДОМ 17-А/2')], ABC.index, '17А/2')
    assert val == '17А/2'

    codes = ['380-050', '380-051', '380-999', '770-001']
    # Hamming <= 2 matches 380-050 & 380-051, not 380-999 (3 diffs) or 770-001
    assert set(code_candidates('380-052', codes)) == {'380-050', '380-051'}
    # 5-digit dropped-digit recovery
    assert '380-050' in code_candidates('380-50', codes)

    cs = code_scores('380-050', [_mat('380-050')], ABC.index, codes)
    assert cs['read'] == '380-050' and cs['scores']['380-050'] > cs['scores']['380-051']

    joiner = MatrixJoiner(ABC)
    m_st = _mat('БАЙКАЛЬСКЯ')
    picked = street_choice('БАЙКАЛЬСКЯ', [[m_st]], joiner, ['БАЙКАЛЬСКАЯ', 'ЛЕНИНА'], ['МИРА'], n_pads=1)
    assert picked == 'БАЙКАЛЬСКАЯ'
    # A numbered street reading must not collapse onto an unnumbered dictionary name
    assert street_choice('НАГОРНАЯ 2', [[_mat('НАГОРНАЯ 2')]], joiner, ['НАГОРНАЯ'], [], n_pads=1) is None


# ---------------------------------------------------------------------------
# dictionaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('as_zip', [False, True])
def test_address_book_and_department_codes(tmp_path, as_zip):
    refs = _write_mini_refs(tmp_path / ('zip' if as_zip else 'dir'), as_zip=as_zip)
    book = AddressBook(refs.addressru)
    assert book.available
    assert len(book.regions()) == 4
    irk = book.region('38')
    assert irk is not None and irk.name == 'ИРКУТСКАЯ' and irk.abbr == 'ОБЛ'

    locs = book.localities(irk)
    assert {p.name for p in locs} == {'ИРКУТСК', 'ХОМУТОВО'}
    city = next(p for p in locs if p.name == 'ИРКУТСК')
    assert {p.name for p in book.streets(city)} == {'ЛЕНИНА', 'БАЙКАЛЬСКАЯ'}

    # Federal city is its own locality and appears in cities()
    msk = book.region('77')
    assert any(p.name == 'МОСКВА' for p in book.localities(msk))
    assert any(p.name == 'МОСКВА' for p in book.cities())
    assert book.region_in_text('УФМС ПО ИРКУТСКОЙ ОБЛАСТИ').key == '38'
    assert book.region_in_text('НИЧЕГО НЕ НАЙДЕНО') is None

    # Preload writes and reloads gzipped index
    cache = tmp_path / 'cache'
    book.preload(cache_dir=cache, key=rules._key)
    assert '38' in book.loc_keys
    book2 = AddressBook(refs.addressru)
    book2.preload(cache_dir=cache, key=rules._key)
    assert book2.loc_keys['38'] == book.loc_keys['38']

    # Empty book fails gracefully
    empty = AddressBook(None)
    assert not empty.available and empty.regions() == ()

    # Department codes & fuzzy_best
    dc = DepartmentCodes(refs.fms_csv)
    assert '380-050' in dc and '999-999' not in dc
    assert DepartmentCodes.region_of('380-050') == '38'
    assert DepartmentCodes.region_of(None) is None
    assert fuzzy_best('ИРКУТСК', locs)[0][0].name == 'ИРКУТСК'


# ---------------------------------------------------------------------------
# rules (including regression test for _nom_adj bugfix!)
# ---------------------------------------------------------------------------

def test_rules_primitives_and_nom_adj_bugfix(mini_dicts):
    assert rules.tokens('Г.ИРКУТСК, УЛ. ЛЕНИНА') == ['Г', 'ИРКУТСК', 'УЛ', 'ЛЕНИНА']
    assert rules.stem('САМАРЫ') == rules.stem('САМАРА')
    assert rules.num_value('ДОМ № 15-А / 2') == '15А/2'
    assert rules.canon('code', 'З80-050') == '380050'
    assert rules.canon('house', 'Д. 7-А') == '7А'
    assert rules.canon('house', 'Д. 7-Б') == '76'   # lookalike Б -> 6 when digits >= letters
    assert rules.canon_region('КБР', mini_dicts) == rules._key('КАБАРДИНО-БАЛКАРСКАЯ')

    # Multi-type single box + second house box acting as flat
    assert rules.numbers({'house': [{'text': 'ДОМ 7 КВ. 12'}]}) == {'house': '7', 'flat': '12'}
    assert rules.numbers({'house': [{'text': '5'}, {'text': '18'}]}) == {'house': '5', 'flat': '18'}

    # Regression guard for _LOC_ADJ backreference bug (used to return 'КИРОВ ИЙ')
    assert rules._nom_adj('КИРОВСКОМ') == 'КИРОВСКИЙ'
    assert rules._nom_adj('ТРОИЦКОМ') == 'ТРОИЦКИЙ'
    assert rules._nom_adj('СЕВЕРНОМ') == 'СЕВЕРНЫЙ'
    assert rules._nom_adj('ОКТЯБРЬСКОМ') == 'ОКТЯБРЬСКИЙ'


def test_rules_place_fields_fms_fill_and_stamp_fields(mini_dicts):
    # Federal city is both region and locality
    pl_msk = rules.place_fields({'place': [{'text': 'Г. МОСКВА, АРБАТ'}]}, None, mini_dicts)
    assert pl_msk['region'] == 'МОСКВА' and pl_msk['locality'] == 'МОСКВА'

    # Region inferred from code + district filled from FMS unit name («В КИРОВСКОМ РАЙОНЕ» -> КИРОВСКИЙ)
    reads = {
        'place': [{'text': 'Г. ИРКУТСК'}],
        'street': [{'text': 'УЛ. ЛЕНИНА'}],
        'house': [{'text': '10'}],
        'code': [{'text': '380-050'}],
    }
    res = rules.stamp_fields(reads, mini_dicts, date='01.02.2020')
    assert res['region'] == 'ИРКУТСКАЯ'
    assert res['locality'] == 'ИРКУТСК'
    assert res['district'] == 'КИРОВСКИЙ'
    assert res['street'] == 'ЛЕНИНА'
    assert res['house'] == '10'
    assert res['code'] == '380-050'

    # FMS fills city when locality is missing («ТП УФМС В Г. ТОЛЬЯТТИ»)
    res_tlt = rules.stamp_fields({'place': [{'text': 'САМАРСКАЯ ОБЛ.'}], 'code': [{'text': '630-010'}]},
                                 mini_dicts)
    assert res_tlt['locality'] == 'ТОЛЬЯТТИ'

    # Wrong-region FMS code must NOT inject another region's district
    out_mismatch = {'code': '380-050'}
    rules.fms_fill(out_mismatch, {'_region': mini_dicts.region_by_key2('77')}, mini_dicts)
    assert 'district' not in out_mismatch


# ---------------------------------------------------------------------------
# postprocess & crop end-to-end on synthetic mini dictionary
# ---------------------------------------------------------------------------

def test_postprocess_helpers_and_dedupe():
    assert strip_date_year_suffix('18.09.2018 Г.') == '18.09.2018'
    assert strict_date('18 09 2018 Г.') == '18.09.2018'
    assert strict_date('31 02 2018') is None
    assert strict_date('18 09 2099') is None
    assert date_init('18 ОКТЯБРЯ 07') == (18, 10, 2007)
    assert date_text('18.10.2007', [_mat('18 ОКТЯБРЯ 2007 Г.')], ABC.index) == '18 ОКТЯБРЯ 2007'

    assert address_pieces('Г. ИРКУТСК, УЛ. ЛЕНИНА, Д. 5, КВ. 7') == [
        ('place', 'Г ИРКУТСК'), ('street', 'УЛ ЛЕНИНА'), ('house', '5'), ('flat', '7'),
    ]
    assert _replace_words('Г. ИРКУОСК', ['ИРКУТСК']) == 'Г. ИРКУТСК'
    # Word cut at edge (substring) or >2 edits away is left untouched
    assert _replace_words('ИРКУТ', ['ИРКУТСК']) == 'ИРКУТ'

    q1 = np.array([[0, 0], [100, 0], [100, 20], [0, 20]], np.float32)
    q2 = np.array([[2, 1], [102, 1], [102, 21], [2, 21]], np.float32)
    q3 = np.array([[0, 40], [100, 40], [100, 60], [0, 60]], np.float32)
    assert [i for i, _ in _dedupe([(0, q1), (1, q2), (2, q3)])] == [0, 2]


def test_postprocessor_and_crop_improver_on_mini_refs(mini_refs):
    pp = OCRPostprocessor(ABC, refs=mini_refs, preload=False)
    m_city = _mat('Г. ИРКУОСК')
    row = 1 + 2 * 'Г. ИРКУОСК'.index('О')
    m_city[row, ABC.index['О']], m_city[row, ABC.index['Т']] = 0.5, 0.4

    out = pp.process_crops([
        {'label': 'date', 'probs': _mat('18 ОКТЯБРЯ 2007 Г.')},
        {'label': 'code', 'probs': _mat('380-050')},
        {'label': 'addr_line', 'probs': m_city},
        {'label': 'addr_line', 'probs': _mat('УЛ. ЛЕНИНА, Д. 11-А')},
        {'label': 'house', 'probs': _mat('11-А')},
    ])
    assert out['fields'][0]['text'] == '18 ОКТЯБРЯ 2007'
    assert out['fields'][2]['text'] == 'Г. ИРКУТСК' and out['fields'][2]['changed'] is True
    assert out['stamp']['locality'] == 'ИРКУТСК'
    assert out['stamp']['region'] == 'ИРКУТСКАЯ'
    assert out['stamp']['street'] == 'ЛЕНИНА'
    assert out['stamp']['district'] == 'КИРОВСКИЙ'

    # process_boxes path
    q = np.array([[0, 0], [80, 0], [80, 20], [0, 20]], np.float32)
    out_b = pp.process_boxes([
        {'label': 'date', 'probs': _mat('18.10.2007'), 'quad': q},
        {'label': 'place', 'probs': _mat('Г. ИРКУТСК'), 'quad': q + [0, 25]},
        {'label': 'street', 'probs': _mat('УЛ. ЛЕНИНА'), 'quad': q + [0, 50]},
        {'label': 'house', 'probs': _mat('10'), 'quad': q + [0, 75]},
        {'label': 'code', 'probs': _mat('380-050'), 'quad': q + [0, 100]},
    ])
    assert out_b['stamp']['date'] == '18.10.2007'
    assert out_b['stamp']['locality'] == 'ИРКУТСК'
    assert out_b['stamp']['street'] == 'ЛЕНИНА'

    # CropImprover standalone
    imp = CropImprover(ABC, refs=mini_refs)
    assert imp.improve('date', _mat('18 09 2018 Г.'))['value'] == '18.09.2018'
    assert imp.improve('house', _mat('80 А'))['text'] == '80 А'
    assert imp.improve('code', _mat('380-050'))['text'] == '380-050'
