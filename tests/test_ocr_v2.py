# -*- coding: utf-8 -*-
"""Tests for the v2 OCR engines (OCRLatin / OCRCyrillic) and their support code:
greedy CTC decode + per-step alphabet masking, the alphabet/mask resolver, and
the v2 preprocessing shape. Patches live in ``images/OCRv2/`` (fed as RGB, as the
pipeline does)."""
import numpy as np
import pytest
from PIL import Image

from document_processing.pipeline_modules import OCRLatin, OCRCyrillic
from document_processing.pipeline_modules.ocr_latin import OCRLatin as OCRLatinCls
from document_processing.config.alphabets import allowed_charset, default_country
from document_processing.processing.preprocessing import OCRv2Preprocessing
from document_processing.processing.postprocessing import OCRProbsPostprocessing

PATCHES = 'images/OCRv2'

# (file, ground truth). Diacritics are substituted to the RUS/USA mask:
# STRÎMBANU -> STRIMBANU, ЮЛІЯ -> ЮЛИЯ.
LATIN_CASES = [
    ('latin_banja.jpg', 'BANJA'),
    ('latin_date.jpg', '03/15/2026'),
    ('latin_diacritic.jpg', 'STRIMBANU'),
]
CYRILLIC_CASES = [
    ('cyrillic_nikolay.png', 'НИКОЛАЙ'),
    ('cyrillic_fms.png', 'ФМС77718'),
    ('cyrillic_julia.png', 'ЮЛИЯ'),
]


def _rgb(name):
    return np.array(Image.open(f'{PATCHES}/{name}').convert('RGB'))


@pytest.mark.parametrize('tier', ['accurate', 'fast'])
class TestOCRv2Engines:
    def test_latin(self, tier):
        eng = OCRLatin(tier=tier, device='cpu')
        for name, gt in LATIN_CASES:
            pred = eng.predict(_rgb(name))[eng.model_name]['ocr_output']
            assert pred == gt, f'latin/{tier} {name}: {pred!r} != {gt!r}'

    def test_cyrillic(self, tier):
        eng = OCRCyrillic(tier=tier, device='cpu')
        for name, gt in CYRILLIC_CASES:
            pred = eng.predict(_rgb(name))[eng.model_name]['ocr_output']
            assert pred == gt, f'cyrillic/{tier} {name}: {pred!r} != {gt!r}'


class TestAlphabetResolver:
    def test_defaults(self):
        assert default_country('cyrillic') == 'RUS'
        assert default_country('latin') == 'USA'

    def test_masks_include_digits_and_letters(self):
        cyr = allowed_charset('cyrillic')  # default RUS
        assert 'А' in cyr and 'Я' in cyr and '5' in cyr and '.' in cyr
        assert 'A' not in cyr  # no Latin letters in the Cyrillic mask
        lat = allowed_charset('latin')  # default USA
        assert 'A' in lat and 'Z' in lat and '5' in lat
        assert 'А' not in lat  # no Cyrillic in the Latin mask

    def test_extensible_country(self):
        ukr = allowed_charset('cyrillic', 'UKR')
        assert 'Ї' in ukr and 'Є' in ukr


class TestOCRProbsDecode:
    def test_greedy_collapse_and_blank(self):
        # alphabet 'AB', blank=0. Sequence: A A blank B -> "AB".
        post = OCRProbsPostprocessing(alphabet='AB', allowed=None, blank_index=0)
        probs = np.array([[[0.1, 0.8, 0.1],   # A
                           [0.1, 0.8, 0.1],   # A (repeat, collapsed)
                           [0.8, 0.1, 0.1],   # blank
                           [0.1, 0.1, 0.8]]]) # B
        assert post(probs) == 'AB'

    def test_masking_substitutes_not_drops(self):
        # alphabet index1='A'(allowed), index2='Ä'(disallowed). Model is confident
        # on Ä; masking must substitute -> 'A', not drop it.
        post = OCRProbsPostprocessing(alphabet='AÄ', allowed={'A'}, blank_index=0)
        probs = np.array([[[0.01, 0.02, 0.97]]])  # argmax = Ä (disallowed)
        assert post(probs) == 'A'


class TestOCRv2Preprocessing:
    def test_shape_and_dtype(self):
        pre = OCRv2Preprocessing(height=32)
        img = np.zeros((64, 300, 3), dtype=np.uint8)
        out = pre(img)
        assert out.shape[0] == 1 and out.shape[1] == 32 and out.shape[3] == 3
        assert out.shape[2] == round(300 * 32 / 64)  # dynamic width, no padding
        assert out.dtype == np.uint8

    def test_bgr_flip(self):
        # a pure-red RGB image should become red-in-BGR (channel 2), height 32
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[..., 0] = 255  # R
        out = OCRv2Preprocessing(height=32, color_order='BGR')(img)
        assert out[0, :, :, 2].max() == 255 and out[0, :, :, 0].max() == 0
