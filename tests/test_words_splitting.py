# -*- coding: utf-8 -*-
"""Word splitting guards: gap fallback, ink check, and fallback quad geometry.

Covers:
* Gap guard (``Pipeline._widest_gap`` and whole-line fallback in ``_split_words``):
  positive cases (zero boxes, missed long word), negative cases (evenly spaced
  numbers, healthy lines, SNILS ban), and threshold control.
* Ink check (``Pipeline._line_ink``): blank/dark strips stay empty when no boxes
  are found, inked lines still fall back, boxes-present lines skip the check,
  known heavy-noise boundary, and threshold control via ``monkeypatch``.
* Fallback geometry: whole-line retries map the actual read patch (including
  rotated Licence_number crops) back to input space while keeping raw detector
  boxes in the diagnostic probe.
"""
from __future__ import annotations

import types

import cv2
import numpy as np
import pytest

from document_processing.geometry import Chain, Offset, Scale, corners
from document_processing.pipeline.pipeline import Pipeline, PipelineResults
from document_processing.pipeline.probe import RecordingStageSink


# ---------------------------------------------------------------------------
# Shared synthetic helpers
# ---------------------------------------------------------------------------

def _b(x1: int, x2: int) -> list:
    """A word box; only the x-interval matters to the gap signal."""
    return [x1, 0, x2, 10, 0.9, 0, 'word']


def _flat(width: int = 200, height: int = 20, value: int = 200) -> np.ndarray:
    """A blank strip: paper and nothing else."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _inked(width: int = 200, height: int = 20, step: int = 7) -> np.ndarray:
    """A strip carrying stroke-like detail."""
    patch = _flat(width, height)
    patch[:, ::step] = 30
    return patch


def _noisy(sigma: float, width: int = 200, height: int = 20, seed: int = 0) -> np.ndarray:
    """A blank strip with sensor noise of a given strength."""
    rs = np.random.RandomState(seed)
    base = _flat(width, height).astype(np.float32)
    return np.clip(base + rs.normal(0, sigma, base.shape), 0, 255).astype(np.uint8)


class FakeWords:
    """Stands in for WordsDetector: returns whatever the test prescribes."""

    model_name = 'WordsDetector'

    def __init__(self, boxes, words):
        self._boxes, self._words = boxes, words

    def predict_transform(self, patch):
        return {self.model_name: {'warped_img': self._words, 'bbox': self._boxes}}


def _split(doc_type: str, boxes, words, *, patch: np.ndarray | None = None,
           field: str = 'Birth_place_ru', width: int = 100,
           threshold: float = Pipeline.WORDS_MAX_GAP):
    """Run ``Pipeline._split_words`` with a fake detector and no models loaded."""
    line = _inked(width=width, height=10, step=5) if patch is None else patch
    fake = types.SimpleNamespace(
        words_detector=FakeWords(boxes, words),
        ocr_options=types.SimpleNamespace(en_fields=[], ru_fields=[field],
                                          needed_split=[field]),
        results=PipelineResults(),
        WORDS_MAX_GAP=threshold,
        LINE_MIN_INK=Pipeline.LINE_MIN_INK,
        _widest_gap=Pipeline._widest_gap,
        _line_ink=Pipeline._line_ink,
        _duplicate_field_indices=lambda *a, **k: set(),
        _paired_duplicate_indices=lambda *a, **k: set(),
        _emit=lambda *a, **k: None,
        _field_quads=lambda *a, **k: None,
    )
    h, w = line.shape[:2]
    text_fields = {'bbox': [[0, 0, w, h, 0.9, 0, field]], 'patches': [line]}
    result = Pipeline._split_words(fake, text_fields, doc_type)
    meta = fake.results.meta_results
    return result, meta.get('WordsFallback') or [], meta.get('WordsNoInk') or [], line


# ---------------------------------------------------------------------------
# Gap signal & fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('boxes,width,expected', [
    ([_b(0, 100)], 100, 0.0),                       # one word fills the line
    ([_b(0, 45), _b(55, 100)], 100, 10 / 45),       # a normal inter-word space
    ([_b(0, 20), _b(80, 100)], 100, 3.0),           # a hole three words wide
    ([_b(0, 25), _b(40, 65), _b(80, 100)], 100, 0.6),   # evenly spaced - a number
    ([_b(40, 60)], 100, 2.0),                       # words missing from both edges
    ([_b(0, 20)], 100, 4.0),                        # everything after the first lost
    ([_b(80, 100)], 100, 4.0),                      # everything BEFORE the last lost
])
def test_gap_is_measured_in_typical_word_widths(boxes, width, expected):
    assert Pipeline._widest_gap(boxes, width) == pytest.approx(expected)


def test_nothing_found_is_one_whole_hole():
    assert Pipeline._widest_gap([], 100) == float('inf')
    assert Pipeline._widest_gap(None, 100) == float('inf')


def test_wide_but_even_spacing_is_not_a_hole():
    even = [_b(0, 10), _b(30, 40), _b(60, 70), _b(90, 100)]
    holed = [_b(0, 10), _b(12, 22), _b(24, 34), _b(90, 100)]
    assert Pipeline._widest_gap(even, 100) < Pipeline._widest_gap(holed, 100)


def test_line_with_no_words_found_is_read_whole():
    """Zero words on an inked crop -> whole-line fallback, gap=None, no refusal."""
    result, flagged, refused, line = _split('INTPASSPORT', boxes=[], words=[])
    patches = result['Birth_place_ru']['patches']
    assert len(patches) == 1 and patches[0] is line
    assert [f['field'] for f in flagged] == ['Birth_place_ru']
    assert flagged[0]['gap'] is None
    assert refused == []


def test_line_whose_longest_word_was_missed_is_read_whole():
    result, flagged, _, line = _split(
        'INTPASSPORT', boxes=[_b(0, 10), _b(12, 22), _b(90, 100)],
        words=[_inked(10, 10, 5), _inked(10, 10, 5), _inked(10, 10, 5)],
    )
    assert result['Birth_place_ru']['patches'] == [line]
    assert flagged and flagged[0]['gap'] > Pipeline.WORDS_MAX_GAP


def test_evenly_spaced_number_is_left_split():
    words = [_inked(25, 10, 5), _inked(25, 10, 5), _inked(20, 10, 5)]
    result, flagged, _, _ = _split(
        'INTPASSPORT', boxes=[_b(0, 25), _b(40, 65), _b(80, 100)], words=words,
        field='Licence_number',
    )
    assert result['Licence_number']['patches'] == words
    assert flagged == []


def test_healthy_line_is_left_split():
    words = [_inked(45, 10, 5), _inked(45, 10, 5)]
    result, flagged, _, _ = _split(
        'INTPASSPORT', boxes=[_b(0, 45), _b(55, 100)], words=words,
    )
    assert result['Birth_place_ru']['patches'] == words
    assert flagged == []


def test_snils_never_falls_back_even_with_the_whole_line_missing():
    words = [_inked(5, 10, 5)]
    result, flagged, refused, _ = _split(
        'SNILS', boxes=[_b(0, 5)], words=words, field='Birth_date',
    )
    assert result['Birth_date']['patches'] == words
    assert flagged == [] and refused == []


def test_gap_threshold_decides_and_nothing_else():
    boxes = [_b(0, 20), _b(60, 100)]
    words = [_inked(20, 10, 5), _inked(40, 10, 5)]
    _, strict, _, _ = _split('INTPASSPORT', boxes=boxes, words=words, threshold=0.5)
    _, lenient, _, _ = _split('INTPASSPORT', boxes=boxes, words=words, threshold=9.0)
    assert [f['field'] for f in strict] == ['Birth_place_ru']
    assert lenient == []


# ---------------------------------------------------------------------------
# Ink check
# ---------------------------------------------------------------------------

def test_blank_strip_carries_no_ink():
    assert Pipeline._line_ink(_flat()) == 0.0


def test_strokes_are_orders_of_magnitude_above_the_threshold():
    assert Pipeline._line_ink(_inked()) > 100 * Pipeline.LINE_MIN_INK


def test_missing_patch_is_not_a_crash():
    assert Pipeline._line_ink(None) == 0.0
    assert Pipeline._line_ink(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def test_measure_reads_strokes_not_darkness():
    assert Pipeline._line_ink(_flat(value=40)) < Pipeline.LINE_MIN_INK


def test_known_limit_noise_boundary():
    """Ordinary noise (sigma=2) stays below threshold; heavy noise (sigma=5) crosses it."""
    assert Pipeline._line_ink(_noisy(2)) < Pipeline.LINE_MIN_INK
    assert Pipeline._line_ink(_noisy(5)) > Pipeline.LINE_MIN_INK


def test_blank_line_is_left_alone_and_stays_empty():
    result, fallback, refused, _ = _split(
        'INTPASSPORT', boxes=[], words=[], patch=_flat(), field='First_name_ru',
    )
    assert result['First_name_ru']['patches'] == []
    assert fallback == []
    assert [r['field'] for r in refused] == ['First_name_ru']


def test_line_with_strokes_is_still_re_read():
    patch = _inked()
    result, fallback, refused, _ = _split(
        'INTPASSPORT', boxes=[], words=[], patch=patch, field='First_name_ru',
    )
    assert result['First_name_ru']['patches'] == [patch]
    assert [f['field'] for f in fallback] == ['First_name_ru']
    assert refused == []


def test_ink_check_is_not_asked_where_boxes_were_found():
    words = [_flat(20), _flat(20)]
    boxes = [[0, 0, 20, 10, 0.9, 0, 'w'], [180, 0, 200, 10, 0.9, 0, 'w']]
    _, fallback, refused, _ = _split(
        'INTPASSPORT', boxes=boxes, words=words, patch=_flat(), field='First_name_ru',
    )
    assert refused == []
    assert [f['field'] for f in fallback] == ['First_name_ru']


def test_snils_is_untouched_by_ink_check_too():
    _, fallback, refused, _ = _split(
        'SNILS', boxes=[], words=[], patch=_flat(), field='Birth_date',
    )
    assert fallback == [] and refused == []


@pytest.mark.parametrize('threshold,expect_refusal', [(1e9, True), (0.0, False)])
def test_ink_threshold_decides_and_nothing_else(monkeypatch, threshold, expect_refusal):
    monkeypatch.setattr(Pipeline, 'LINE_MIN_INK', threshold)
    _, fallback, refused, _ = _split(
        'INTPASSPORT', boxes=[], words=[], patch=_inked(), field='First_name_ru',
    )
    assert bool(refused) is expect_refusal
    assert bool(fallback) is not expect_refusal


# ---------------------------------------------------------------------------
# Fallback quad geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('rotated', [False, True])
def test_whole_line_fallback_keeps_raw_boxes_but_maps_the_read_patch(empty, rotated):
    p = object.__new__(Pipeline)
    p.results = PipelineResults()
    p.probe = RecordingStageSink()
    p.ocr_options = types.SimpleNamespace(
        ru_fields=['Licence_number'], en_fields=[],
        needed_split=['Licence_number'], needs_licence_rotation=rotated,
    )
    line = np.random.default_rng(41).integers(0, 256, (20, 200, 3), dtype=np.uint8)
    patch = cv2.rotate(line, cv2.ROTATE_90_COUNTERCLOCKWISE) if rotated else line
    boxes = [] if empty else [[0, 0, 1, 5, .9], [3, 0, 4, 5, .8]]
    words = [patch[:5, int(b[0]):int(b[2])] for b in boxes]
    p.words_detector = types.SimpleNamespace(
        model_name='WordsDetector',
        predict_transform=lambda _: {'WordsDetector': {'bbox': boxes, 'warped_img': words}},
    )
    p.results._meta_results['Geometry'] = Chain((Scale(.5, .5),))
    p.results._meta_results['FieldFrames'] = [(Offset(-10, -20), 200, 20)]
    p._split_words(
        {'bbox': [[10, 20, 210, 40, .9, 0, 'Licence_number']], 'warped_img': [patch]},
        'INTPASSPORT',
    )

    assert p.results.words_patches['Licence_number']['patches'][0] is patch
    assert len(p.results.word_quads['Licence_number']) == 1
    np.testing.assert_allclose(p.results.word_quads['Licence_number'][0], corners(20, 40, 420, 80))
    assert p.probe.get('words.Licence_number.bbox') == [boxes]


def test_an_intact_split_still_maps_each_word():
    p = object.__new__(Pipeline)
    p.results, p.probe = PipelineResults(), None
    p.ocr_options = types.SimpleNamespace(
        ru_fields=['Field'], en_fields=[],
        needed_split=['Field'], needs_licence_rotation=False,
    )
    line = np.full((20, 100, 3), 100, np.uint8)
    boxes = [[0, 0, 45, 20, .9], [55, 0, 100, 20, .8]]
    p.words_detector = types.SimpleNamespace(
        model_name='WordsDetector',
        predict_transform=lambda _: {'WordsDetector': {'bbox': boxes, 'warped_img': [line[:, :45], line[:, 55:]]}},
    )
    p._split_words({'bbox': [[10, 20, 110, 40, .9, 0, 'Field']], 'warped_img': [line]}, 'OTHER')
    assert not p.results.words_fallback
    np.testing.assert_allclose(
        p.results.word_quads['Field'],
        [corners(10, 20, 55, 40), corners(65, 20, 110, 40)],
    )
