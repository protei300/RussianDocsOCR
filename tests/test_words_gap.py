# -*- coding: utf-8 -*-
"""The gap guard on the word split: read the line whole when the split lost a word.

Two failure modes were measured on real documents and both are the same defect
seen from different sides: a 9 px crop where the detector returned NO words (the
field disappeared without a trace), and a line whose LONGEST word was the one
missed. Both leave the same fingerprint - an empty stretch on the line about as
wide as a word - so one signal covers both.

The signal is a ratio to the line's own median word width, not a share of the
line, and that is the whole point: an internal passport's Licence_number is three
digit groups with wide gaps and is read CORRECTLY. Measured, not assumed - the
share-of-the-line variant put those fields at 0.64-0.75, in the same range as
genuinely damaged lines, while by the gap they sit at a median of 0.92 against a
threshold of 3.0.

The tests are built so that a guard which does nothing fails, and so does a guard
which fires everywhere:

* POSITIVE - lines that really lost a word (nothing found, a word-wide hole in
  the middle, a word missing from the edge) must fall back;
* NEGATIVE - an evenly spaced line, and SNILS at any gap, must NOT;
* and the FLAG must mean what it says: the accuracy measurement uses it to decide
  which lines get the missing-space correction, so a flag raised without an
  actual whole-line read would let any difference in under an amnesty.
"""
import types

import numpy as np
import pytest

from document_processing.pipeline.pipeline import Pipeline, PipelineResults


# ---------------------------------------------------------------------------
# The signal itself - a pure function, testable without a model
# ---------------------------------------------------------------------------
def _b(x1, x2):
    """A word box; only the x-interval matters to the signal."""
    return [x1, 0, x2, 10, 0.9, 0, 'word']


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
    """No boxes means no denominator - and the line is entirely missing.

    This is the measured case where a field used to vanish silently, so it must
    read as "worst possible", not as zero.
    """
    assert Pipeline._widest_gap([], 100) == float('inf')
    assert Pipeline._widest_gap(None, 100) == float('inf')


def test_wide_but_even_spacing_is_not_a_hole():
    """The reason the signal is a ratio and not a share of the line.

    Both lines below leave the same total emptiness; only the second one is
    missing something.
    """
    even = [_b(0, 10), _b(30, 40), _b(60, 70), _b(90, 100)]
    holed = [_b(0, 10), _b(12, 22), _b(24, 34), _b(90, 100)]
    assert Pipeline._widest_gap(even, 100) < Pipeline._widest_gap(holed, 100)


# ---------------------------------------------------------------------------
# The guard inside the split
# ---------------------------------------------------------------------------
class FakeWords:
    """Stands in for WordsDetector: returns whatever the test prescribes."""

    model_name = 'WordsDetector'

    def __init__(self, boxes, words):
        self._boxes, self._words = boxes, words

    def predict_transform(self, patch):
        return {self.model_name: {'warped_img': self._words, 'bbox': self._boxes}}


def _patch(width=100):
    """A line crop carrying stroke-like detail.

    Not a blank rectangle on purpose: since the ink check landed, a crop with no
    strokes is deliberately NOT re-read (see test_words_ink.py), so a blank patch
    would make these tests measure that check instead of this one.
    """
    patch = np.full((10, width, 3), 200, dtype=np.uint8)
    patch[:, ::5] = 30
    return patch


def _split(doc_type, boxes, words, field='Birth_place_ru', width=100,
           threshold=Pipeline.WORDS_MAX_GAP):
    """Run ``_split_words`` with a fake detector and no models loaded."""
    line = _patch(width)
    fake = types.SimpleNamespace(
        words_detector=FakeWords(boxes, words),
        ocr_options=types.SimpleNamespace(en_fields=[], ru_fields=[field],
                                          needed_split=[field]),
        results=PipelineResults(),  # the real object: the double must not outlive the contract
        WORDS_MAX_GAP=threshold,
        LINE_MIN_INK=Pipeline.LINE_MIN_INK,
        _widest_gap=Pipeline._widest_gap,
        _line_ink=Pipeline._line_ink,
        _duplicate_field_indices=lambda *a, **k: set(),
        _emit=lambda *a, **k: None,
    )
    text_fields = {'bbox': [[0, 0, width, 10, 0.9, 0, field]], 'patches': [line]}
    result = Pipeline._split_words(fake, text_fields, doc_type)
    return result, fake.results.meta_results.get('WordsFallback') or [], line


def test_line_with_no_words_found_is_read_whole():
    """The measured case that lost a field silently: zero words on a 9 px crop."""
    result, flagged, line = _split('INTPASSPORT', boxes=[], words=[])
    patches = result['Birth_place_ru']['patches']
    assert len(patches) == 1 and patches[0] is line
    assert [f['field'] for f in flagged] == ['Birth_place_ru']
    # None, not a number: with no boxes there is no word width to divide by, and
    # a made-up figure here would look like a measurement.
    assert flagged[0]['gap'] is None


def test_line_whose_longest_word_was_missed_is_read_whole():
    """«Тракторозаводский район»: the long word is exactly the one dropped."""
    result, flagged, line = _split(
        'INTPASSPORT', boxes=[_b(0, 10), _b(12, 22), _b(90, 100)],
        words=[_patch(10), _patch(10), _patch(10)])
    assert result['Birth_place_ru']['patches'] == [line]
    assert flagged and flagged[0]['gap'] > Pipeline.WORDS_MAX_GAP


def test_evenly_spaced_number_is_left_split():
    """Licence_number of an internal passport: wide gaps, read correctly.

    The case the previous signal (share of the line covered) could not tell from
    real damage - measured at 0.64-0.75 there, against damaged lines at 0.45-0.62.
    """
    words = [_patch(25), _patch(25), _patch(20)]
    result, flagged, _ = _split(
        'INTPASSPORT', boxes=[_b(0, 25), _b(40, 65), _b(80, 100)], words=words,
        field='Licence_number')
    assert result['Licence_number']['patches'] == words
    assert flagged == []


def test_healthy_line_is_left_split():
    """The guard must be silent where the split did its job.

    Without this the whole change would look like a success while quietly
    reading every document as one glued line.
    """
    words = [_patch(45), _patch(45)]
    result, flagged, _ = _split(
        'INTPASSPORT', boxes=[_b(0, 45), _b(55, 100)], words=words)
    assert result['Birth_place_ru']['patches'] == words
    assert flagged == []


def test_snils_never_falls_back_even_with_the_whole_line_missing():
    """A ban by construction, not by hoping the threshold spares it.

    SNILS picks the OCR engine by word-index parity - its dates read «31 октября
    1998», so odd-indexed words go to the Cyrillic engine. A line read whole has
    one word and no parity left, so the routing would silently change engines.
    """
    words = [_patch(5)]
    result, flagged, _ = _split(
        'SNILS', boxes=[_b(0, 5)], words=words, field='Birth_date')
    assert result['Birth_date']['patches'] == words
    assert flagged == []


def test_flag_means_the_line_was_actually_read_whole():
    """The flag is a contract, and this is its negative control.

    The accuracy measurement corrects for missing spaces ONLY on flagged lines.
    A flag raised without an actual whole-line read would turn that correction
    into a blanket amnesty, so flag and effect must be inseparable: every flagged
    field holds exactly one patch, and an unflagged one holds what the detector
    returned.
    """
    flagged_result, flagged, line = _split('INTPASSPORT', boxes=[], words=[])
    assert len(flagged_result['Birth_place_ru']['patches']) == 1
    assert flagged_result['Birth_place_ru']['patches'][0] is line

    words = [_patch(45), _patch(45)]
    clean_result, clean_flags, _ = _split(
        'INTPASSPORT', boxes=[_b(0, 45), _b(55, 100)], words=words)
    assert clean_flags == []
    assert len(clean_result['Birth_place_ru']['patches']) == 2


def test_threshold_decides_and_nothing_else():
    """Same line, two thresholds: the guard is driven by the declared number.

    Guards the value from drifting into "always" or "never" through some other
    condition - if this ever passes with both thresholds giving the same answer,
    something else is deciding.
    """
    boxes = [_b(0, 20), _b(60, 100)]
    words = [_patch(20), _patch(40)]
    _, strict, _ = _split('INTPASSPORT', boxes=boxes, words=words, threshold=0.5)
    _, lenient, _ = _split('INTPASSPORT', boxes=boxes, words=words, threshold=9.0)
    assert [f['field'] for f in strict] == ['Birth_place_ru']
    assert lenient == []
