# -*- coding: utf-8 -*-
"""The ink check: do not re-read a line that carries no strokes.

The gap guard treats "no word boxes" as "the split lost the text". That
observation has a second cause - there IS no text - and on a real web sample the
difference mattered: a name line anonymised at the source into a blurred strip
was re-read whole and came back as «ЛВН». An honest empty field had turned into
something that looks like an answer, which for an identifier is worse than
nothing.

So the question is asked in exactly one place: where the detector found NO boxes.
Where boxes were found the text is there by definition, and asking would be noise.

The tests pull in both directions on purpose, mirroring the two acceptance
criteria that constrain this change from opposite sides:

* a line with strokes must still be re-read (the guard's repairs must survive);
* a line without strokes must be left exactly as it was before the guard existed.

If the measure or the threshold ever drifts, exactly one of those breaks - and
which one says in which direction it drifted.
"""
import types

import numpy as np
import pytest

from document_processing.pipeline.pipeline import Pipeline, PipelineResults


def _flat(width=200, height=20, value=200):
    """A blank strip: paper and nothing else."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def _inked(width=200, height=20):
    """A strip carrying stroke-like detail."""
    patch = _flat(width, height)
    patch[:, ::7] = 30
    return patch


def _noisy(sigma, width=200, height=20, seed=0):
    """A blank strip with sensor noise of a given strength."""
    rs = np.random.RandomState(seed)
    base = _flat(width, height).astype(np.float32)
    return np.clip(base + rs.normal(0, sigma, base.shape), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# The measure
# ---------------------------------------------------------------------------
def test_blank_strip_carries_no_ink():
    assert Pipeline._line_ink(_flat()) == 0.0


def test_strokes_are_orders_of_magnitude_above_the_threshold():
    assert Pipeline._line_ink(_inked()) > 100 * Pipeline.LINE_MIN_INK


def test_missing_patch_is_not_a_crash():
    assert Pipeline._line_ink(None) == 0.0
    assert Pipeline._line_ink(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0


def test_measure_reads_strokes_not_darkness():
    """A uniformly DARK strip has no strokes, and must not pass as inked.

    This is the whole reason the measure is not "spread of brightness" or "share
    of dark pixels": both were measured on real crops and overlap between blurred
    strips and strips that do carry text.
    """
    dark = _flat(value=40)
    assert Pipeline._line_ink(dark) < Pipeline.LINE_MIN_INK


def test_sensor_noise_alone_stays_below_the_threshold():
    """Ordinary noise must not be mistaken for text.

    The companion limit is documented in the next test rather than hidden: past
    a certain noise level it IS mistaken for text.
    """
    assert Pipeline._line_ink(_noisy(2)) < Pipeline.LINE_MIN_INK


def test_known_limit_heavy_noise_reads_as_ink():
    """A named boundary, not an accident: strong noise defeats this measure.

    Measured on synthetic strips: the threshold is crossed around sigma 3.4 of
    per-pixel noise. A blank strip on a grainy photo therefore still reads as
    "has strokes", the line gets re-read, and the defect this check exists to
    prevent can reappear. The check is a floor, not a proof, and this test is
    here so the limit is visible in the suite instead of being rediscovered.
    """
    assert Pipeline._line_ink(_noisy(2)) < Pipeline.LINE_MIN_INK
    assert Pipeline._line_ink(_noisy(5)) > Pipeline.LINE_MIN_INK


# ---------------------------------------------------------------------------
# The check inside the split
# ---------------------------------------------------------------------------
class FakeWords:
    model_name = 'WordsDetector'

    def __init__(self, boxes, words):
        self._boxes, self._words = boxes, words

    def predict_transform(self, patch):
        return {self.model_name: {'warped_img': self._words, 'bbox': self._boxes}}


def _split(patch, boxes, words, doc_type='INTPASSPORT', field='First_name_ru'):
    fake = types.SimpleNamespace(
        words_detector=FakeWords(boxes, words),
        ocr_options=types.SimpleNamespace(en_fields=[], ru_fields=[field],
                                          needed_split=[field]),
        results=PipelineResults(),  # the real object: the double must not outlive the contract
        WORDS_MAX_GAP=Pipeline.WORDS_MAX_GAP,
        LINE_MIN_INK=Pipeline.LINE_MIN_INK,
        _widest_gap=Pipeline._widest_gap,
        _line_ink=Pipeline._line_ink,
        _duplicate_field_indices=lambda *a, **k: set(),
        _emit=lambda *a, **k: None,
    )
    text_fields = {'bbox': [[0, 0, patch.shape[1], patch.shape[0], 0.9, 0, field]],
                   'patches': [patch]}
    result = Pipeline._split_words(fake, text_fields, doc_type)
    meta = fake.results.meta_results
    return (result, meta.get('WordsFallback') or [], meta.get('WordsNoInk') or [])


def test_blank_line_is_left_alone_and_stays_empty():
    """The measured defect: «ЛВН» manufactured from a blurred strip.

    Refusing must restore the behaviour from BEFORE the guard existed - no
    boxes, no patches, an empty field - not some new kind of emptiness.
    """
    result, fallback, refused = _split(_flat(), boxes=[], words=[])
    assert result['First_name_ru']['patches'] == []
    assert fallback == []
    assert [r['field'] for r in refused] == ['First_name_ru']


def test_line_with_strokes_is_still_re_read():
    """The guard's repairs must survive the protection meant to bound them."""
    patch = _inked()
    result, fallback, refused = _split(patch, boxes=[], words=[])
    assert result['First_name_ru']['patches'] == [patch]
    assert [f['field'] for f in fallback] == ['First_name_ru']
    assert refused == []


def test_check_is_not_asked_where_boxes_were_found():
    """Only the no-boxes case is ambiguous; elsewhere the text is there.

    A blank-looking crop whose detector DID return boxes must still go through
    the ordinary path - if this ever fails, the check has spread beyond the one
    place it was declared to act in.
    """
    words = [_flat(20), _flat(20)]
    boxes = [[0, 0, 20, 10, 0.9, 0, 'w'], [180, 0, 200, 10, 0.9, 0, 'w']]
    _, fallback, refused = _split(_flat(), boxes=boxes, words=words)
    assert refused == []                       # not asked
    assert [f['field'] for f in fallback] == ['First_name_ru']   # gap still fires


def test_refusal_and_fallback_are_separate_facts():
    """"Never looked" and "looked and said no" must not collapse into one.

    A consumer that cannot tell them apart cannot tell a healthy document from
    one whose field was suppressed.
    """
    _, fallback_inked, refused_inked = _split(_inked(), boxes=[], words=[])
    _, fallback_blank, refused_blank = _split(_flat(), boxes=[], words=[])
    assert fallback_inked and not refused_inked
    assert refused_blank and not fallback_blank


def test_snils_is_untouched_by_the_check_too():
    """SNILS never falls back at all, so it can never be refused either."""
    result, fallback, refused = _split(_flat(), boxes=[], words=[],
                                       doc_type='SNILS', field='Birth_date')
    assert fallback == [] and refused == []


@pytest.mark.parametrize('threshold,expect_refusal', [(1e9, True), (0.0, False)])
def test_threshold_decides_and_nothing_else(threshold, expect_refusal):
    """Same blank-ish line, two thresholds: the declared number is what decides."""
    patch = _inked()
    fake_threshold = Pipeline.LINE_MIN_INK
    try:
        Pipeline.LINE_MIN_INK = threshold
        _, fallback, refused = _split(patch, boxes=[], words=[])
        assert bool(refused) is expect_refusal
        assert bool(fallback) is not expect_refusal
    finally:
        Pipeline.LINE_MIN_INK = fake_threshold
