"""Whole-line retries must report the geometry of the actual OCR patch."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from document_processing.geometry import Chain, Offset, Scale, corners
from document_processing.pipeline.pipeline import Pipeline, PipelineResults
from document_processing.pipeline.probe import RecordingStageSink


@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('rotated', [False, True])
def test_whole_line_fallback_keeps_raw_boxes_but_maps_the_read_patch(empty, rotated):
    p = object.__new__(Pipeline)
    p.results = PipelineResults()
    p.probe = RecordingStageSink()
    p.ocr_options = SimpleNamespace(ru_fields=['Licence_number'], en_fields=[],
                                   needed_split=['Licence_number'],
                                   needs_licence_rotation=rotated)
    line = np.random.default_rng(41).integers(0, 256, (20, 200, 3), dtype=np.uint8)
    patch = cv2.rotate(line, cv2.ROTATE_90_COUNTERCLOCKWISE) if rotated else line
    boxes = [] if empty else [[0, 0, 1, 5, .9], [3, 0, 4, 5, .8]]
    words = [patch[:5, int(b[0]):int(b[2])] for b in boxes]
    p.words_detector = SimpleNamespace(model_name='WordsDetector', predict_transform=lambda _: {
        'WordsDetector': {'bbox': boxes, 'warped_img': words}})
    p.results._meta_results['Geometry'] = Chain((Scale(.5, .5),))
    p.results._meta_results['FieldFrames'] = [(Offset(-10, -20), 200, 20)]
    p._split_words({'bbox': [[10, 20, 210, 40, .9, 0, 'Licence_number']],
                    'warped_img': [patch]}, 'INTPASSPORT')

    assert p.results.words_patches['Licence_number']['patches'][0] is patch
    assert len(p.results.word_quads['Licence_number']) == 1
    np.testing.assert_allclose(p.results.word_quads['Licence_number'][0],
                               corners(20, 40, 420, 80))
    # The detector's original boxes still diagnose why the fallback happened.
    assert p.probe.get('words.Licence_number.bbox') == [boxes]


def test_an_intact_split_still_maps_each_word():
    p = object.__new__(Pipeline)
    p.results, p.probe = PipelineResults(), None
    p.ocr_options = SimpleNamespace(ru_fields=['Field'], en_fields=[],
                                   needed_split=['Field'], needs_licence_rotation=False)
    line = np.full((20, 100, 3), 100, np.uint8)
    boxes = [[0, 0, 45, 20, .9], [55, 0, 100, 20, .8]]
    p.words_detector = SimpleNamespace(model_name='WordsDetector', predict_transform=lambda _: {
        'WordsDetector': {'bbox': boxes, 'warped_img': [line[:, :45], line[:, 55:]]}})
    p._split_words({'bbox': [[10, 20, 110, 40, .9, 0, 'Field']], 'warped_img': [line]}, 'OTHER')
    assert not p.results.words_fallback
    np.testing.assert_allclose(p.results.word_quads['Field'],
                               [corners(10, 20, 55, 40), corners(65, 20, 110, 40)])
