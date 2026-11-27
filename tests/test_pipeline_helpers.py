"""Pure unit tests for pipeline helpers, deduplication, preprocessing, and stage edge cases.

No ONNX weights or service dependencies are loaded here.
"""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import document_processing
from document_processing import config as dp_config
from document_processing.pipeline.documents import DocumentsStage
from document_processing.pipeline.mrz_reading import MrzReadingStage
from document_processing.pipeline.pipeline import Pipeline
from document_processing.pipeline.results import PipelineResults
from document_processing.pipeline.stamps import StampsStage
from document_processing.pipeline_modules.ocr_batch import _WidthToTime
from document_processing.pipeline_modules.ocr_corrections import check_en_sex, check_rus_sex
from document_processing.pipeline_modules.textfields_detector.textfields_detector import (
    TextFieldsDetector,
)
from document_processing.processing.postprocessing import YOLOSegmentorPostprocessing
from document_processing.processing.preprocessing import BasePreprocessing, YoloPreprocessing

UNIQUE = ('Licence_number', 'Issue_organisation_code')


def _bbox(conf, label):
    return [0, 0, 1, 1, conf, 0, label]


# ---------------------------------------------------------------------------
# Field deduplication (unique & paired ru/en)
# ---------------------------------------------------------------------------

class TestDuplicateFieldDedup:
    def test_drops_lower_confidence_duplicate(self):
        bboxes = [_bbox(0.6, 'Licence_number'), _bbox(0.9, 'Licence_number')]
        drop = Pipeline._duplicate_field_indices(bboxes, UNIQUE)
        assert drop == {0}, 'Should drop the lower-confidence Licence_number'

    def test_single_field_kept(self):
        bboxes = [_bbox(0.6, 'Licence_number'), _bbox(0.9, 'First_name_ru')]
        assert Pipeline._duplicate_field_indices(bboxes, UNIQUE) == set()

    def test_dedups_each_unique_field(self):
        bboxes = [
            _bbox(0.5, 'Licence_number'),           # 0 -> drop
            _bbox(0.8, 'Licence_number'),           # 1 keep
            _bbox(0.7, 'Issue_organisation_code'),  # 2 keep
            _bbox(0.4, 'Issue_organisation_code'),  # 3 -> drop
            _bbox(0.9, 'Birth_date'),               # 4 keep (not unique-listed)
        ]
        assert Pipeline._duplicate_field_indices(bboxes, UNIQUE) == {0, 3}

    def test_empty(self):
        assert Pipeline._duplicate_field_indices([], UNIQUE) == set()


class TestPairedDuplicates:
    def test_same_line_labeled_both_ways_keeps_the_confident_label(self):
        boxes = [[513, 384, 801, 423, 0.922, 11, 'Birth_place_ru'],
                 [513, 384, 801, 422, 0.619, 12, 'Birth_place_en']]
        assert Pipeline._paired_duplicate_indices(boxes) == {1}

    def test_the_english_label_wins_when_it_is_the_confident_one(self):
        boxes = [[100, 10, 300, 40, 0.55, 11, 'Birth_place_ru'],
                 [100, 10, 300, 40, 0.90, 12, 'Birth_place_en']]
        assert Pipeline._paired_duplicate_indices(boxes) == {0}

    def test_a_real_ru_en_pair_side_by_side_is_kept(self):
        boxes = [[407, 291, 571, 317, 0.95, 11, 'Birth_place_ru'],
                 [582, 289, 652, 316, 0.93, 12, 'Birth_place_en']]
        assert Pipeline._paired_duplicate_indices(boxes) == set()

    def test_stacked_lines_overlapping_like_a_licence_are_kept(self):
        boxes = [[100, 100, 400, 140, 0.9, 1, 'Last_name_ru'],
                 [100, 116, 400, 156, 0.9, 3, 'Last_name_en']]
        assert Pipeline._paired_duplicate_indices(boxes) == set()

    def test_different_fields_on_the_same_box_are_not_this_rule(self):
        boxes = [[100, 10, 300, 40, 0.9, 11, 'Birth_place_ru'],
                 [100, 10, 300, 40, 0.6, 2, 'Last_name_en']]
        assert Pipeline._paired_duplicate_indices(boxes) == set()


# ---------------------------------------------------------------------------
# Address digit routing & OCR corrections
# ---------------------------------------------------------------------------

class TestAddressDigitRouting:
    @pytest.mark.parametrize('en_text,expected', [
        ('67', True),
        ('103', True),
        ('76', True),
        ('5', True),
        ('16', True),
        ('5A', True),
        ('1PH', False),
        ('KVLIFBCKAA', False),
        ('P-H', False),
        (',CT', False),
        ('', False),
        (None, False),
    ])
    def test_is_number_token(self, en_text, expected):
        assert Pipeline._is_number_token(en_text) is expected


def test_empty_sex_correction_returns_empty():
    assert check_en_sex('') == ''
    assert check_en_sex('...') == ''
    assert check_en_sex('  ') == ''
    assert check_rus_sex('') == ''
    assert check_rus_sex('...') == ''
    assert check_rus_sex('  ') == ''
    assert check_en_sex('M') == 'M'
    assert check_en_sex('F') == 'F'
    assert check_rus_sex('МУЖ.') == 'М'
    assert check_rus_sex('ЖЕН.') == 'Ж'


# ---------------------------------------------------------------------------
# Packaging, config, preprocessing, batching & stage edge cases
# ---------------------------------------------------------------------------

def test_package_dunder_all_exports_valid_attributes():
    assert set(document_processing.__all__) == {'Pipeline', '__version__'}
    for name in document_processing.__all__:
        assert hasattr(document_processing, name)


def test_resolve_model_dir_honors_rdocs_models_root(tmp_path, monkeypatch):
    monkeypatch.setenv('RDOCS_MODELS_ROOT', str(tmp_path))
    resolved = dp_config.resolve_model_dir('Borders' if 'Borders' in dp_config._RAW_CFG else 'DocDetector')
    assert resolved == tmp_path / 'document_processing' / 'models' / 'Borders'

    monkeypatch.setenv('RDOCS_MODELS_ROOT', '')
    default_resolved = dp_config.resolve_model_dir('DocDetector')
    assert default_resolved == Path(dp_config.DEFAULT_CFG['DocDetector'])


def test_pipeline_results_incomplete_accessors_do_not_raise_keyerror():
    r = PipelineResults()
    assert r.rotated_image is None
    assert r.angle is None
    assert r.img_with_fixed_perspective is None


def test_documents_stage_resets_scratch_and_handles_degenerate_crop():
    pipe = object.__new__(Pipeline)
    pipe._date_lines = {'Birth_date': [np.zeros((5, 5, 3), np.uint8)]}
    pipe._field_lines = {'Last_name_ru': [np.zeros((5, 5, 3), np.uint8)]}
    pipe._address_line_obboxes = [[0, 0, 1, 1, 0]]
    pipe._doc_year = 2011
    pipe._start_document()
    assert pipe._date_lines == {}
    assert pipe._field_lines == {}
    assert pipe._address_line_obboxes == []
    assert pipe._doc_year is None

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert DocumentsStage._document_crop(pipe, frame, {'box': [150, 150, 160, 160]}) is None
    assert DocumentsStage._document_crop(pipe, frame, {'box': [50, 50, 40, 40]}) is None


def test_prepare_image_clamps_extreme_aspect_and_rejects_bad_img_size():
    pipe = object.__new__(Pipeline)
    pipe._start_document()
    strip = np.zeros((1, 5000, 3), dtype=np.uint8)
    out = pipe._prepare_image(strip, img_size=1500)
    assert out.shape[0] == 1 and out.shape[1] in (1499, 1500) and out.shape[2] == 3

    with pytest.raises(ValueError, match='img_size'):
        pipe._prepare_image(strip, img_size=0)


def test_yolo_letterbox_handles_extreme_aspect_and_rejects_empty():
    pre = YoloPreprocessing(image_size=[640, 640, 3], padding_size=[0, 0], padding_color=[114, 114, 114])
    strip = np.zeros((1, 2000, 3), dtype=np.uint8)
    tensor, *_ = pre(strip)
    assert tensor.shape == (1, 640, 640, 3)

    with pytest.raises(ValueError, match='Empty image'):
        pre(np.zeros((0, 10, 3), dtype=np.uint8))


def test_textfields_predict_transform_clamps_boxes():
    det = object.__new__(TextFieldsDetector)
    det.model_name = 'TextFieldsDetector'
    det.model = SimpleNamespace(predict=lambda img: [[-5, -2, 120, 60, 0.9, 'Last_name_ru']])
    img = np.zeros((40, 80, 3), dtype=np.uint8)
    out = det.predict_transform(img)['TextFieldsDetector']
    assert out['warped_img'][0].shape == (40, 80, 3)


def test_base_preprocessing_raises_on_corrupt_image_file(tmp_path):
    bad = tmp_path / 'corrupt.jpg'
    bad.write_bytes(b'not a jpeg')
    pre = BasePreprocessing(image_size=[32, 32, 3], padding_size=[0, 0], padding_color=[0, 0, 0])
    with pytest.raises(ValueError, match='Could not decode image'):
        pre(bad)


def test_width_to_time_handles_non_power_of_two_stride_and_threads():
    class FakeOCR:
        class inference_model:
            @staticmethod
            def predict(xs):
                w = xs[0].shape[2]
                return [np.zeros((1, (w + 2) // 3, 10), dtype=np.float32)]

    calib = _WidthToTime(FakeOCR())
    results = []
    errors = []

    def worker():
        try:
            for w in range(10, 80):
                results.append((w, calib.t_of(w)))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for w, t in results:
        assert t == (w + 2) // 3


def test_yolo_segmentor_emits_uint8_masks():
    post = YOLOSegmentorPostprocessing(mask_filter=0.5)
    proto = np.ones((16, 16, 4), dtype=np.float32)
    masks = np.ones((1, 4), dtype=np.float32)
    bbox = np.array([[2, 2, 14, 14]], dtype=np.float32)
    out_masks, segs = post(proto, masks, bbox, (0, 0), [16, 16])
    assert out_masks.dtype == np.uint8
    assert len(segs) == 1 and len(segs[0]) >= 4


def test_tie_cancellations_does_not_re_cancel_or_promote_older():
    stamps = [
        {'kind': 'deregistered', 'cancels': None, 'current': False},
        {'kind': 'registered', 'cancels': None, 'current': False},
        {'kind': 'registered', 'cancels': None, 'current': False},
    ]
    StampsStage._tie_cancellations(stamps)
    assert stamps[0]['cancels'] == 1
    assert not any(s['current'] for s in stamps)

    stamps2 = [
        {'kind': 'deregistered', 'cancels': None, 'current': False},
        {'kind': 'deregistered', 'cancels': None, 'current': False},
        {'kind': 'registered', 'cancels': None, 'current': False},
        {'kind': 'registered', 'cancels': None, 'current': False},
    ]
    StampsStage._tie_cancellations(stamps2)
    assert stamps2[0]['cancels'] == 2
    assert stamps2[1]['cancels'] == 3
    assert not any(s['current'] for s in stamps2)

    stamps3 = [
        {'kind': 'registered', 'cancels': None, 'current': False},
        {'kind': 'deregistered', 'cancels': None, 'current': False},
        {'kind': 'registered', 'cancels': None, 'current': False},
    ]
    StampsStage._tie_cancellations(stamps3)
    assert stamps3[1]['cancels'] == 2
    assert stamps3[0]['current'] is True
    assert stamps3[2]['current'] is False


def test_read_mrz_skips_retries_when_already_overlong():
    class _SpyMrz(MrzReadingStage):
        def __init__(self):
            self.calls = 0
            self._mrz_zone = {
                'canvas': np.zeros((100, 500, 3), dtype=np.uint8),
                'boxes': [[10, 10, 490, 40]],
                'span': None,
            }
            self.ocr_lat = self

        @property
        def model_name(self):
            return 'm'

        def predict(self, crop):
            self.calls += 1
            return {'m': {'ocr_output': 'A' * 50}}

        def fix_errors(self, field_type, text):
            return text

    spy = _SpyMrz()
    overlong = 'P<RUSROGOVA<<IRINA<<<<<<<<<<<<<<<<<<<<<<<<<<<'  # 45 chars
    assert spy._read_mrz(0, overlong) == overlong
    assert spy.calls == 0
