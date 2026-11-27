"""Tests for the four image-quality / anti-spoofing ONNX modules and tile batching.

Each module is loaded once per file (module-scoped fixture on CPU) and checked
against tracked fixture images with non-empty guards and [0, 1] score bounds.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from document_processing.pipeline_modules import Blur, Glare, LCDSpoofing, PrintSpoofing
from document_processing.pipeline_modules.blur_detector.quality import (
    QualityChecker as BlurChecker,
)
from document_processing.pipeline_modules.glare_detector.quality import (
    QualityChecker as GlareChecker,
)

IMG_EXTS = ('.jpg', '.jpeg', '.png')


def _fixture_images(rel_dir: str) -> list[Path]:
    root = Path(rel_dir)
    if not root.is_dir():
        pytest.skip(
            f'{rel_dir}: material withdrawn from the public tree on 2026-08-25 '
            '(whole photographs of real documents); runs in the closed repository'
        )
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    assert files, f'fixture directory {rel_dir} exists but contains no images'
    return files


@pytest.fixture(scope='module')
def blur_module():
    _fixture_images('images/Originals/Blur')
    return Blur(model_format='ONNX', device='cpu')


@pytest.fixture(scope='module')
def glare_module():
    _fixture_images('images/Originals/Glare')
    return Glare(model_format='ONNX', device='cpu')


@pytest.fixture(scope='module')
def lcd_module():
    _fixture_images('images/LCDSpoofing')
    return LCDSpoofing(model_format='ONNX', device='cpu')


@pytest.fixture(scope='module')
def print_module():
    _fixture_images('images/PrintSpoofing')
    return PrintSpoofing(model_format='ONNX', device='cpu')


def test_blur_module(blur_module):
    for image_file in _fixture_images('images/Originals/Blur'):
        ground_truth = image_file.stem.split('_')[0]  # 'good' or 'bad'
        result = blur_module.predict(image_file)
        assert blur_module.model_name in result, f'Key {blur_module.model_name!r} missing'
        status, quality = result[blur_module.model_name]
        assert status == ground_truth, f'{image_file.name}: expected {ground_truth!r}, got {status!r}'
        assert 0.0 <= quality <= 1.0, f'Quality score out of range: {quality}'


def test_glare_module(glare_module):
    for image_file in _fixture_images('images/Originals/Glare'):
        ground_truth = image_file.stem.split('_')[0]  # 'good' or 'bad'
        result = glare_module.predict(image_file)
        assert glare_module.model_name in result, f'Key {glare_module.model_name!r} missing'
        status, quality = result[glare_module.model_name]
        assert status == ground_truth, f'{image_file.name}: expected {ground_truth!r}, got {status!r}'
        assert 0.0 <= quality <= 1.0, f'Quality score out of range: {quality}'


def test_lcd_spoofing_module(lcd_module):
    for image_file in _fixture_images('images/LCDSpoofing'):
        ground_truth = image_file.stem.split('_')[0].upper()
        result = lcd_module.predict(image_file)
        assert lcd_module.model_name in result
        status, score = result[lcd_module.model_name]
        assert status == ground_truth, f'{image_file.name}: expected {ground_truth!r}, got {status!r}'
        assert 0.0 <= score <= 1.0, f'Score out of range: {score}'


def test_print_spoofing_module(print_module):
    for image_file in _fixture_images('images/PrintSpoofing'):
        ground_truth = image_file.stem.split('_')[0].upper()
        result = print_module.predict(image_file)
        assert print_module.model_name in result
        status, score = result[print_module.model_name]
        assert status == ground_truth, f'{image_file.name}: expected {ground_truth!r}, got {status!r}'
        assert 0.0 <= score <= 1.0, f'Score out of range: {score}'


# ---------------------------------------------------------------------------
# Pure unit tests (no weights loaded)
# ---------------------------------------------------------------------------

def test_glare_predict_transform_matches_predict_direction():
    class FakeModel:
        def __init__(self, label, conf):
            self._res = [(label, conf)]

        def predict(self, img):
            return self._res

    g_clean = object.__new__(Glare)
    g_clean.model_name = 'Glare'
    g_clean.model = FakeModel('NO', 0.99)
    res_clean = g_clean.predict_transform(np.full((64, 64, 3), 200, np.uint8))
    assert res_clean['Glare'][0] == 'good' and res_clean['Glare'][1] == 0.0
    assert 'warped_img' not in res_clean

    g_bad = object.__new__(Glare)
    g_bad.model_name = 'Glare'
    g_bad.model = FakeModel('GLARE', 0.95)
    res_bad = g_bad.predict_transform(np.full((64, 64, 3), 200, np.uint8))
    assert res_bad['Glare'][0] == 'bad' and res_bad['Glare'][1] == 1.0
    assert 'warped_img' in res_bad


def test_blur_and_glare_batch_28_tiles_into_single_inference_call():
    calls = []

    class BatchedModel:
        preprocessings = [SimpleNamespace(image_size=(128, 128, 3), padding_size=(0, 0))]

        class inference_model:
            @staticmethod
            def predict(xs):
                calls.append(xs[0].shape)
                return [np.tile(np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], np.float32), (xs[0].shape[0], 1))]

        postprocessings = [lambda row: ('NonBlur', 1.0)]

    img = np.full((256, 448, 3), 200, np.uint8)
    bc = BlurChecker(BatchedModel(), (7, 4))
    assert bc.check_image_quality(img) == 1.0
    assert calls == [(28, 128, 128, 3)]
    bc.annotate_image(img, reuse=True)
    assert len(calls) == 1  # reused without re-running inference

    calls.clear()

    class BatchedGlare(BatchedModel):
        postprocessings = [lambda row: ('NO', 0.99)]

    gc = GlareChecker(BatchedGlare(), (7, 4))
    assert gc.check_image_quality(img) == 0.0
    assert calls == [(28, 128, 128, 3)]
    gc.annotate_image(img, reuse=True)
    assert len(calls) == 1
