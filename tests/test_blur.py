"""Blur detector against labelled photographs.

OUT OF SERVICE IN THE PUBLIC REPOSITORY since 2026-08-25. The fixtures under
``images/Originals/Blur`` were whole photographs of real documents, and the label
IS the capture condition - a blurred shot cannot be produced from a clean render
or from a field crop, which is why nothing here could simply be swapped. They were
withdrawn with the rest of the real-document material; the suite runs in the closed
repository, where the material stays.

Kept as a skipping test rather than deleted, so the gap shows up in every run
instead of looking like a detector nobody tested. The older crops still in
``images/Blur`` are NOT a substitute: their names carry no good/bad convention and
the detector calls ``NonBlur.jpg`` blurred, so wiring them in would replace a
strict guard with a wrong one.
"""
import pytest
from pathlib import Path
from document_processing.processing.models import ModelLoader
from document_processing.pipeline_modules import *

FIXTURES = Path('images/Originals/Blur')

pytestmark = pytest.mark.skipif(
    not FIXTURES.is_dir(),
    reason='material withdrawn from the public tree on 2026-08-25 (whole photographs '
           'of real documents); runs in the closed repository, awaits synthetic '
           'blurred/sharp renders here')


@pytest.fixture
def module():
    return Blur(model_format='ONNX', device='cpu')


class TestBlur:
    def test_module(self, module):
        files = sorted(FIXTURES.iterdir())
        assert files, 'fixture directory exists but is empty'
        for image_file in files:
            ground_truth = image_file.stem.split('_')[0]  # 'good' or 'bad'
            result = module.predict(image_file)
            assert module.model_name in result, f'Key {module.model_name!r} missing'
            status, quality = result[module.model_name]
            assert status == ground_truth, f'{image_file.name}: expected {ground_truth!r}, got {status!r}'
            assert 0.0 <= quality <= 1.0, f'Quality score out of range: {quality}'
