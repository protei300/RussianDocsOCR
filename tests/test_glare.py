"""Glare detector against labelled photographs.

OUT OF SERVICE IN THE PUBLIC REPOSITORY since 2026-08-25. The fixtures under
``images/Originals/Glare`` were whole photographs of real documents, and the label
IS the capture condition - a glared shot cannot be produced from a clean render or
from a field crop, which is why nothing here could simply be swapped. They were
withdrawn with the rest of the real-document material and the suite runs in the
closed repository, where it stays.

Kept as a skipping test rather than deleted, so the gap shows up in every run
instead of looking like a detector nobody tested. The older crops still in
``images/Glare`` are NOT a substitute: they carry a different labelling convention
(``glare_``/``no_``) and the detector agrees with only ten of twelve of them, so
wiring them in would turn a strict guard into a soft one without saying so.
"""
import pytest
from pathlib import Path
from document_processing.processing.models import ModelLoader
from document_processing.pipeline_modules import *

FIXTURES = Path('images/Originals/Glare')

pytestmark = pytest.mark.skipif(
    not FIXTURES.is_dir(),
    reason='material withdrawn from the public tree on 2026-08-25 (whole photographs '
           'of real documents); runs in the closed repository, awaits synthetic '
           'glared/clean renders here')


@pytest.fixture
def module():
    return Glare(model_format='ONNX', device='cpu')


class TestGlare:
    def test_module(self, module):
        files = sorted(FIXTURES.iterdir())
        assert files, 'fixture directory exists but is empty'
        for image_file in files:
            ground_truth = image_file.stem.split('_')[0]  # 'good' or 'bad'
            result = module.predict(image_file)
            assert module.model_name in result, f'Key {module.model_name!r} missing'
            status, quality = result[module.model_name]
            assert status == ground_truth, f'{image_file.name}: expected {ground_truth!r}, got {status!r}'
            assert quality >= 0.0, f'Quality score must be non-negative: {quality}'
