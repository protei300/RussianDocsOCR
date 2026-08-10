import pytest
from pathlib import Path
from document_processing.processing.models import ModelLoader
from document_processing.pipeline_modules import *


@pytest.fixture
def module():
    return Blur(model_format='ONNX', device='cpu')


class TestBlur:
    def test_module(self, module):
        for image_file in Path('images/Originals/Blur').iterdir():
            ground_truth = image_file.stem.split('_')[0]  # 'good' or 'bad'
            result = module.predict(image_file)
            assert module.model_name in result, f'Key {module.model_name!r} missing'
            status, quality = result[module.model_name]
            assert status == ground_truth, f'{image_file.name}: expected {ground_truth!r}, got {status!r}'
            assert 0.0 <= quality <= 1.0, f'Quality score out of range: {quality}'
