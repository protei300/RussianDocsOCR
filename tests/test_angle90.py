import pytest
from russian_docs_ocr.document_processing.pipeline_modules import *
from russian_docs_ocr.document_processing.processing.models import ModelLoader
from pathlib import Path



@pytest.fixture
def model():
    model_loader = ModelLoader()
    return model_loader(Path('russian_docs_ocr/document_processing/models/Angles90/ONNX/model.json'))

@pytest.fixture
def module():
    return Angle90(model_format='ONNX', device='cpu')



class TestAngle90:

    def test_model(self, model):
        for img in Path('tests/images/Angle90').iterdir():
            angle, conf = model.predict(img)
            angle_expected = int(img.stem.split('_', maxsplit=1)[1])
            assert angle == angle_expected, 'Wrong angle detected'

    def test_module(self, module):
        img = next(iter(Path('tests/images/Angle90').iterdir()))
        result = module.predict_transform(img)

        assert module.model_name in result.keys(), f'Module name - {module.model_name} not found in result dict'

        result = result[module.model_name]

        assert 'angle' in result.keys(), 'No angle field in results'
        assert 'warped_img' in result.keys(), 'No warped_img field in results'
        angle = result['angle']
        angle_expected = int(img.stem.split('_', maxsplit=1)[1])
        assert angle == angle_expected, 'Wrong angle detected'

