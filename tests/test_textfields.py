import pytest
from pathlib import Path
from document_processing.pipeline_modules import *
from document_processing.processing.models import ModelLoader


@pytest.fixture
def model():
    model_loader = ModelLoader()
    return model_loader(Path('../document_processing/models/TextFields/ONNX/model.json'))


@pytest.fixture
def module():
    return TextFieldsDetector(model_format='ONNX', device='cpu')


@pytest.fixture
def sample_images():
    # Use front-side (CR) images — more likely to contain detectable text fields
    imgs = list(Path('../samples/DL_2011').glob('*_CR_*.jpg'))[:3]
    if not imgs:
        imgs = list(Path('../samples/DL_2011').glob('*.jpg'))[:3]
    # Both globs empty would leave test_model looping over nothing and passing.
    assert imgs, 'no DL_2011 samples found'
    return imgs


class TestTextFieldsDetector:
    def test_model(self, model, sample_images):
        """Raw model returns a list of bounding boxes."""
        for img_file in sample_images:
            result = model.predict(img_file)
            assert isinstance(result, list), f'Model must return a list of bboxes, got {type(result)}'

    def test_module(self, module, sample_images):
        """Module predict() returns correct dict structure."""
        img_file = sample_images[0]
        result = module.predict(img_file)
        assert module.model_name in result, f'Key {module.model_name!r} missing'
        inner = result[module.model_name]
        assert 'bbox' in inner, 'Missing bbox field'
        assert isinstance(inner['bbox'], list), 'bbox must be a list'

    def test_module_transform(self, module, sample_images):
        """predict_transform() returns bbox list and warped_img patches."""
        img_file = sample_images[0]
        result = module.predict_transform(img_file)
        assert module.model_name in result, f'Key {module.model_name!r} missing'
        inner = result[module.model_name]
        assert 'bbox' in inner, 'Missing bbox field'
        assert 'warped_img' in inner, 'Missing warped_img field'
        assert isinstance(inner['warped_img'], list), 'warped_img must be a list of image patches'
        assert len(inner['warped_img']) == len(inner['bbox']), (
            'Number of patches must match number of bboxes'
        )
