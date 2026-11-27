import numpy as np
import pytest
from pathlib import Path
from document_processing.pipeline_modules import TextFieldsDetector


@pytest.fixture(scope='module')
def module():
    return TextFieldsDetector(model_format='ONNX', device='cpu')


@pytest.fixture(scope='module')
def model(module):
    return module.model


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
        """Raw model returns a non-empty list of bounding boxes on DL_2011 samples."""
        for img_file in sample_images:
            result = model.predict(img_file)
            assert isinstance(result, list), f'Model must return a list of bboxes, got {type(result)}'
            assert len(result) > 0, f'No text fields detected in {img_file.name}'
            assert len(result[0]) >= 6, f'Expected [x1,y1,x2,y2,conf,cls,...], got {result[0]}'

    def test_module(self, module, sample_images):
        """Module predict() returns non-empty bbox list under module.model_name."""
        img_file = sample_images[0]
        result = module.predict(img_file)
        assert module.model_name in result, f'Key {module.model_name!r} missing'
        inner = result[module.model_name]
        assert 'bbox' in inner, 'Missing bbox field'
        assert isinstance(inner['bbox'], list) and len(inner['bbox']) > 0, 'Expected detected bboxes'

    def test_module_transform(self, module, sample_images):
        """predict_transform() returns non-empty bbox list and valid warped_img patches."""
        img_file = sample_images[0]
        result = module.predict_transform(img_file)
        assert module.model_name in result, f'Key {module.model_name!r} missing'
        inner = result[module.model_name]
        assert 'bbox' in inner, 'Missing bbox field'
        assert 'warped_img' in inner, 'Missing warped_img field'
        assert isinstance(inner['warped_img'], list) and len(inner['bbox']) > 0
        assert len(inner['warped_img']) == len(inner['bbox']), (
            'Number of patches must match number of bboxes'
        )
        for patch in inner['warped_img']:
            assert isinstance(patch, np.ndarray) and patch.ndim == 3 and patch.size > 0
