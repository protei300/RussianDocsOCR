import pytest
import numpy as np
from pathlib import Path
from document_processing.pipeline_modules import *
from document_processing.processing.models import ModelLoader


@pytest.fixture
def model():
    model_loader = ModelLoader()
    return model_loader(Path('../document_processing/models/Borders/ONNX/model.json'))


@pytest.fixture
def module():
    return DocDetector(model_format='ONNX', device='cpu')


@pytest.fixture
def sample_images():
    return list(Path('../samples/DL_2011').glob('*.jpg'))[:3]


class TestDocDetector:
    def test_model(self, model, sample_images):
        """Raw model returns (bbox, mask, segm) tuple."""
        for img_file in sample_images:
            bbox, mask, segm = model.predict(img_file)
            assert hasattr(bbox, '__len__'), 'bbox must be array-like'
            assert hasattr(mask, '__len__'), 'mask must be array-like'
            assert hasattr(segm, '__len__'), 'segm must be array-like'
            assert len(bbox) > 0, f'No bboxes detected in {img_file.name}'

    def test_module(self, module, sample_images):
        """Module predict() returns correct dict structure."""
        img_file = sample_images[0]
        result = module.predict(img_file)
        assert module.model_name in result, f'Key {module.model_name!r} missing'
        inner = result[module.model_name]
        assert 'bbox' in inner, 'Missing bbox field'
        assert 'mask' in inner, 'Missing mask field'
        assert 'segm' in inner, 'Missing segm field'

    def test_module_transform(self, module, sample_images):
        """predict_transform() returns detection results plus border_img and warped_img."""
        img_file = sample_images[0]
        result = module.predict_transform(img_file)
        assert module.model_name in result, f'Key {module.model_name!r} missing'
        inner = result[module.model_name]
        assert 'bbox' in inner, 'Missing bbox field'
        assert 'border_img' in inner, 'Missing border_img field'
        assert 'warped_img' in inner, 'Missing warped_img field'
        assert isinstance(inner['warped_img'], np.ndarray), 'warped_img must be a numpy array'
