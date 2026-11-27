import pytest
import numpy as np
from pathlib import Path
from document_processing.pipeline_modules import DocDetector


@pytest.fixture(scope='module')
def module():
    return DocDetector(model_format='ONNX', device='cpu')


@pytest.fixture(scope='module')
def model(module):
    return module.model


@pytest.fixture
def sample_images():
    imgs = list(Path('../samples/DL_2011').glob('*.jpg'))[:3]
    # test_model loops over these; an empty list would make it pass without
    # running a single prediction.
    assert imgs, 'no DL_2011 samples found'
    return imgs


class TestDocDetector:
    def test_model(self, model, sample_images):
        """Raw model returns non-empty (bbox, mask, segm) tuple."""
        for img_file in sample_images:
            bbox, mask, segm = model.predict(img_file)
            assert len(bbox) > 0, f'No bboxes detected in {img_file.name}'
            assert len(bbox[0]) >= 4, f'Unexpected bbox shape in {img_file.name}: {bbox[0]}'
            assert len(mask) > 0 and len(segm) > 0, f'Empty mask/segm in {img_file.name}'

    def test_module(self, module, sample_images):
        """Module predict() returns non-empty detection payload."""
        img_file = sample_images[0]
        result = module.predict(img_file)
        assert module.model_name in result, f'Key {module.model_name!r} missing'
        inner = result[module.model_name]
        assert len(inner['bbox']) > 0, 'Expected at least one border bbox'
        assert len(inner['mask']) > 0, 'Expected non-empty mask'
        assert len(inner['segm']) > 0, 'Expected non-empty segmentation contour'

    def test_module_transform(self, module, sample_images):
        """predict_transform() returns non-empty detections plus valid warped_img."""
        img_file = sample_images[0]
        result = module.predict_transform(img_file)
        assert module.model_name in result, f'Key {module.model_name!r} missing'
        inner = result[module.model_name]
        assert len(inner['bbox']) > 0, 'Expected at least one border bbox'
        assert isinstance(inner['border_img'], np.ndarray) and inner['border_img'].ndim == 3
        warped = inner['warped_img']
        assert isinstance(warped, np.ndarray) and warped.ndim == 3 and min(warped.shape[:2]) > 50, (
            f'Invalid warped_img shape: {getattr(warped, "shape", type(warped))}'
        )
