import pytest
from pathlib import Path
from document_processing.pipeline_modules import *
from document_processing.processing.models import ModelLoader

SAMPLES_DIR = Path('../samples')


@pytest.fixture
def model():
    model_loader = ModelLoader()
    return model_loader(Path('../document_processing/models/DocTypeAngles/ONNX/model.json'))


@pytest.fixture
def module():
    return DocTypeAngles(model_format='ONNX', device='cpu')


class TestDocTypeAngles:
    def test_model(self, model):
        """Raw model returns (doctype_meta, angle_meta) for each image."""
        for doc_dir in sorted(SAMPLES_DIR.iterdir()):
            if not doc_dir.is_dir():
                continue
            for img_file in list(doc_dir.glob('*.jpg'))[:1]:
                doctype_meta, angle_meta = model.predict(img_file)
                doc_type, dist, thresh = doctype_meta
                angle, conf = angle_meta
                assert isinstance(doc_type, str), f'doc_type must be a string, got {type(doc_type)}'
                assert angle in (0, 90, 180, 270), f'Unexpected angle value: {angle}'
                assert 0.0 <= conf <= 1.0, f'Angle confidence out of range: {conf}'

    def test_module(self, module):
        """Module predict() follows the standard {model_name: payload} contract."""
        doc_dir = next(d for d in sorted(SAMPLES_DIR.iterdir()) if d.is_dir())
        img_file = next(doc_dir.glob('*.jpg'))
        result = module.predict(img_file)
        assert module.model_name in result, f'Missing {module.model_name} key'
        meta = result[module.model_name]
        for key in ('doc_type', 'doc_type_confidence', 'angle', 'angle_confidence'):
            assert key in meta, f'Missing {key} in payload'
        assert 'warped_img' not in meta, 'predict() must not return warped_img'

    def test_module_accuracy(self, module):
        """Predicted doc type matches samples directory name."""
        for doc_dir in sorted(SAMPLES_DIR.iterdir()):
            if not doc_dir.is_dir():
                continue
            expected_type = doc_dir.name
            for img_file in list(doc_dir.glob('*.jpg'))[:2]:
                result = module.predict(img_file)
                predicted = result[module.model_name]['doc_type']
                assert predicted == expected_type, (
                    f'{img_file.name}: expected {expected_type!r}, got {predicted!r}'
                )

    def test_module_transform(self, module):
        """predict_transform() adds the upright image to the payload."""
        doc_dir = next(d for d in sorted(SAMPLES_DIR.iterdir()) if d.is_dir())
        img_file = next(doc_dir.glob('*.jpg'))
        result = module.predict_transform(img_file)
        assert module.model_name in result, f'Missing {module.model_name} key'
        meta = result[module.model_name]
        for key in ('doc_type', 'doc_type_confidence', 'angle', 'angle_confidence',
                    'warped_img'):
            assert key in meta, f'Missing {key} in payload'
