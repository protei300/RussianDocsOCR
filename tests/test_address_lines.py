import numpy as np
import pytest
from pathlib import Path
from russian_docs_ocr.document_processing.pipeline_modules import AddressLinesDetector


@pytest.fixture
def module():
    return AddressLinesDetector(model_format='ONNX', device='cpu')


@pytest.fixture
def canvases():
    imgs = list(Path('tests/images/AddressLines').glob('canvas_*.png'))
    assert imgs, 'No AddressLines canvas fixtures found'
    return imgs


class TestAddressLinesDetector:
    def test_predict_contract(self, module, canvases):
        """predict() returns oriented boxes as [cx, cy, w, h, angle, conf, cls, label]."""
        result = module.predict(canvases[0])
        assert module.model_name in result, f'Key {module.model_name!r} missing'
        obboxes = result[module.model_name]['obbox']
        assert isinstance(obboxes, list)
        assert len(obboxes) >= 1, 'Expected at least one address line on the canvas'
        for ob in obboxes:
            assert len(ob) == 8, f'OBB entry must have 8 fields, got {len(ob)}'
            assert isinstance(ob[7], str), 'Last field must be the class label'

    def test_predict_transform(self, module, canvases):
        """predict_transform() returns oriented boxes plus one upright patch each,
        ordered top-to-bottom."""
        inner = module.predict_transform(canvases[0])[module.model_name]
        obboxes, patches = inner['obbox'], inner['warped_img']
        assert len(obboxes) == len(patches), 'One patch per detected line expected'
        for patch in patches:
            assert isinstance(patch, np.ndarray) and patch.size > 0, 'Empty line patch'
        ys = [ob[1] for ob in obboxes]
        assert ys == sorted(ys), 'Lines must be ordered top-to-bottom (reading order)'
