import pytest
from document_processing.pipeline_modules import ocr_engnums
from document_processing.processing.models import ModelLoader
from pathlib import Path
import glob
import os
import numpy as np
from PIL import Image


@pytest.fixture
def model():
    model_loader = ModelLoader()
    return model_loader(Path('../document_processing/models/OCR/eng+nums/ONNX/model.json'))


@pytest.fixture
def module():
    return ocr_engnums.OCREngNums(model_format='ONNX', device='cpu')


class TestOCREngNums:
    def test_model(self, model):
        images = glob.glob(f'images/OCREngNums/*/*')
        for img in images:
            gt = os.path.splitext(str(img))[0]
            gt = gt.split(os.path.sep)[-1]
            pred = model.predict(np.array(Image.open(img)))
            assert pred == gt

    def test_dates(self):
        ocr = ocr_engnums.OCREngNums()
        assert ocr.fix_errors('Issue_date', '12.12-1234') == '12.12.1234'
        assert ocr.fix_errors('Expiration_date', '01-02-2222') == '01.02.2222'
        assert ocr.fix_errors('Birth_date', '01-02.199') == '01.02.199'
