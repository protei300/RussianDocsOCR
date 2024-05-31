import pytest
from document_processing.pipeline_modules import ocr_rus
from document_processing.processing.models import ModelLoader
from pathlib import Path
import glob
import numpy as np
from PIL import Image
import os


@pytest.fixture
def model():
    model_loader = ModelLoader()
    return model_loader(Path('../document_processing/models/OCR/rus/ONNX/model.json'))


@pytest.fixture
def module():
    return ocr_rus.OCRRus(model_format='ONNX', device='cpu')


class TestOCRRus:
    def test_model(self, model):
        images = glob.glob(f'images/OCRRus/*/*')
        for img in images:
            gt = os.path.splitext(str(img))[0]
            gt = gt.split(os.path.sep)[-1]
            pred = model.predict(np.array(Image.open(img)))
            assert pred == gt

    def test_names(self):
        ocr = ocr_rus.OCRRus()
        assert ocr.fix_errors('First_name_ru', '.Иван') == 'Иван'

    def test_rus_sex(self):
        ocr = ocr_rus.OCRRus()
        assert ocr.fix_errors('Sex_ru', 'муж') == 'М'
        assert ocr.fix_errors('Sex_ru', 'жен.') == 'Ж'
        assert ocr.fix_errors('Sex_ru', '.муж.') == 'М'
        assert ocr.fix_errors('Sex_ru', '..муж..') == 'М'
        assert ocr.fix_errors('Sex_ru', 'м') == 'М'
        assert ocr.fix_errors('Sex_ru', 'ж') == 'Ж'
        assert ocr.fix_errors('Sex_ru', 'ма') == 'М'
        assert ocr.fix_errors('Sex_ru', 'жа') == 'Ж'
        assert ocr.fix_errors('Sex_ru', 'муш') == 'М'
        assert ocr.fix_errors('Sex_ru', '.шен.') == 'Ж'
