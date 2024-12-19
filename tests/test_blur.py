import glob
from pathlib import Path
from russian_docs_ocr.document_processing.processing.models import ModelLoader
from russian_docs_ocr.document_processing.pipeline_modules import *


def test_blur():
    loader = ModelLoader()
    model = loader(Path('russian_docs_ocr/document_processing/models/Blur/ONNX/model.json'))
    for image_file in glob.glob('images/Blur/*'):
        image_file_path = Path(image_file)
        ground_truth = image_file_path.name.split('.')[0]
        result = model.predict(image_file_path)[0]
        assert ground_truth == result


def test_blur_originals():
    blur = Blur('ONNX')
    for image_file in glob.glob('tests/images/Originals/Blur/*'):
        image_file_path = Path(image_file)
        ground_truth = image_file_path.name.split('.')[0].split('_')[0]
        result = blur.predict(image_file_path)['Blur'][0]
        assert ground_truth == result




