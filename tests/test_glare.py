import glob
from pathlib import Path
from document_processing.processing.models import ModelLoader
from document_processing.pipeline_modules import *


def test_glare():
    loader = ModelLoader()
    model = loader(Path('../document_processing/models/Glare/ONNX/model.json'))
    for image_file in glob.glob('images/Glare/*'):
        image_file_path = Path(image_file)
        ground_truth = image_file_path.name.split('.')[0].split('_')[0].upper()
        result = model.predict(image_file_path)[0]
        assert ground_truth == result


def test_glare_originals():
    glare = Glare('ONNX')
    for image_file in glob.glob('../tests/images/Originals/Glare/*'):
        image_file_path = Path(image_file)
        ground_truth = image_file_path.name.split('.')[0].split('_')[0]
        result = glare.predict(image_file_path)['Glare'][0]
        assert ground_truth == result
