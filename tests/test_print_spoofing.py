import glob
from pathlib import Path
from document_processing.pipeline_modules import *


def test_print_spoofing():
    print_spoofing = PrintSpoofing('ONNX')
    for image_file in glob.glob('../tests/images/PrintSpoofing/*'):
        image_file_path = Path(image_file)
        ground_truth = image_file_path.name.split('.')[0].split('_')[0].upper()
        result = print_spoofing.predict(image_file_path)['PrintSpoofing'][0]
        assert ground_truth == result




