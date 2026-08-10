import glob
from pathlib import Path
from document_processing.pipeline_modules import *


def test_lcd_spoofing():
    lcd = LCDSpoofing('ONNX')
    for image_file in glob.glob('../tests/images/LCDSpoofing/*'):
        image_file_path = Path(image_file)
        ground_truth = image_file_path.name.split('.')[0].split('_')[0].upper()
        result = lcd.predict(image_file_path)['LCDSpoofing'][0]
        assert ground_truth == result




