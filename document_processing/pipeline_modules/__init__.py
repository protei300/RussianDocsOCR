from .angles_classificator import Angle90
from .doc_detector import DocDetector
from .blur_detector import Blur
from .glare_detector import Glare
from .doctype_classificator import DocType
from .textfields_detector import TextFieldsDetector
from .blur_detector import Blur
from .lcd_spoofing_detector import LCDSpoofing
from .print_spoofing_detector import PrintSpoofing
from .words_detector import WordsDetector
from .ocr_rus import OCRRus
from .ocr_engnums import OCREngNums

__all__ = 'Angle90', 'DocDetector', 'DocType', 'TextFieldsDetector', 'Blur', 'Glare', 'LCDSpoofing', 'PrintSpoofing', \
    'WordsDetector', 'OCRRus', 'OCREngNums'
