from .doc_detector import DocDetector
from .blur_detector import Blur
from .glare_detector import Glare
from .textfields_detector import TextFieldsDetector
from .address_lines_detector import AddressLinesDetector
from .address_textkind_classifier import AddressTextKindClassifier
from .lcd_spoofing_detector import LCDSpoofing
from .print_spoofing_detector import PrintSpoofing
from .words_detector import WordsDetector
from .ocr_rus import OCRRus
from .ocr_engnums import OCREngNums
from .doctype_angles_classificator import DocTypeAngles
from .deskewer import DocDeskewer

__all__ = 'DocDetector', 'TextFieldsDetector', 'Blur', 'Glare', 'LCDSpoofing', 'PrintSpoofing', \
    'WordsDetector', 'OCRRus', 'OCREngNums', 'DocTypeAngles', 'DocDeskewer', \
    'AddressLinesDetector', 'AddressTextKindClassifier'
