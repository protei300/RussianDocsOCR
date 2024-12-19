from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Union, Dict, Tuple

import cv2
import numpy as np

from ..pipeline_modules import *


@dataclass(init=False)
class OCROptionsClass:
    """Class for storing OCR options for different document types.

    Holds common OCR options like fields needed, split preferences, etc.
    Sub-classes implement options specific to document types.
    """

    """list: Fields that need to be split for this doc type."""
    needed_split = []

    """list: English fields to recognize for this doc type."""
    en_fields = []

    """list: Russian fields to recognize for this doc type."""
    ru_fields = []

    """bool: Whether this doc type needs license number rotation."""
    needs_licence_rotation = False


    @classmethod
    def make_options(cls, doc_type):
        """Factory method to make OCR options for a document type.

        Args:
            doc_type (str): Document type string

        Returns:
            OCROptionsClass instance with options for the document type.
        """
        if 'intpassport' in doc_type.lower():
            return OCROptionsINTPassport()
        elif 'extpassport' in doc_type.lower():
            return OCROptionsEXTPassport()
        elif 'dl' in doc_type.lower():
            return OCROptionsDL()
        elif 'snils' in doc_type.lower():
            return OCROptionsSNILS()

class OCROptionsINTPassport(OCROptionsClass):
    """OCR options for internal Russian passports."""

    needed_split = ["Licence_number",
                    "Birth_place_ru", "Issue_organization_ru",
                    ]

    en_fields = ["Licence_number", "Issue_date", "Expiration_date", "Birth_date", "Issue_organisation_code", ]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", "Sex_ru"]
    needs_licence_rotation = True

class OCROptionsEXTPassport(OCROptionsClass):
    """OCR options for external Russian passports."""

    needed_split = ["Licence_number", "Birth_place_ru", "Birth_place_en", ]

    en_fields = ["Last_name_en", "First_name_en", "Licence_number", "Issue_date",
                 "Expiration_date", "Birth_date", "Birth_place_en",
                 "Issue_organization_en", "Living_region_en", "Sex_en",
                 "Issue_organisation_code", "Middle_name_en"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", "Sex_ru"]


class OCROptionsDL(OCROptionsClass):
    """OCR options for Russian driver's licenses."""

    needed_split = ["Licence_number", "Driver_class", "Birth_place_ru", "Birth_place_en",
                    "Living_region_ru", "Living_region_en", ]
    en_fields = ["Last_name_en", "First_name_en", "Licence_number", "Issue_date",
                 "Expiration_date", "Driver_class", "Birth_date", "Birth_place_en",
                 "Issue_organization_en", "Living_region_en",  "Issue_organisation_code", "Middle_name_en"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Issue_organization_ru",
                 "Living_region_ru", "Middle_name_ru", ]

class OCROptionsSNILS(OCROptionsClass):
    """OCR options for Russian SNILS documents."""

    needed_split = ["Last_name_ru", "First_name_ru", "Licence_number", "Issue_date",
                    "Birth_date", "Birth_place_ru", "Middle_name_ru", "Sex_ru", ]
    en_fields = ["Licence_number", "Issue_date", "Birth_date"]
    ru_fields = ["Last_name_ru", "First_name_ru", "Birth_place_ru", "Middle_name_ru", "Sex_ru", ]

class PipelineResults:
    """Stores results and metadata from a model pipeline.

    Attributes:
        meta_results (dict): Metadata from pipeline stages
        _timings (dict): Timing measurements for stages

    """
    def __init__(self):
        """Initializes empty result storage."""

        self.meta_results = dict(Quality={})
        self._timings = dict()

    @property
    def ocr(self) -> Union[Dict, None]:
        """Gets OCR extraction results dict, if available."""
        if self.meta_results.get('OCR'):
            return self.meta_results.get('OCR')
        else:
            return None

    @property
    def doctype(self) -> Union[str, None]:
        """Gets detected document type, if available."""
        doctype = self.meta_results.get('DocType')
        return doctype

    @property
    def quality(self) -> dict:
        """Gets image quality measurements."""
        return self.meta_results['Quality']

    @property
    def rotated_image(self) -> np.ndarray:
        """Gets image rotated by the Angle90 stage."""
        return self.meta_results['Angle90']['warped_img']

    @property
    def img_with_fixed_perspective(self) -> Union[list, None]:
        """Get result from doc detection net"""
        if self.meta_results.get('DocDetector'):
            return self.meta_results['DocDetector']['warped_img']
        else:
            return None

    @property
    def text_fields(self) -> Union[Tuple[list, list], None]:
        """Get text field patches with their meta"""
        if self.meta_results.get('TextFieldsDetector'):
            return self.meta_results['TextFieldsDetector']['bbox'], self.meta_results['TextFieldsDetector']['warped_img']
        else:
            return None

    @property
    def text_fields_meta(self) -> Union[Dict, None]:
        """Get text field meta"""
        if self.meta_results.get('TextFieldsDetector'):
            return self.meta_results['TextFieldsDetector']
        else:
            return None

    @property
    def words_patches(self) -> Union[Dict, None]:
        """Get split words patches"""
        if self.meta_results.get('WordsDetector'):
            return self.meta_results['WordsDetector']
        else:
            return None

    @property
    def full_report(self) -> dict:
        """Returns full report in dict format"""
        summary_dict = {}
        summary_dict['DocType'] = self.doctype
        summary_dict['OCR'] = self.ocr
        summary_dict['Quality'] = self.quality
        summary_dict['Timings'] = self.timings
        return summary_dict

    @property
    def timings(self) -> dict:
        """Gets per stage timings and total time."""
        total_time = 0
        timings = self._timings.copy()
        for value in timings.values():
            total_time += value
        timings['total'] = total_time
        return timings

    @timings.setter
    def timings(self, value):
        """Sets updated timings."""
        self._timings = self._timings | value




class Pipeline:
    """Pipeline for OCR processing of documents.

    Performs steps of pre-processing, text detection and OCR
    to extract text from documents.
    """

    def __init__(self, model_format='ONNX', device='cpu', verbose=False):
        """
        Initialize pipeline.

        Args:
            model_format (str): Format of models to use - ONNX, OpenVINO etc.
            device (str): Device for model inference - cpu, gpu etc.
            verbose (bool): Whether to print debug information.
        """
        print(f'DEVICE: {device}')
        self.angle90 = Angle90(model_format=model_format, device=device, verbose=verbose)
        self.doctype = DocType(model_format=model_format, device=device, verbose=verbose)
        self.doc_detector = DocDetector(model_format=model_format, device=device, verbose=verbose)
        self.text_fields = TextFieldsDetector(model_format=model_format, device=device, verbose=verbose)
        self.words_detector = WordsDetector(model_format=model_format, device=device, verbose=verbose)
        self.ocr_ru = OCRRus(model_format='ONNX' if model_format == 'OpenVINO' else model_format,
                             device=device, verbose=verbose)
        self.ocr_en = OCREngNums(model_format='ONNX' if model_format == 'OpenVINO' else model_format,
                                 device=device, verbose=verbose)
        self.lcd_spoofing = LCDSpoofing(model_format=model_format, device=device, verbose=verbose)
        self.print_spoofing = PrintSpoofing(model_format=model_format, device=device, verbose=verbose)
        self.glare = Glare(model_format=model_format, device=device, verbose=verbose)
        self.blur = Blur(model_format=model_format, device=device, verbose=verbose)
        self.ocr_options = OCROptionsClass

    def __call__(self, img_path: Union[Path, str, np.ndarray],
                 ocr=True,
                 get_doc_borders=True,
                 find_text_fields=True,
                 check_quality=True,
                 low_quality=True,
                 docconf=0.5,
                 img_size=1500,
                 ) -> PipelineResults:
        """
        Main pipeline processing method.

        Args:
            img_path: Path to input image.
            ocr: Whether to perform OCR.
            get_doc_borders: Whether to detect document borders.
            find_text_fields: Whether to detect text fields.
            check_quality: Whether to check image quality.
            low_quality: Whether to process low quality images.
            docconf: Minimum doc confidence threshold.
            img_size: Resize image to this size for processing.

        Returns:
            PipelineResults with extracted information.
        """

        self.results = PipelineResults()

        img = self._prepare_image(img_path, img_size=img_size)

        self.time_measure = {}

        #getting angles
        self._model_call(self._angle, img)
        img = self.results.rotated_image

        #getting doctype and conf
        self._model_call(self._doctype, img)
        doc_type = self.results.doctype
        if doc_type == 'NONE':
            print("[!] The document on picture has unknown type")
            return self.results
        doc_type, year = doc_type.rsplit('_', maxsplit=1)
        self.ocr_options = self.ocr_options.make_options(doc_type)

        #getting quality
        if check_quality:
            self._model_call(self._glare, img)
            self._model_call(self._blur, img)
            self._model_call(self._print_spoofing, img)
            self._model_call(self._lcd_spoofing, img)

        # checking quality of doc
        if not low_quality:
            quality = self.results.quality
            if quality.get('Glare', False) == 'bad' or quality.get('Blur', False) == 'bad' or quality['DocConf'] > docconf:
                print("[!] Doc quality is too low. You can check using results.quality, "
                      "or bypass using low_quality=True")
                return self.results


        #detecting doc
        if get_doc_borders:
            self._model_call(self._doc_detector, img)
            img = self.results.img_with_fixed_perspective

        # detecting fields
        if find_text_fields:
            #Intpassport has licence number rotated 90 deg
            rotate_licence = self.ocr_options.needs_licence_rotation
            self._model_call(self._fields_detector, img, rotate_licence=rotate_licence)
            text_fields = self.results.text_fields_meta
        else:
            return self.results


        #splitting words
        if text_fields:
            self._model_call(self._split_words, text_fields.copy(), doc_type)
            words_splitted = self.results.words_patches

            #OCR words
            if ocr and words_splitted:
                self._model_call(self._ocr, words_splitted, doc_type)

        return self.results


    def _angle(self, img):
        """
        Detect and fix angle of image.

        Args:
            img: Input image as numpy array.

        Returns:
            np.ndarray: Image with fixed angle.
        """
        result = self.angle90.predict_transform(img)
        self.results.meta_results = self.results.meta_results | result
        # return result[self.angle90.model_name]['warped_img']

    def _doctype(self, img):
        """
        Detect document type and confidence.

        Args:
            img: Input image

        Returns:
            str: Detected document type
            float: Confidence score
        """
        result = self.doctype.predict(img)
        doc_type, confidence = result[self.doctype.model_name].values()
        self.results.meta_results['DocType'] = doc_type
        self.results.meta_results['Quality']['DocConf'] = confidence
        return doc_type

    def _glare(self, img):
        """Check for glare quality"""
        qual, coef = self.glare.predict(img)[self.glare.model_name]
        self.results.meta_results['Quality']['Glare'] = qual
        return qual

    def _blur(self, img):
        """Check for blur quality."""
        qual, coef = self.blur.predict(img)[self.blur.model_name]
        self.results.meta_results['Quality']['Blur'] = qual
        return qual

    def _print_spoofing(self, img):
        """Check for print spoofing."""
        qual, coef = self.print_spoofing.predict(img)[self.print_spoofing.model_name]
        self.results.meta_results['Quality']['PrintSpoofing'] = qual
        return qual

    def _lcd_spoofing(self, img):
        """Check for LCD spoofing."""
        qual, coef = self.lcd_spoofing.predict(img)[self.lcd_spoofing.model_name]
        self.results.meta_results['Quality']['LCDSpoofing'] = qual
        return qual



    def _doc_detector(self, img):
        """
        Detect document borders and fix perspective.

        Args:
            img: Input image

        Returns:
            np.ndarray: Image with fixed perspective
        """
        result = self.doc_detector.predict_transform(img)
        self.results.meta_results = self.results.meta_results | result
        # img = result[self.doc_detector.model_name]['warped_img']
        # return img

    def _fields_detector(self, img, rotate_licence=False):
        """
        Detect text fields in document.

        Args:
            img: Input image
            rotate_licence: Whether to rotate license field

        Returns:
            dict: Detected text fields and patches
        """
        result = self.text_fields.predict_transform(img)
        text_fields = result[self.text_fields.model_name]


        if rotate_licence:
            for i, field in enumerate(text_fields['bbox']):
                if field[-1] == 'Licence_number':
                    text_fields['warped_img'][i] = cv2.rotate(text_fields['warped_img'][i],
                                                              cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.results.meta_results = self.results.meta_results | result



    def _split_words(self, text_fields: dict, doc_type:str):
        """
        Split text fields into words.

        Args:
            text_fields: Detected text fields

        Returns:
            dict: Text fields splitted into words
        """

        bboxes, patches = text_fields.values()

        # we have 2 licence numbers, saving only best one
        for block_name in ['Licence_number', 'Issue_organisation_code']:
            block_number = []
            for i, block in enumerate(text_fields['bbox']):
                if block[-1] == block_name:
                    block_number.append(i)
            if len(block_number) == 2:
                conf1 = text_fields['bbox'][block_number[0]][4]
                conf2 = text_fields['bbox'][block_number[1]][4]
                if conf1 > conf2:
                    text_fields['bbox'].pop(block_number[1])
                    text_fields['warped_img'].pop(block_number[1])
                else:
                    text_fields['bbox'].pop(block_number[0])
                    text_fields['warped_img'].pop(block_number[0])


        result = {}
        for i, bbox in enumerate(bboxes):

            if bbox[-1] not in self.ocr_options.en_fields and bbox[-1] not in self.ocr_options.ru_fields:
                continue


            if bbox[-1] in self.ocr_options.needed_split:
                words = self.words_detector.predict_transform(patches[i])[self.words_detector.model_name]['warped_img']
            else:
                words = [patches[i], ]

            if result.get(bbox[-1]):
                result[bbox[-1]]['patches'].extend(words)
            else:
                result[bbox[-1]] = {'patches': words,
                                    'ocr': []}

        self.results.meta_results[self.words_detector.model_name] = result
        return result

    def _ocr(self, words_dict: dict, doc_type:str):
        """
        Perform OCR on splitted words.

        Args:
            words: Text fields splitted into words

        Returns:
            dict: OCR text for input words
        """
        ocr_dict = {}
        for field_name, words in words_dict.items():
            ocred_words = []
            for i, word in enumerate(words['patches']):
                if doc_type == 'SNILS' and 'date' in field_name.lower() and i % 2 == 1 or \
                        field_name in self.ocr_options.ru_fields:
                    result = self.ocr_ru.predict(word)[self.ocr_ru.model_name]['ocr_output']
                    result = self.ocr_ru.fix_errors(field_type=field_name, text=result)
                    words['ocr'].append(result)
                    ocred_words.append(result)
                elif field_name in self.ocr_options.en_fields:
                    result = self.ocr_en.predict(word)[self.ocr_en.model_name]['ocr_output']
                    result = self.ocr_en.fix_errors(field_type=field_name, text=result)
                    words['ocr'].append(result)
                    ocred_words.append(result)


            if 'date' in field_name.lower() and doc_type != 'SNILS':
                ocr_dict[field_name] = '.'.join(ocred_words)
            elif 'date' in field_name.lower() and doc_type == 'SNILS':
                ocr_dict[field_name] = ' '.join(ocred_words)
            else:
                if ocr_dict.get(field_name):
                    ocr_dict[field_name] += ' ' + ' '.join(ocred_words)
                else:
                    ocr_dict[field_name] = ' '.join(ocred_words)

            ocr_dict[field_name] = ocr_dict[field_name].replace('  ', ' ').strip()


        # saving both OCR clear result and OCR of each patch
        self.results.meta_results['OCR'] = ocr_dict
        # self.results.meta_results[self.words_detector.model_name] = words_dict

    def _model_call(self, func, *args, **kwargs):
        """ Wrapper for making timing calculations."""
        time_start = time()
        result = func(*args, **kwargs)
        self.results.timings = {func.__name__: round(time() - time_start, 4)}
        return result

    def _prepare_image(self, img_path: Union[Path, str, np.ndarray], img_size: int = 1500):
        """
        Load image from path, validate it and resize.

        Args:
            img_path: Path to input image.
            img_size: Resize image to this size.

        Returns:
            np.ndarray: Loaded and resized image.
        """

        if isinstance(img_path, Path):
            img = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results.meta_results['image_path'] = img_path.as_posix()
        elif isinstance(img_path, str):
            img_path = Path(img_path)
            img = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results.meta_results['image_path'] = img_path
        elif isinstance(img_path, np.ndarray):
            img = img_path
        else:
            raise Exception("Unsupported image type")

        # check size of image, and resize if above 1500
        h, w = img.shape[:2]
        ratio = max(max(h, w) / img_size, 1)
        new_h, new_w = int(h // ratio), int(w // ratio)
        img = cv2.resize(img, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.results.meta_results['original_img'] = img

        return img










