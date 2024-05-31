from pathlib import Path
from typing import Union

import numpy as np

from ..base_module import BaseModule


class OCRRus(BaseModule):
    """Performs OCR on Russian text fields.

    Handles post-processing corrections on recognized Russian
    names, sex and other text fields.

    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes the Russian text OCR model."""
        self.model_name = 'OCRRus'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Runs OCR inference on image.

        Args:
            img: Image containing text

        Returns:
            Recognized text
        """
        self.load_img(img)

        ocr_output = self.model.predict(img)
        meta = {
            self.model_name: {
                'ocr_output': ocr_output
            }
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        pass

    def fix_errors(self, field_type: str, text: str) -> str:
        """Applies corrections based on field type.

        Args:
            field_type: Type of text field
            text: Recognized text

        Returns:
            Corrected text
        """
        if field_type in ['Last_name_ru',
                          'First_name_ru',
                          'Birth_place_ru',
                          'Living_region_ru',
                          'Middle_name_ru']:
            return self.check_russian_names(text)
        elif field_type in ['Sex_ru']:
            return self.check_rus_sex(text)
        else:
            return text

    @staticmethod
    def check_russian_names(name: str) -> str:
        """Cleans up recognized Russian names.

        Args:
            name (str): Recognized name text

        Returns:
            str: Cleaned up text
        """
        return name.lstrip('.')

    @staticmethod
    def check_rus_sex(sex: str) -> str:
        """Standardizes recognized Russian sex text.

        Args:
            sex (str): Recognized sex text

        Returns:
            str: Standardized as M or Ж
        """
        strip = sex.lstrip('.').upper()
        to_check = strip.replace('.', '')
        # if len(to_check) >= 3:
        #     result = 'МУЖ' if 'М' in to_check else 'ЖЕН'
        #     if '.' in strip:
        #         result = result + '.'
        # else:
        result = 'М' if 'М' in to_check else 'Ж'
            # if '.' in strip:
            #     result = result + '.'
        return result
