from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2


class LCDSpoofing(BaseModule):
    """Detects LCD display spoofing in document images.

    Analyzes image characteristics to identify spoofing from
    displays/screens. Provides a prediction and confidence score.
    0 - a fake or electronic version
    1 - a real one

    """

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose=False):
        """Initializes the anti-spoofing model."""
        self.model_name = 'LCDSpoofing'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Predicts whether image is LCD spoofed or not.

        Args:
           img (ndarray): Document image

        Returns:
           tuple:
               bool: Spoofing prediction
               float: Confidence (0-1)
        """
        self.load_img(img)
        result, conf = self.model.predict(img)
        meta = {
            self.model_name: (result, conf)
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Stub method for future extensions"""
        meta = {}

        return meta






