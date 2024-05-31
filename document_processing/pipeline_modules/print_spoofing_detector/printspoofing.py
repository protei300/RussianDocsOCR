from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2


class PrintSpoofing(BaseModule):
    """Detects printout spoofing in document images.

    Analyzes image to identify fakes and photocopies.
    Provides prediction and confidence score.

    Attributes:
        threshold (float): Minimum confidence threshold
            for classifying as original (default: 0.9)

    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose=False):
        """Initializes the anti-spoofing model."""
        self.model_name = 'PrintSpoofing'
        self.threshold = 0.9
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Classifies if image is a spoofed printout.

        Args:
            img (ndarray): Document image

        Returns:
            tuple:
                bool/str: Spoof prediction
                float: Confidence score
        """
        self.load_img(img)
        result, conf = self.model.predict(img)
        if conf < self.threshold:
            meta = {
                self.model_name: ('FAKE', conf)
            }
        else:
            meta = {
                self.model_name: (result, conf)
            }
        return meta







