from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2
from .quality import QualityChecker


class Glare(BaseModule):
    """
    Glare detection
    Detects Glare at a document.
    0 is good, it means - no glare
    1 is bad, it means - absolutely glared document

    One flash can spoil the recognition process, and I set zero level of Glare to pass the quality test
    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes glare detection model."""
        self.model_name = 'Glare'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=False)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Analyzes image and returns glare score.

        Args:
            img: Input document image

        Returns:
            Glare score between 0 and 1
            (lower is better, 0 means no glare detected)
        """
        canvas_size = (7, 4)
        checker = QualityChecker(self.model, canvas_size)
        image = self.load_img(img)
        quality = checker.check_image_quality(image)

        if quality > 0:
            meta = {
                self.model_name: ('bad', quality)
            }
        else:
            meta = {
                self.model_name: ('good', quality)
            }

        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects glare regions and highlights them.

        Args:
            img: Input document image

        Returns:
            Glare score, annotated image
        """
        canvas_size = (7, 4)
        checker = QualityChecker(self.model, canvas_size)
        image = cv2.imread(str(img))
        quality = checker.check_image_quality(image)
        transformed_image = checker.annotate_image(image)
        if quality > 0.9:
            meta = {
                self.model_name: ('good', quality)
            }
        else:
            meta = {
                self.model_name: ('bad', quality),
                'warped_img': transformed_image
            }

        return meta






