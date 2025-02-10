from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2
from .quality import QualityChecker


class Blur(BaseModule):
    """
    Blur detection
    Detects Blur, Background and faces at a canvas
    Blur has three levels 0, 5 and 10
    0 means a sharp document
    0.5 or 5 means a middle level of blur
    1 means absolutely blured document
    Background and faces are ignored, because it usually looks like blured
    In general for the whole document 1 is good and 0 is bad

    I set 0.9 of the quality level or 10% of blur for let a document passing the quality test
    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes the blur detection model."""
        self.model_name = 'Blur'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Predicts overall blur score for image between 0-1.

        0 - Extremely blurred
        0.5 - Moderate blur
        1 - Sharp image

        Args:
            img: Input document image

        Returns:
            detected blur amount
        """
        canvas_size = (7, 4)
        checker = QualityChecker(self.model, canvas_size)
        image = self.load_img(img)
        quality = checker.check_image_quality(image)
        # print(quality)

        # This needs a fix in a future.
        if quality > 0.9:
            meta = {
                self.model_name: ('good', quality)
            }
        else:
            meta = {
                self.model_name: ('bad', quality)
            }

        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Predicts blur and highlights blurred regions.

        Args:
            img: Input document image

        Returns:
            Blur score, annotated version with blurred regions highlighted
        """
        canvas_size = (7, 4)
        checker = QualityChecker(self.model, canvas_size)
        image = self.load_img(img)
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






