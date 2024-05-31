from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np

class TextFieldsDetector(BaseModule):
    """Detects text field regions in document images.

    Identifies areas like names, numbers, dates etc and
    returns bounding boxes and image patches.

    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes the text field detection model."""
        self.model_name = 'TextFieldsDetector'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects text fields, returns bounding boxes.

        Args:
            img: Input document image

        Returns:
            List of detected text field bounding boxes
        """
        self.load_img(img)
        bbox = self.model.predict(img)
        meta = {
            self.model_name:
                {
                    'bbox': bbox,

                }
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects fields and extracts image patches.

        Args:
            img: Input document image

        Returns:
            Bounding boxes, List of extracted image patches
        """
        img = self.load_img(img)
        bbox = self.model.predict(img)
        img_patches = []
        for box in bbox:
            img_patches.append(img[box[1]:box[3], box[0]: box[2]])
        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                    'warped_img': img_patches,
                }
        }
        return meta
