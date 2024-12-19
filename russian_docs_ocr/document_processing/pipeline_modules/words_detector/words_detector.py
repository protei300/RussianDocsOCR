from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np

class WordsDetector(BaseModule):
    """Detects and segments words in document text fields.

    Identifies individual words within text fields and
    returns bounding boxes and image patches for each word.

    Useful for cropping words to prepare for OCR.

    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes the words detection model."""
        self.model_name = 'WordsDetector'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects words, returns bounding boxes.

        Args:
            img: Image containing text field

        Returns:
            List of detected word bounding boxes
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
        """Detects words and extracts image patches.

        Args:
            img: Image containing text field

        Returns:
            Bounding boxes, List of extracted word image patches
        """
        img = self.load_img(img)
        bbox = self.model.predict(img)
        img_patches = []
        bbox.sort(key=lambda x: x[0])
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
