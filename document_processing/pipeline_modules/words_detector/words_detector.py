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
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool=False, runtime: str = None):
        """Initializes the words detection model."""
        self.model_name = 'WordsDetector'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose, runtime=runtime)

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

    @staticmethod
    def _reading_order(boxes):
        """Sort word boxes into reading order: cluster into lines by vertical
        center proximity (within half a word height), lines top-to-bottom,
        words left-to-right inside a line.

        A plain x-sort interleaves the lines of multi-line fields (measured
        on the birth-certificate Birth_place/ZAGS fields: word salad with
        CER ~0.65); for single-line fields the result is exactly the old
        x-sorted order."""
        lines = []  # each: [mean_cy, mean_h, [boxes...]]
        for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
            cy, h = (box[1] + box[3]) / 2, box[3] - box[1]
            for line in lines:
                if abs(cy - line[0]) < 0.5 * max(h, line[1]):
                    n = len(line[2])
                    line[0] = (line[0] * n + cy) / (n + 1)
                    line[1] = (line[1] * n + h) / (n + 1)
                    line[2].append(box)
                    break
            else:
                lines.append([cy, h, [box]])
        ordered = []
        for _, _, line_boxes in lines:  # already top-to-bottom
            ordered += sorted(line_boxes, key=lambda b: b[0])
        return ordered

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects words and extracts image patches (in reading order).

        Args:
            img: Image containing text field

        Returns:
            Bounding boxes, List of extracted word image patches
        """
        img = self.load_img(img)
        bbox = self.model.predict(img)
        img_patches = []
        bbox = self._reading_order(bbox)
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
