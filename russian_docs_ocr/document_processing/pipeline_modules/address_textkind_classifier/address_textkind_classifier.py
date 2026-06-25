from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2


class AddressTextKindClassifier(BaseModule):
    """Classifies an address-line crop as printed vs handwritten.

    Address lines are wide and short; the crop is letterboxed onto a square
    canvas (preserving aspect, gray padding) before the standard square
    classification resize, so glyph proportions are not distorted. Returns the
    label and P(handwritten).
    """

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes the printed-vs-handwritten address-line classifier."""
        self.model_name = 'AddressTextKindClassifier'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    @staticmethod
    def letterbox_square(img: np.ndarray, color=(114, 114, 114)) -> np.ndarray:
        """Pads a wide line crop onto a square canvas (centered, gray padding)
        so glyph proportions survive the square classification resize.

        Args:
            img: Line crop (H × W × 3).
            color: Padding color.

        Returns:
            Square (size × size × 3) image with the crop centered.
        """
        h, w = img.shape[:2]
        size = max(h, w)
        canvas = np.full((size, size, 3), color, np.uint8)
        y0, x0 = (size - h) // 2, (size - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = img
        return canvas

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Returns ('printed'|'handwritten', P(handwritten))."""
        image = self.load_img(img)
        label, prob = self.model.predict(self.letterbox_square(image))[0]
        return {self.model_name: (label, float(prob))}

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Alias for predict() (no geometric transform for a classifier)."""
        return self.predict(img)
