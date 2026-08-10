from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2


class AddressLinesDetector(BaseModule):
    """Detects oriented (rotated) address-line regions on the residence
    registration page of the Russian internal passport (INTPASSPORTADDR).

    Uses a YOLO-OBB detector so tilted / perspective-skewed stamp lines are
    captured directly without a separate deskew step. ``predict_transform``
    returns each line cropped and rotated upright, in reading order
    (top-to-bottom), ready for word splitting and OCR.
    """

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        self.model_name = 'AddressLinesDetector'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects oriented address lines.

        Returns:
            dict with oriented boxes as [cx, cy, w, h, angle, conf, cls, label].
        """
        img = self.load_img(img)
        obboxes = self.model.predict(img)
        return {self.model_name: {'obbox': obboxes}}

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects oriented address lines and extracts upright line patches.

        Returns:
            dict with oriented boxes and the corresponding deskewed line images
            (reading order: top-to-bottom).
        """
        img = self.load_img(img)
        obboxes = self.model.predict(img)
        line_patches = [self.crop_rotated(img, ob[:5]) for ob in obboxes]
        return {
            self.model_name: {
                'obbox': obboxes,
                'warped_img': line_patches,
            }
        }

    @staticmethod
    def crop_rotated(img: np.ndarray, obbox) -> np.ndarray:
        """Crops a rotated rectangle and returns it rotated upright.

        Args:
            img: source image (H, W, C).
            obbox: (cx, cy, w, h, angle_rad).

        Returns:
            Upright line patch (h, w, C).
        """
        cx, cy, w, h, angle = obbox
        w_i, h_i = max(1, int(round(w))), max(1, int(round(h)))
        M = cv2.getRotationMatrix2D((float(cx), float(cy)), np.degrees(angle), 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        patch = cv2.getRectSubPix(rotated, (w_i, h_i), (float(cx), float(cy)))
        return patch
