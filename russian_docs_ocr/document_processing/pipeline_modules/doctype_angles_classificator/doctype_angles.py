from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2

class DocTypeAngles(BaseModule):
    """Detects type of document from image.

    Identifies whether image contains a passport, driver's license etc.

    """

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes document type and angles detection model."""
        self.model_name = 'DocTypeAngles'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Predicts document type and confidence score.

        Args:
            img: Input document image

        Returns:
            Dict with document type string and confidence
        """
        self.load_img(img)

        doctype_meta, angle_meta = self.model.predict(img)
        doc_type, dist, thresh = doctype_meta
        doctype_conf = np.round(1 - dist / thresh, 2)
        angle, angle_conf = angle_meta
        meta = {
            'DocType':
                {
                    'doc_type': doc_type,
                    'confidence': doctype_conf,
                },
            'Angle90':
                {
                    'angle': angle,
                    'confidence': angle_conf,
                }
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Predicts document type and confidence score and applies rotation.

        Args:
            img: Input document image

        Returns:
            dict with angle, confidence and rotated image
        """
        img = self.load_img(img)
        doctype_meta, angle_meta = self.model.predict(img)
        doc_type, dist, thresh = doctype_meta
        doctype_conf = np.round(1 - dist / thresh, 2)

        angle, angle_conf = angle_meta
        for _ in range(angle // 90):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        meta = {
            'DocType':
                {
                    'doc_type': doc_type,
                    'confidence': doctype_conf,
                },
            'Angle90':
                {
                    'angle': angle,
                    'confidence': angle_conf,
                    'warped_img': img,
                }
        }
        return meta
