from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2

class DocTypeAngles(BaseModule):
    """Classifies document type and 90-degree rotation angle from an image.

    Identifies whether the image contains a passport, driver's license etc.
    and by how many degrees (multiple of 90) it is rotated.

    Output follows the standard module contract - a single dict keyed by
    ``model_name`` ('DocTypeAngles') with a flat payload:

        {'DocTypeAngles': {
            'doc_type': str,                # e.g. 'INTPASSPORT_2011', 'NONE'
            'doc_type_confidence': float,   # 1 - dist/threshold, rounded
            'angle': int,                   # 0/90/180/270
            'angle_confidence': float,
            'warped_img': np.ndarray,       # predict_transform only: upright image
        }}
    """

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes document type and angles detection model."""
        self.model_name = 'DocTypeAngles'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def __predict_meta(self, img: np.ndarray) -> dict:
        """Runs the model and builds the flat payload dict (no rotation)."""
        doctype_meta, angle_meta = self.model.predict(img)
        doc_type, dist, thresh = doctype_meta
        angle, angle_conf = angle_meta
        # thresh == 0 is the postprocessing's "no centroid within radius" sentinel
        # (it also sets dist to inf, so the division would raise ZeroDivisionError).
        # That case is maximally-unknown, hence confidence 0.0.
        confidence = np.round(1 - dist / thresh, 2) if thresh > 0 else 0.0
        return {
            'doc_type': doc_type,
            'doc_type_confidence': confidence,
            'angle': angle,
            'angle_confidence': angle_conf,
        }

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Predicts document type and rotation angle with confidences.

        Args:
            img: Input document image

        Returns:
            {'DocTypeAngles': {...}} - see class docstring.
        """
        img = self.load_img(img)
        return {self.model_name: self.__predict_meta(img)}

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Same as predict(), plus the image rotated upright ('warped_img').

        Args:
            img: Input document image

        Returns:
            {'DocTypeAngles': {...}} - see class docstring.
        """
        img = self.load_img(img)
        meta = self.__predict_meta(img)
        for _ in range(meta['angle'] // 90):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        meta['warped_img'] = img
        return {self.model_name: meta}
