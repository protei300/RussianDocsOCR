from ..base_module import BaseModule
from .image_transformation import fix_perspective
from typing import Union
from pathlib import Path
import cv2
import numpy as np

# A second document page (passport spread) must cover at least this fraction of
# the largest segment's area to be kept. Confidence is NOT a reliable filter
# here (spurious strips often score 0.9+ while faint real pages score ~0.65),
# so selection is area-based. Genuine spread pages run ~0.84-1.0; spurious
# background blobs run <=~0.5, so 0.6 separates them.
SECOND_SEGMENT_AREA_FRAC = 0.6



class DocDetector(BaseModule):
    """Detects document and fixes perspective issues.

    Detects the document quadrangle, segments it from the background,
    and fixes perspective issues by transforming the document to
    a rectangular shape.

    Provides options to get just the detection outputs or also warp
    the document image to fix its perspective.

    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes the document detection model"""
        self.model_name = 'DocDetector'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Predicts document detection outputs.

        Args:
            img: Input document image

        Returns:
            Dictionary with bboxes, masks and segmentation
        """
        self.load_img(img)

        bbox, mask, segm = self.model.predict(img)
        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                    'mask': mask,
                    'segm': segm,
                }
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray], stack: str = 'auto',
                          max_pages: int = 2) -> dict:
        """Predicts outputs and fixes document perspective.

        Args:
            img: Input document image
            stack: Multi-document merge direction passed to fix_perspective
                   ('auto' picks horizontal/vertical from page layout).
            max_pages: Max document segments to keep. Single-page doc types
                   should pass 1 so background blobs are never stitched in;
                   internal-passport spreads pass 2.

        Returns:
            Dictionary with detections outputs and warped image
        """
        img = self.load_img(img)
        bbox, mask, segm = self.model.predict(img)

        if segm:

            # Select document segments by contour area: keep the largest page
            # plus, optionally, a second page whose area is at least
            # SECOND_SEGMENT_AREA_FRAC of it. This drops spurious thin
            # fragments (which can still carry high confidence) while keeping
            # genuine two-page spreads. Capped at 2 pages.
            areas = []
            for s in segm:
                pts = np.asarray(s, dtype=np.float32).reshape(-1, 2) if s is not None else np.empty((0, 2))
                areas.append(cv2.contourArea(pts) if len(pts) >= 3 else 0.0)

            if areas and max(areas) > 0:
                order = list(np.argsort(areas)[::-1])
                max_area = areas[order[0]]
                keep = [order[0]]
                for idx in order[1:]:
                    if len(keep) >= max(1, max_pages):
                        break
                    if areas[idx] >= SECOND_SEGMENT_AREA_FRAC * max_area:
                        keep.append(idx)
                keep = sorted(keep)
                bbox = [bbox[i] for i in keep]
                mask = [mask[i] for i in keep]
                segm = [segm[i] for i in keep]


            try:
                result_img, borders_img = fix_perspective(img=img, segments=segm, stack=stack)
            except Exception as e:
                print(f'[!] Failed to fix perspective: {e!r}')
                result_img = borders_img = img
        else:
            result_img = borders_img = img
        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                    'mask': mask,
                    'segm': segm,
                    'border_img': borders_img,
                    'warped_img': result_img,

                }
        }

        return meta
