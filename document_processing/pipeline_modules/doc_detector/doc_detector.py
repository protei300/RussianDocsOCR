from ..base_module import BaseModule
from .image_transformation import fix_perspective, rectify_pages, stitch_pages
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

# A segment with (almost) no ink inside it is not a document page, however
# confident the model is: the white lid of a flatbed scanner next to a
# passport scores 0.90-0.96 as 'Document' and, being 3x larger than a page,
# used to win the area-based selection above and push both real pages out
# (~20 of 100 scans in the Damir set came out as an empty canvas). Ink is the
# mean gradient magnitude inside the eroded mask at ~400 px: measured 1.5-5.4
# on lids, 19-101 on passport pages (19 = a badly blurred one), 24-43 on the
# mostly bare registration page. A blank segment is dropped only when another
# segment with real ink exists, so a lone blank sheet still goes through as
# before.
BLANK_INK = 8.0
INK_SCALE_PX = 400


def segment_ink(gray: np.ndarray, contour) -> float:
    """Mean gradient magnitude inside ``contour`` (eroded so the segment's own
    edge does not count), on the image downscaled to INK_SCALE_PX."""
    h, w = gray.shape[:2]
    s = INK_SCALE_PX / float(max(h, w))
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    m = np.zeros(small.shape, np.uint8)
    cv2.fillPoly(m, [np.int32(np.round(pts * s))], 255)
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    if not m.any():
        return 0.0
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.hypot(gx, gy)[m > 0].mean())


def drop_blank_segments(img: np.ndarray, segm):
    """Indices of the segments to keep: blank ones (ink < BLANK_INK) are
    dropped when at least one inked segment exists. Returns the kept
    indices and the per-segment ink values."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    ink = [segment_ink(gray, s) if s is not None else 0.0 for s in segm]
    if any(v >= BLANK_INK for v in ink):
        keep = [i for i, v in enumerate(ink) if v >= BLANK_INK]
    else:
        keep = list(range(len(segm)))
    return keep, ink



class DocDetector(BaseModule):
    """Detects document and fixes perspective issues.

    Detects the document quadrangle, segments it from the background,
    and fixes perspective issues by transforming the document to
    a rectangular shape.

    Provides options to get just the detection outputs or also warp
    the document image to fix its perspective.

    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool=False, runtime: str = None):
        """Initializes the document detection model"""
        self.model_name = 'DocDetector'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose, runtime=runtime)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Predicts document detection outputs.

        Args:
            img: Input document image

        Returns:
            Dictionary with bboxes, masks and segmentation
        """
        img = self.load_img(img)

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
            # First drop segments without ink (a scanner lid, a blank sheet
            # next to the document): see BLANK_INK. Then the area rule.
            inked, ink_values = drop_blank_segments(img, segm)
            if len(inked) < len(segm):
                bbox = [bbox[i] for i in inked]
                mask = [mask[i] for i in inked]
                segm = [segm[i] for i in inked]
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
                # Pages are rectified separately and stitched afterwards, so
                # the individual pages can be handed to the pipeline: a
                # detector's 640x640 input gives a stitched spread only half of
                # itself per page. The stitched canvas is still produced here -
                # it stays the thing every existing consumer expects.
                pages, quads, borders_img = rectify_pages(img=img, segments=segm)
                if pages:
                    result_img, placements = stitch_pages(pages, quads, stack=stack)
                else:
                    result_img, placements = img, []
            except Exception as e:
                print(f'[!] Failed to fix perspective: {e!r}')
                result_img = borders_img = img
                pages, quads, placements = [], [], []
        else:
            result_img = borders_img = img
            pages, quads, placements = [], [], []
        meta = {
            self.model_name:
                {
                    'bbox': bbox,
                    'mask': mask,
                    'segm': segm,
                    'border_img': borders_img,
                    'warped_img': result_img,
                    # per-page rectified images plus the quad each came from and
                    # where it landed on the stitched canvas; empty for a
                    # single-page document, where 'warped_img' already IS the page
                    'pages': pages,
                    'page_quads': quads,
                    'page_placements': placements,

                }
        }

        return meta
