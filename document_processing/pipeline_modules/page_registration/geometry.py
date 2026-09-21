"""Template-free page geometry for every document type.

The three geometric steps developed for the internal passport do not need
the printed blank: the page quad from straight lines fitted to the Borders
contour (quad_fit), the straightening of a warped page by its own lines
and text-line profiles (line_refine) and the bend map from the local tilt
of the lines (line_dewarp). This helper applies them to the pages of ANY
document type, keeping the Borders path's own output convention (a page
rectified to its quad's side lengths, with the 1 % DOC_MARGIN_FRAC cushion)
so the rest of the pipeline sees the same kind of canvas as before. The
template locator stays passport-only (PageRegistrar).

Per document type only the page aspect ratio is known here, and only for
extrapolating a side that lies on the photo frame; types without an entry
simply skip that extrapolation.
"""
from typing import Optional

import cv2
import numpy as np

from ..doc_detector.image_transformation import DOC_MARGIN_FRAC, expand_quad, order_points
from .line_dewarp import apply_dewarp, dewarp_by_lines
from .line_refine import apply_refinement, refine_by_lines
from .quad_fit import fit_quad_lines

# page height / width of the physical page, for extrapolating a clipped side
DOC_ASPECT_H_OVER_W = {
    'INTPASSPORT': 88.0 / 125.0,     # one page of the booklet, landscape
    'EXTPASSPORT': 88.0 / 125.0,
    'EXTPASSPORTBIO': 88.0 / 125.0,
    'DL': 54.0 / 85.6,               # ID-1 card
    'SNILS': 54.0 / 85.6,            # laminated card of the 1996 form
}


def aspect_for(doc_type: Optional[str]) -> Optional[float]:
    if not doc_type:
        return None
    base = doc_type.upper().rsplit('_', 1)[0]
    return DOC_ASPECT_H_OVER_W.get(base)


class PageGeometry:
    """Quads from Borders contours, page warp and straightening, no template."""

    def __init__(self, line_quads: bool = True, line_refine: bool = True, line_dewarp: bool = True,
                 margin: float = DOC_MARGIN_FRAC, min_residual_deg: float = 1.0, min_disp_px: float = 6.0):
        # Thresholds are HIGHER than the passport registrar's (0.3 deg, 3 px): on
        # well-shot documents a correction below a degree only re-samples the
        # page for nothing and flips OCR fields both ways (measured on samples/:
        # -8 fields at 0.3 deg); the pipeline's own deskew ignores tilts under 2 deg.
        self.min_residual_deg = min_residual_deg
        self.min_disp_px = min_disp_px
        self.line_quads = line_quads
        self.line_refine = line_refine
        self.line_dewarp = line_dewarp
        self.margin = margin

    def quads(self, segments, image_shape, aspect_h_over_w: Optional[float] = None):
        out, infos = [], []
        for s in segments or []:
            q, info = fit_quad_lines(s, image_shape, aspect_h_over_w, extrapolate=aspect_h_over_w is not None)
            if q is None:
                continue
            if not self.line_quads:
                from ..doc_detector.image_transformation import extract_quad
                q, info = extract_quad(s), {'method': 'polygon'}
                if q is None:
                    continue
            out.append(order_points(q).astype(np.float32))
            infos.append(info)
        return out, infos

    def quad_transform(self, quad: np.ndarray, image_shape=None, keep_outside: bool = False):
        """Homography from the photo to the rectified page (its own side
        lengths, with the Borders path's cushion) and the page size (w, h).

        Exactly the Borders path's ``four_point_transform`` geometry, corner
        clipping to the photo included, so that with nothing else applied the
        page is bit-for-bit the default one (verified: 'warp only' must
        reproduce the default eval). ``keep_outside`` skips the clipping for
        a quad whose side was extrapolated past the frame on purpose."""
        rect = expand_quad(order_points(quad), self.margin)
        if image_shape is not None and not keep_outside:
            rect[:, 0] = np.clip(rect[:, 0], 0, image_shape[1])
            rect[:, 1] = np.clip(rect[:, 1], 0, image_shape[0])
        tl, tr, br, bl = rect
        w = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
        h = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
        if w < 2 or h < 2:
            return None, (w, h)
        dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        return cv2.getPerspectiveTransform(rect.astype(np.float32), dst).astype(np.float64), (w, h)

    @staticmethod
    def _warp(img, M, size, replicate=False):
        # an extrapolated clipped side reaches outside the photo: fill it by
        # border replication; otherwise the default path's constant border
        mode = cv2.BORDER_REPLICATE if replicate else cv2.BORDER_CONSTANT
        return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR, borderMode=mode)

    def warp(self, img: np.ndarray, quad: np.ndarray, keep_outside: bool = False):
        M, size = self.quad_transform(quad, img.shape, keep_outside)
        return None if M is None else self._warp(img, M, size, keep_outside)

    def rectify(self, img: np.ndarray, quad: np.ndarray, keep_outside: bool = False):
        """One page from the photo: quad warp, straightening homography and
        bend map - with the quad warp and the straightening COMPOSED into a
        single resampling of the photo. Resampling a page twice softens the
        strokes, and on well-shot documents that cost more fields than the
        geometry gained (measured on samples/: -6 for the re-cut alone, -16
        with a second warp on top). Returns (page, info) or (None, info)."""
        M, size = self.quad_transform(quad, img.shape, keep_outside)
        if M is None:
            return None, {'applied': False, 'reason': 'degenerate quad'}
        page = self._warp(img, M, size, keep_outside)
        inset = int(round(self.margin * size[0]))
        info = {'applied': False, 'reason': 'disabled'}
        if self.line_refine:
            gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY) if page.ndim == 3 else page
            Hm, info = refine_by_lines(gray, inset=inset, min_residual_deg=self.min_residual_deg)
            if Hm is not None:
                page = self._warp(img, Hm @ M, size, keep_outside)   # one resampling, not two
        if self.line_dewarp:
            gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY) if page.ndim == 3 else page
            v, dinfo = dewarp_by_lines(gray, inset=inset, min_disp_px=self.min_disp_px)
            if v is not None:
                page = apply_dewarp(page, v)
            info = dict(info, dewarp=dinfo)
        return page, info

    def straighten(self, page: np.ndarray):
        """line_refine then line_dewarp on an already rectified page (page,
        info); prefer ``rectify`` when the photo is at hand."""
        inset = int(round(self.margin * page.shape[1]))
        info = {'applied': False, 'reason': 'disabled'}
        if self.line_refine:
            gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY) if page.ndim == 3 else page
            Hm, info = refine_by_lines(gray, inset=inset, min_residual_deg=self.min_residual_deg)
            if Hm is not None:
                page = apply_refinement(page, Hm)
        if self.line_dewarp:
            gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY) if page.ndim == 3 else page
            v, dinfo = dewarp_by_lines(gray, inset=inset, min_disp_px=self.min_disp_px)
            if v is not None:
                page = apply_dewarp(page, v)
            info = dict(info, dewarp=dinfo)
        return page, info
