"""Template-based page registration for booklet documents (internal passport).

The Borders path rectifies each page from its *segmentation contour*: a quad is
fitted to the mask and warped to a rectangle whose size is taken from the quad's
own side lengths. Everything downstream therefore inherits the mask's errors -
a clipped or bent contour gives a keystoned page, two pages of one spread get
different scales, and on a flat scan the model may take the white scanner lid
for the document and the page is lost entirely.

This module rectifies pages the other way round: it aligns the *printed blank*
of the page to a canonical template. The internal passport blank is identical
across all issued documents (captions, rules, header, ornamental frame), so a
reference page with every person-specific zone erased (``templates/``) is a
fixed, person-free target. For each page:

1. SIFT features of the photo are matched against the template's static-print
   features (ratio test + MAGSAC homography) to get *coarse* candidates: from
   each Borders quad when they exist (features restricted to the quad) and from
   the whole frame. A candidate with too few inliers is dropped.
2. The best candidates are *refined* in the canonical frame: the page is warped
   with the coarse homography, re-detected and re-matched against the template
   within a small search window (two passes, shrinking window), and optionally
   polished with a dense ECC fit on gradient magnitude under the static mask.
   Working in the canonical frame matters: it upsamples the page to the
   template's scale, where the small captions become matchable - a page that
   gives ~10 coarse inliers routinely gives 40+ refined ones.
3. When the other page of the spread is already registered, a *chain* prior is
   tried too: page 3 sits directly under page 2, so the other page's homography
   shifted by one page height is refined directly, without a coarse stage. This
   is what recovers the second page on flat scans, where its own static print
   is too small for coarse matching.
4. A page is accepted only if enough refined inliers survive, they are spread
   over the page (not one ornament), and the page quad in the photo is a sane
   convex shape of plausible size. Otherwise the caller keeps the Borders warp
   for that page.

Several reference pages per template page are supported (``refs`` in the
template json) - the print of different production runs differs slightly, and
the best-matching reference wins per page.

Output pages always have the template's fixed size, so the two pages of a
spread are stacked at one scale and every field lands at the same canvas
position for every document. Pure OpenCV, no learned weights.
"""
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ...geometry import Chain, Homography, VerticalRemap
from ..doc_detector.image_transformation import extract_quad
from .line_dewarp import apply_dewarp, dewarp_by_lines
from .line_refine import apply_refinement, refine_by_lines
from .quad_fit import fit_quad_lines

TEMPLATES_DIR = Path(__file__).resolve().parent / 'templates'

# Acceptance thresholds, see docs/progress-log.md for the measurements.
MIN_COARSE_INLIERS = 8
MIN_REFINED_INLIERS = 18
MIN_REFINED_INLIERS_CHAIN = 10 # with a geometric prior a smaller, well-spread set is convincing
MIN_SPREAD_PX_CHAIN = 140
MIN_SPREAD_PX = 120            # refined inliers must span at least this, both axes (rejects a
                               # match on one ornament; page 3 static print is mostly left-column)
RATIO_TEST = 0.8
COARSE_REPROJ_PX = 4.0
REFINE_REPROJ_PX = (3.0, 2.0)  # per refinement pass
REFINE_RADIUS_PX = (45.0, 15.0)
CHAIN_RADIUS_PX = (90.0, 15.0)
CHAIN_GAP_FRAC = 0.03          # spine gap between the two pages, fraction of page height
QUAD_DILATE_FRAC = 0.10        # Borders quad grown before selecting features
MAX_COARSE_CANDIDATES = 2      # refined per page, best coarse inliers first
GOOD_COARSE_INLIERS = 20       # a candidate this strong stops the search for more
ECC_SCALE = 0.4
ECC_ITERS = 30
# Output cushion around the canonical page, fraction of its size per side. The
# registered frame is exact, and the passport prints right up to the page edge
# (series/number along the right edge, the MRZ along the bottom): with no
# cushion the OCR read the vertical number's '3' as '8' and lost the MRZ's
# second line on samples/ (measured), which the Borders path avoided only by
# its own 1% margin. 3% keeps every edge-printed element whole.
PAGE_MARGIN_FRAC = 0.03


def _order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)[:, 0]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _gradient(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    m = np.sqrt(gx * gx + gy * gy)
    return m / (m.mean() + 1e-6)


class PageTemplate:
    """One reference of a canonical page: erased print, static mask, features."""

    def __init__(self, name: str, image_path: Path, mask_path: Path, sift):
        self.name = name
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        self.gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray.shape[:2]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        self.mask = (mask > 127).astype(np.uint8)
        self.kp, self.desc = sift.detectAndCompute(self.gray, self.mask * 255)
        self.pts = np.float32([k.pt for k in self.kp])
        self.grad = _gradient(self.gray)
        self.corners = np.float32([[0, 0], [self.width, 0],
                                   [self.width, self.height], [0, self.height]])

    def quad_in_image(self, H_img_to_page: np.ndarray) -> np.ndarray:
        """Page corners mapped back into the photo (4, 2)."""
        Hinv = np.linalg.inv(H_img_to_page)
        return cv2.perspectiveTransform(self.corners.reshape(1, -1, 2), Hinv).reshape(4, 2)


class PageRegistration:
    """Result for one page. ``H`` maps photo pixels to canonical page pixels."""

    def __init__(self, name, H=None, inliers=0, method='none', ecc=0.0, quad=None, ref=None):
        self.name = name
        self.H = H
        self.inliers = inliers
        self.method = method
        self.ecc = ecc
        self.quad = quad
        self.ref = ref

    @property
    def ok(self) -> bool:
        return self.H is not None

    def as_dict(self):
        return {'name': self.name, 'ok': self.ok, 'inliers': int(self.inliers),
                'method': self.method, 'ref': self.ref, 'ecc': round(float(self.ecc), 4),
                'quad': None if self.quad is None else self.quad.round(1).tolist(),
                'H': None if self.H is None else self.H.tolist()}


class PageRegistrar:
    """Registers the pages of one document type against its templates.

    Args:
        doc_type: template set to load (``templates/<doc_type>.json``).
        nfeatures: SIFT budget on the photo.
        use_ecc: run the dense ECC polish after the guided refinement.
        line_quads: take the Borders page quads from straight-line fits to the
            segmentation contour (``quad_fit.fit_quad_lines``) instead of the
            polygon corners; also extrapolates a side clipped by the frame.
        line_refine: after a page is warped into its frame, straighten it by
            its own lines and text-line profiles (``line_refine``).
        line_dewarp: then unbend it by the local tilt of its text lines
            (``line_dewarp``; the spine bend a homography cannot express).
    """

    def __init__(self, doc_type: str = 'INTPASSPORT', templates_dir: Optional[Path] = None,
                 nfeatures: int = 6000, use_ecc: bool = False, line_quads: bool = True,
                 line_refine: bool = True, line_dewarp: bool = True):
        self.line_quads = line_quads
        self.line_refine = line_refine
        self.line_dewarp = line_dewarp
        self.doc_type = doc_type.upper()
        tdir = Path(templates_dir) if templates_dir else TEMPLATES_DIR
        meta = json.loads((tdir / f'{self.doc_type.lower()}.json').read_text(encoding='utf-8'))
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        # pages: list of (name, [PageTemplate, ...]) in canvas order
        self.pages = []
        for p in meta['pages']:
            refs = p.get('refs') or [{'image': p['image'], 'mask': p['mask']}]
            self.pages.append((p['name'], [
                PageTemplate(p['name'], tdir / r['image'], tdir / r['mask'], self.sift)
                for r in refs]))
        first = self.pages[0][1][0]
        self.page_w, self.page_h = first.width, first.height
        self.margin = int(round(PAGE_MARGIN_FRAC * self.page_w))
        # size of one output page: canonical frame plus the cushion on every side
        self.out_w = self.page_w + 2 * self.margin
        self.out_h = self.page_h + 2 * self.margin
        self.use_ecc = use_ecc
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    @property
    def page_names(self):
        return [n for n, _ in self.pages]

    # ------------------------------------------------------------------ matching
    def _match(self, tpl: PageTemplate, kp, desc, reproj, radius=None, prior=None):
        """Template -> photo matches, MAGSAC homography (photo -> page).

        With ``prior`` (a photo->page homography) and ``radius`` only matches whose
        photo point lands within ``radius`` px of the template point after the
        prior are kept (guided matching).

        Returns (H, inliers, inlier template points) - H None on failure."""
        if desc is None or len(desc) < 8:
            return None, 0, None
        knn = self.matcher.knnMatch(tpl.desc, desc, k=2)
        good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < RATIO_TEST * n.distance]
        if len(good) < 8:
            return None, len(good), None
        src = np.float32([kp[m.trainIdx].pt for m in good])       # photo
        dst = tpl.pts[[m.queryIdx for m in good]]                  # page
        if prior is not None and radius is not None:
            pred = cv2.perspectiveTransform(src.reshape(1, -1, 2), prior).reshape(-1, 2)
            disp = pred - dst
            # A prior from the other page or a Borders quad is typically off by
            # a translation (spine gap, mask margin) larger than the window.
            # Centre the window on the median displacement of the roughly-near
            # matches first; the homography itself is still fitted photo->page.
            rough = np.linalg.norm(disp, axis=1) < 2.5 * radius
            if rough.sum() >= 6:
                disp = disp - np.median(disp[rough], axis=0)
            keep = np.linalg.norm(disp, axis=1) < radius
            src, dst = src[keep], dst[keep]
            if len(src) < 8:
                return None, len(src), None
        H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, reproj,
                                     maxIters=10000, confidence=0.999)
        if H is None or mask is None:
            return None, 0, None
        m = mask.ravel().astype(bool)
        return H, int(m.sum()), dst[m]

    def _features_in_quad(self, kp, desc, quad, shape):
        c = quad.mean(axis=0)
        grown = c + (quad - c) * (1 + 2 * QUAD_DILATE_FRAC)
        poly = np.round(grown).astype(np.int32)
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(m, poly, 1)
        pts = np.round(np.float32([k.pt for k in kp])).astype(int)
        pts[:, 0] = np.clip(pts[:, 0], 0, shape[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, shape[0] - 1)
        inside = m[pts[:, 1], pts[:, 0]] == 1
        idx = np.flatnonzero(inside)
        return [kp[i] for i in idx], (desc[idx] if len(idx) else None)

    # ---------------------------------------------------------------- refinement
    def _spread_ok(self, pts, min_px=MIN_SPREAD_PX) -> bool:
        if pts is None or len(pts) < 4:
            return False
        span = pts.max(axis=0) - pts.min(axis=0)
        return span[0] >= min_px and span[1] >= min_px

    def _refine(self, gray, tpl: PageTemplate, H, radii, min_inliers=MIN_REFINED_INLIERS):
        """Guided re-match in the canonical frame, optional ECC.

        Features are detected once on the page warped with the coarse H; the
        second pass re-matches the same features under the first pass's result
        with a tighter window and threshold (no second warp/detection - that is
        where the time went). Returns (H, inliers, inlier points, ecc); inliers
        0 when refinement failed."""
        page = cv2.warpPerspective(gray, H, (tpl.width, tpl.height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        kp, desc = self.sift.detectAndCompute(page, None)
        prior = np.eye(3)
        inl, pts = 0, None
        for radius, reproj in zip(radii, REFINE_REPROJ_PX):
            Hd, inl_p, pts_p = self._match(tpl, kp, desc, reproj, radius=radius, prior=prior)
            if Hd is None or inl_p < min_inliers:
                return H, 0, None, 0.0
            prior, inl, pts = Hd, inl_p, pts_p
        H = prior @ H
        ecc = 0.0
        if self.use_ecc:
            page = cv2.warpPerspective(gray, H, (tpl.width, tpl.height), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)
            He, ecc = self._ecc(tpl, page)
            if He is not None:
                H = He @ H
        return H, inl, pts, ecc

    def _ecc(self, tpl: PageTemplate, page_gray):
        a = cv2.resize(tpl.grad, None, fx=ECC_SCALE, fy=ECC_SCALE)
        b = cv2.resize(_gradient(page_gray), None, fx=ECC_SCALE, fy=ECC_SCALE)
        m = cv2.resize(tpl.mask, None, fx=ECC_SCALE, fy=ECC_SCALE, interpolation=cv2.INTER_NEAREST)
        warp = np.eye(3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(
                a, b, warp, cv2.MOTION_HOMOGRAPHY,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, 1e-5), m, 5)
        except cv2.error:
            return None, 0.0
        S = np.diag([ECC_SCALE, ECC_SCALE, 1.0]).astype(np.float64)
        Si = np.diag([1 / ECC_SCALE, 1 / ECC_SCALE, 1.0]).astype(np.float64)
        warp_full = Si @ warp.astype(np.float64) @ S      # page -> current warp
        return np.linalg.inv(warp_full), float(cc)

    # ---------------------------------------------------------------- validation
    def _sane_quad(self, quad, shape, ref_area=None):
        if quad is None or not np.all(np.isfinite(quad)):
            return False
        if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
            return False
        area = cv2.contourArea(quad.astype(np.float32))
        h, w = shape[:2]
        if area < 0.02 * h * w or area > 1.5 * h * w:
            return False
        q = _order_points(quad)
        wt = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
        ht = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
        if ht < 1e-3 or not (0.6 < (wt / ht) / (self.page_w / self.page_h) < 1.7):
            return False
        if ref_area is not None and ref_area > 0 and not (0.4 < area / ref_area < 2.5):
            return False
        return True

    def _accept(self, gray, tpl, H, radii, ref_area, chain=False):
        """Refine a candidate and validate it. Returns PageRegistration or None."""
        if not self._sane_quad(tpl.quad_in_image(H), gray.shape, ref_area):
            return None
        min_inl = MIN_REFINED_INLIERS_CHAIN if chain else MIN_REFINED_INLIERS
        min_spread = MIN_SPREAD_PX_CHAIN if chain else MIN_SPREAD_PX
        H, inl, pts, ecc = self._refine(gray, tpl, H, radii, min_inl)
        if inl < min_inl or not self._spread_ok(pts, min_spread):
            return None
        quad = tpl.quad_in_image(H)
        if not self._sane_quad(quad, gray.shape, ref_area):
            return None
        return PageRegistration(tpl.name, H, inl, 'x', ecc, quad)

    # ------------------------------------------------------------------- driver
    def register(self, img_rgb: np.ndarray, quads: Optional[List[np.ndarray]] = None):
        """Register every template page in ``img_rgb`` (upright RGB photo).

        Args:
            img_rgb: rotated document photo, as produced by the angle stage.
            quads: optional Borders quads, each (4, 2) in photo pixels.

        Returns:
            list of ``PageRegistration`` in template order; ``.ok`` is False for a
            page that could not be registered (caller falls back for it).
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb
        kp, desc = self.sift.detectAndCompute(gray, None)
        quads = [np.asarray(q, np.float32).reshape(4, 2) for q in (quads or [])]
        quad_feats = [self._features_in_quad(kp, desc, q, gray.shape) for q in quads]

        results = [PageRegistration(n) for n, _ in self.pages]
        taken = set()
        pending = list(range(len(self.pages)))
        # Two rounds so a page registered in round 1 can seed the other via the
        # chain prior. Round 2 runs the chain step ONLY: the coarse search and
        # the quad prior are deterministic and already failed in round 1, and
        # they are where a failing frame's time went (16 refinements, ~1.5 s
        # of 1.8 s, measured); with nothing registered after round 1 there is
        # no chain seed and the second round is skipped altogether.
        for _round in range(2):
            if _round == 1 and not any(r.ok for r in results):
                break
            for ti in list(pending):
                pname, refs = self.pages[ti]
                found = None
                cands = []   # (inliers, H, method, ref_area, tpl)
                if _round == 1:
                    refs_to_search = []
                else:
                    refs_to_search = refs
                for tpl in refs_to_search:
                    for qi, (qk, qd) in enumerate(quad_feats):
                        if qi in taken:
                            continue
                        H, inl, _ = self._match(tpl, qk, qd, COARSE_REPROJ_PX)
                        if H is not None and inl >= MIN_COARSE_INLIERS:
                            cands.append((inl, H, f'quad{qi}', cv2.contourArea(quads[qi]), tpl))
                    if not cands or max(c[0] for c in cands) < GOOD_COARSE_INLIERS:
                        H, inl, _ = self._match(tpl, kp, desc, COARSE_REPROJ_PX)
                        if H is not None and inl >= MIN_COARSE_INLIERS:
                            cands.append((inl, H, 'global', None, tpl))
                    if cands and max(c[0] for c in cands) >= GOOD_COARSE_INLIERS:
                        break   # strong enough; do not pay for the next reference
                cands.sort(key=lambda c: -c[0])
                for inl, H, method, ref_area, tpl in cands[:MAX_COARSE_CANDIDATES]:
                    found = self._accept(gray, tpl, H, REFINE_RADIUS_PX, ref_area)
                    if found is not None:
                        found.method, found.ref = method, refs.index(tpl)
                        break
                if found is None:
                    # chain prior from an already registered page of the spread
                    for tj, other in enumerate(results):
                        if tj == ti or not other.ok:
                            continue
                        dy = (ti - tj) * self.page_h * (1 + CHAIN_GAP_FRAC)
                        prior = np.array([[1, 0, 0], [0, 1, -dy], [0, 0, 1]], np.float64) @ other.H
                        for tpl in refs:
                            found = self._accept(gray, tpl, prior, CHAIN_RADIUS_PX, None, chain=True)
                            if found is not None:
                                found.method, found.ref = f'chain:{other.name}', refs.index(tpl)
                                break
                        if found is not None:
                            break
                if found is None and _round == 0:
                    # Borders quad as a geometric prior: the quad's own four-point
                    # warp, refined by template matching. Feature matching inside
                    # the quad may fail (small page, soft print) while the quad
                    # itself is nearly right - the old stack's result becomes the
                    # starting point instead of the answer.
                    for qi, q in enumerate(quads):
                        if qi in taken:
                            continue
                        prior = cv2.getPerspectiveTransform(
                            _order_points(q), refs[0].corners).astype(np.float64)
                        for tpl in refs:
                            found = self._accept(gray, tpl, prior, CHAIN_RADIUS_PX,
                                                 cv2.contourArea(q), chain=True)
                            if found is not None:
                                found.method, found.ref = f'quadprior{qi}', refs.index(tpl)
                                break
                        if found is not None:
                            break
                if found is None:
                    continue
                found.name = pname
                results[ti] = found
                pending.remove(ti)
                if found.method.startswith('quadprior'):
                    taken.add(int(found.method[9:]))
                elif found.method.startswith('quad'):
                    taken.add(int(found.method[4:]))
            if not pending:
                break
        return results

    def native_scale(self, regs) -> float:
        """Output scale (<= 1) at which no registered page is upsampled.

        The canonical frame is 1000 px wide; a page that spans only ~600 px in
        the photo would be stretched by 1.7x to fill it, and the OCR then reads
        the softened strokes wrong (the vertical series/number '3' as '8', the
        MRZ garbled - measured on samples/). The Borders path never upsampled,
        so neither does this one: the whole spread is emitted at the scale of
        its smallest registered page, keeping the canonical layout up to that
        one common factor. Returns 1.0 when no page is registered."""
        scale = 1.0
        for r in regs:
            if not r.ok:
                continue
            q = _order_points(r.quad)
            w_native = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
            scale = min(scale, w_native / self.page_w)
        return float(max(scale, 0.25))

    def out_size(self, scale: float = 1.0):
        """(width, height) of one output page at ``scale``."""
        return (max(1, int(round(self.out_w * scale))), max(1, int(round(self.out_h * scale))))

    def quad_matrix(self, quad: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """The perspective matrix ``warp_quad`` warps a photo quad with (photo -> page).

        Exposed so the warp and the way back (geometry.py) come from one matrix."""
        m = self.margin
        src = _order_points(quad)
        dst = np.float32([[m, m], [m + self.page_w, m], [m + self.page_w, m + self.page_h],
                          [m, m + self.page_h]]) * np.float32([scale, scale])
        return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))

    def warp_matrix(self, img_rgb: np.ndarray, M: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """The warp both ``warp_quad`` and ``warp_page`` apply, given its matrix."""
        return cv2.warpPerspective(img_rgb, M, self.out_size(scale), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

    def warp_quad(self, img_rgb: np.ndarray, quad: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Warp a photo quad (page edges, e.g. from Borders) into the canonical
        page frame: same output size and cushion as warp_page, so a page warped
        from its Borders quad and a page warped from its template homography
        stack into one consistent canvas."""
        return self.warp_matrix(img_rgb, self.quad_matrix(quad, scale), scale)

    def page_quads(self, segments, image_shape):
        """Page quads (ordered TL, TR, BR, BL) from Borders contours, plus the
        per-quad fit info. With ``line_quads`` the sides are straight lines
        fitted to the contour (see quad_fit); otherwise the polygon corners."""
        quads, infos = [], []
        for s in segments or []:
            if self.line_quads:
                q, info = fit_quad_lines(s, image_shape, self.page_h / self.page_w)
            else:
                q = extract_quad(s)
                info = {'method': 'polygon'}
            if q is not None:
                quads.append(_order_points(q).astype(np.float32))
                infos.append(info)
        return quads, infos

    def straighten(self, page: np.ndarray, scale: float = 1.0):
        """Straighten a warped page by its own lines: first the homography
        (line_refine), then the bend map (line_dewarp) on the result.
        Returns (page, info); the page is unchanged when nothing is applied."""
        page, info, _ = self.straighten_with_geometry(page, scale)
        return page, info

    def straighten_with_geometry(self, page: np.ndarray, scale: float = 1.0):
        """Same as ``straighten``, plus the map from the returned page back to
        ``page`` (geometry.py): the straightening homography and the bend map
        in the order applied, or None when nothing was applied."""
        inset = int(round(self.margin * scale))
        info = {'applied': False, 'reason': 'disabled'}
        maps = []
        if self.line_refine:
            gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY) if page.ndim == 3 else page
            Hm, info = refine_by_lines(gray, inset=inset)
            if Hm is not None:
                page = apply_refinement(page, Hm)
                maps.append(Homography(Hm))
        if self.line_dewarp:
            gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY) if page.ndim == 3 else page
            v, dinfo = dewarp_by_lines(gray, inset=inset)
            if v is not None:
                page = apply_dewarp(page, v)
                maps.append(VerticalRemap(v))
            info = dict(info, dewarp=dinfo)
        return page, info, (Chain(tuple(maps)) if maps else None)

    @staticmethod
    def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
        """Intersection over union of two convex photo quads."""
        a = _order_points(a).astype(np.float32)
        b = _order_points(b).astype(np.float32)
        inter_area, _ = cv2.intersectConvexConvex(a, b)
        union = cv2.contourArea(a) + cv2.contourArea(b) - inter_area
        return float(inter_area / union) if union > 0 else 0.0

    def warp_page(self, img_rgb: np.ndarray, reg: PageRegistration, scale: float = 1.0) -> np.ndarray:
        """Warp the photo to the canonical page with the PAGE_MARGIN_FRAC cushion
        on every side, at ``scale`` (see native_scale). At scale 1 the output
        is ``out_w x out_h`` and the template frame sits at offset ``margin``."""
        return self.warp_matrix(img_rgb, self.page_matrix(reg, scale), scale)

    def page_matrix(self, reg: PageRegistration, scale: float = 1.0) -> np.ndarray:
        """The matrix ``warp_page`` warps the photo with (photo -> page): the
        template homography, the cushion offset and the scale."""
        m = self.margin
        shift = np.array([[1, 0, m], [0, 1, m], [0, 0, 1]], np.float64)
        S = np.diag([scale, scale, 1.0])
        return S @ shift @ reg.H
