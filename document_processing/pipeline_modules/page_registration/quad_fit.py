"""Page quad from straight-line fits to the segmentation contour.

The Borders mask usually follows the physical page edge well; what breaks the
old ``extract_quad`` (convex hull + polygon simplification) is the CORNER
choice: a thumb merged into the mask, a bent corner or a run of stair-steps
along a blurred edge moves one polygon vertex, and the whole side swings. A
straight line fitted to the hundred-odd contour points of that side survives
the thumb (RANSAC drops it as an outlier) where a polygon vertex does not.

Method: start from the polygon quad, assign every contour point to its nearest
side, fit each side with RANSAC + least squares, intersect adjacent lines,
repeat (the assignment improves as the lines do). Only sides with enough
well-spread inliers are refit; the others keep the polygon's estimate. Result
must be convex and agree with the initial quad (IoU) - otherwise the caller
keeps the polygon quad.

Optionally a side that lies on the photo frame (the page runs out of the
picture) is EXTRAPOLATED from the opposite side and the page's known aspect
ratio, so a clipped page keeps its scale instead of being stretched to fill
the frame; the missing strip becomes border padding. Affine approximation,
good enough for a fallback and marked in the returned info.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from ..doc_detector.image_transformation import extract_quad, order_points

RESAMPLE_STEP_PX = 3.0      # contour resampled to uniform arc-length spacing
ITERATIONS = 3
RANSAC_SAMPLES = 200
INLIER_FRAC = 0.012         # inlier band, fraction of the side's length (min 2 px)
MIN_SIDE_POINTS = 12
MIN_SUPPORT_FRAC = 0.30     # inliers must span this fraction of the side's length
ASSIGN_OVERHANG = 0.05      # a point may project slightly past a side's ends
MIN_IOU_WITH_INIT = 0.6
MAX_OUTSIDE_FRAC = 0.35     # a corner may leave the frame by this fraction of the side (clipped page)
FRAME_TOL_PX = 3.0
EXTRAPOLATE_MIN_VISIBLE = 0.5   # extrapolate a clipped side only if >= this of the page is visible
EXTRAPOLATE_MAX_VISIBLE = 0.97  # ... and the page is actually short of its aspect


def _resample(contour: np.ndarray, step: float) -> np.ndarray:
    pts = np.asarray(contour, np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return pts
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return pts
    n = max(int(total / step), len(pts))
    t = np.linspace(0.0, total, n, endpoint=False)
    return np.stack([np.interp(t, cum, closed[:, 0]), np.interp(t, cum, closed[:, 1])], axis=1)


def _fit_line(pts: np.ndarray, thr: float, rng: np.random.Generator):
    """RANSAC line through ``pts``; returns (point, unit direction, inlier mask) or None."""
    n = len(pts)
    if n < 2:
        return None
    i = rng.integers(0, n, RANSAC_SAMPLES)
    j = rng.integers(0, n, RANSAC_SAMPLES)
    d = pts[j] - pts[i]
    L = np.linalg.norm(d, axis=1)
    keep = L > 1e-6
    if not keep.any():
        return None
    d, i = d[keep] / L[keep, None], i[keep]
    normal = np.stack([-d[:, 1], d[:, 0]], axis=1)                 # (m, 2)
    dist = np.abs(np.einsum('mni,mi->mn', pts[None, :, :] - pts[i][:, None, :], normal))
    counts = (dist < thr).sum(axis=1)
    best = int(np.argmax(counts))
    inl = dist[best] < thr
    for _ in range(2):                                              # least-squares refit
        if inl.sum() < 2:
            return None
        vx, vy, x0, y0 = cv2.fitLine(pts[inl].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        p0, dv = np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)
        inl = np.abs((pts - p0) @ np.array([-dv[1], dv[0]])) < thr
    return p0, dv, inl


def _intersect(a, b) -> Optional[np.ndarray]:
    p, d = a[0], a[1]
    q, e = b[0], b[1]
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 1e-9:
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p + t * d


def _quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = order_points(a).astype(np.float32), order_points(b).astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def _clipped_sides(quad: np.ndarray, shape) -> list:
    """Indices of quad sides (0 top, 1 right, 2 bottom, 3 left) lying on the frame."""
    h, w = shape[:2]
    out = []
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        for coord, val in ((0, 0.0), (0, float(w - 1)), (1, 0.0), (1, float(h - 1))):
            if abs(a[coord] - val) <= FRAME_TOL_PX and abs(b[coord] - val) <= FRAME_TOL_PX:
                out.append(k)
                break
    return out


def _extrapolate(quad: np.ndarray, side: int, aspect_h_over_w: float):
    """Move the clipped ``side`` out to where the page's aspect ratio puts it.

    Returns (new quad, visible fraction) or (None, visible fraction) when the
    visible part is too short to trust or the page is already complete."""
    # side k runs quad[k] -> quad[k+1]; its corners are joined to the opposite
    # side's corners by the adjacent sides: quad[k] <- quad[k+3], quad[k+1] <- quad[k+2]
    k0, k1 = side, (side + 1) % 4
    o0, o1 = (side + 3) % 4, (side + 2) % 4
    opp_len = np.linalg.norm(quad[o1] - quad[o0])
    if opp_len < 1:
        return None, 0.0
    expect = opp_len * (aspect_h_over_w if side in (0, 2) else 1.0 / aspect_h_over_w)
    u0, u1 = quad[k0] - quad[o0], quad[k1] - quad[o1]
    l0, l1 = np.linalg.norm(u0), np.linalg.norm(u1)
    if l0 < 1 or l1 < 1:
        return None, 0.0
    visible = 0.5 * (l0 + l1) / expect
    if not (EXTRAPOLATE_MIN_VISIBLE <= visible <= EXTRAPOLATE_MAX_VISIBLE):
        return None, float(visible)
    new = quad.copy()
    new[k0] = quad[o0] + u0 / l0 * expect
    new[k1] = quad[o1] + u1 / l1 * expect
    return new, float(visible)


def fit_quad_lines(contour, image_shape=None, aspect_h_over_w: Optional[float] = None,
                   extrapolate: bool = True) -> Tuple[Optional[np.ndarray], dict]:
    """Quad of a page from line fits to its segmentation contour.

    Args:
        contour: (N, 2) contour points (photo pixels).
        image_shape: (h, w[, c]) of the photo; enables the frame-clipping check.
        aspect_h_over_w: page height / width; with ``image_shape`` enables
            extrapolation of a side that lies on the frame.
        extrapolate: allow that extrapolation.

    Returns:
        (quad ordered TL, TR, BR, BL as float32, info dict). ``quad`` is None only
        when no quad at all can be made from the contour. ``info['method']`` is
        'lines' when the line fit was used, 'polygon' when the polygon quad was
        kept (reason in ``info['reason']``).
    """
    init = extract_quad(contour)
    if init is None:
        return None, {'method': 'none'}
    init = order_points(init).astype(np.float64)
    info = {'method': 'polygon', 'refit': [False] * 4, 'clipped': [], 'extrapolated': None,
            'visible': None}
    pts = _resample(contour, RESAMPLE_STEP_PX)
    if len(pts) < 4 * MIN_SIDE_POINTS:
        info['reason'] = 'few points'
        return init.astype(np.float32), info
    rng = np.random.default_rng(0)
    # contour points on the photo frame are where the page LEFT the picture, not
    # where its edge is: they never vote for a side. A side with no other points
    # keeps the polygon's estimate (which runs along the frame) and is reported
    # as clipped; a side that is only partly out of frame is fitted from its
    # visible part alone and extended past the frame.
    usable = np.ones(len(pts), bool)
    if image_shape is not None:
        h, w = image_shape[:2]
        usable = ~((pts[:, 0] <= FRAME_TOL_PX) | (pts[:, 0] >= w - 1 - FRAME_TOL_PX)
                   | (pts[:, 1] <= FRAME_TOL_PX) | (pts[:, 1] >= h - 1 - FRAME_TOL_PX))
    quad = init.copy()
    refit = [False] * 4
    for _ in range(ITERATIONS):
        A = quad
        B = np.roll(quad, -1, axis=0)
        d = B - A
        L = np.linalg.norm(d, axis=1)
        if (L < 2).any():
            info['reason'] = 'degenerate side'
            return init.astype(np.float32), info
        u = d / L[:, None]
        rel = pts[None, :, :] - A[:, None, :]                           # (4, n, 2)
        proj = np.einsum('kni,ki->kn', rel, u) / L[:, None]
        dist = np.abs(np.einsum('kni,ki->kn', rel, np.stack([-u[:, 1], u[:, 0]], axis=1)))
        dist[(proj < -ASSIGN_OVERHANG) | (proj > 1 + ASSIGN_OVERHANG)] = np.inf
        side = np.argmin(dist, axis=0)
        assigned = np.isfinite(dist.min(axis=0))
        lines = []
        for k in range(4):
            sel = pts[(side == k) & assigned & usable]
            fit = None
            if len(sel) >= MIN_SIDE_POINTS:
                fit = _fit_line(sel, max(2.0, INLIER_FRAC * L[k]), rng)
                if fit is not None:
                    span = np.ptp((sel[fit[2]] - fit[0]) @ fit[1])
                    if span < MIN_SUPPORT_FRAC * L[k]:
                        fit = None
            refit[k] = fit is not None
            lines.append((fit[0], fit[1]) if fit is not None else (A[k], u[k]))
        corners = []
        for k in range(4):
            c = _intersect(lines[k - 1], lines[k])
            if c is None:
                info['reason'] = 'parallel sides'
                return init.astype(np.float32), info
            corners.append(c)
        quad = order_points(np.array(corners)).astype(np.float64)
    info['refit'] = refit
    if not any(refit):
        info['reason'] = 'no side fitted'
        return init.astype(np.float32), info
    if not cv2.isContourConvex(quad.astype(np.float32)):
        info['reason'] = 'not convex'
        return init.astype(np.float32), info
    iou = _quad_iou(quad, init)
    info['iou_init'] = round(iou, 3)
    if iou < MIN_IOU_WITH_INIT:
        info['reason'] = 'disagrees with polygon'
        return init.astype(np.float32), info
    if image_shape is not None:
        h, w = image_shape[:2]
        slack = MAX_OUTSIDE_FRAC * float(np.linalg.norm(quad[1] - quad[0]))
        if (quad[:, 0] < -slack).any() or (quad[:, 0] > w + slack).any() \
                or (quad[:, 1] < -slack).any() or (quad[:, 1] > h + slack).any():
            info['reason'] = 'corner outside frame'
            return init.astype(np.float32), info
        # a side on the frame border has no true edge behind it: keep it exactly
        # on the frame (the fit may have tilted it by a pixel)
        info['clipped'] = _clipped_sides(quad, image_shape)
        for k in info['clipped']:
            for idx in (k, (k + 1) % 4):
                quad[idx, 0] = np.clip(quad[idx, 0], 0, w - 1)
                quad[idx, 1] = np.clip(quad[idx, 1], 0, h - 1)
        if extrapolate and aspect_h_over_w and len(info['clipped']) == 1:
            new, visible = _extrapolate(quad, info['clipped'][0], aspect_h_over_w)
            info['visible'] = round(visible, 3)
            if new is not None:
                quad, info['extrapolated'] = new, info['clipped'][0]
    info['method'] = 'lines'
    return quad.astype(np.float32), info
