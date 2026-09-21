"""Bend correction of a rectified page from the local tilt of its text lines.

A homography (see line_refine) makes a FLAT page straight. A booklet page
bent near the spine is not flat: its text lines curve, and their local tilt
changes along the line (with x) - a homography cannot express that. This
module measures the local tilt t(x, y) of the horizontal structures on a
grid of overlapping cells (projection profiles, blur-tolerant) plus the LSD
segments, fits a low-order polynomial

    t(x, y) = c0 + c1 x + c2 y + c3 x^2 + c4 x y      (x, y normalised)

and integrates it along x into a vertical displacement map

    v(x, y) = c0 x + c1 x^2/2 + c2 x y + c3 x^3/3 + c4 x^2 y/2

which is zero on the page's centre column and grows outward. The page is
then remapped by y' = y + v(x, y). Only the vertical displacement is
observable from tilts (horizontal foreshortening along a bend is not), and
that is what moves text lines out of a field detector's box and tilts words
for the OCR. Applied only when the map is worth it (max |v| above a few px)
and it actually reduces the residual tilt; bounded to a fraction of the page
height. No template, no weights, deterministic.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

from .line_refine import (ANGLE_TOL_DEG, BLOB_MAX_FRAC, LSD_MAX_WEIGHT, LSD_MIN_LEN_FRAC, PROFILE_COARSE,
                          PROFILE_FINE_STEP, PROFILE_MIN_PEAK, PROFILE_SCALE,
                          PROFILE_WEIGHT, _seg_angle, _segments, _wmedian)

GRID = 5                      # cells per axis (overlapping), each a third of the page
MIN_CELLS = 8                 # profile cells with a peak needed to fit a map
MIN_DISP_PX = 3.0             # max |v| below this is noise: leave the page alone
MAX_DISP_FRAC = 0.04          # max |v| above this fraction of the page height: reject
MIN_GAIN = 0.25               # weighted median |tilt| must drop by this fraction
IRLS_ITERS = 3
IRLS_SCALE_DEG = 1.0
RIDGE = 1e-3                  # tiny ridge on the coefficients


def measure_cells(gray: np.ndarray, inset: int = 0) -> np.ndarray:
    """Local tilt evidence: rows (x, y, tilt_deg, weight, source) with source
    0 = LSD horizontal segment (at its centre), 1 = profile cell (at its centre)."""
    H, W = gray.shape[:2]
    rows = []
    # LSD at half scale: segment angles and positions scale, and the detector
    # is the slowest step of the page (75 ms at full size)
    half = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    segs = _segments(half, LSD_MIN_LEN_FRAC * W * 0.5) * 2.0
    if len(segs):
        ang = _seg_angle(segs)
        L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        wgt = np.minimum(L / (0.1 * W), LSD_MAX_WEIGHT)
        for s, a, w in zip(segs, ang, wgt):
            if abs(a) < ANGLE_TOL_DEG:
                rows.append([0.5 * (s[0] + s[2]), 0.5 * (s[1] + s[3]), a, w, 0])
    x0, y0, x1, y1 = inset, inset, W - inset, H - inset
    if x1 - x0 < 90 or y1 - y0 < 90:
        return np.array(rows, np.float64).reshape(-1, 5)
    cw, ch = (x1 - x0) / 3.0, (y1 - y0) / 3.0
    cells = [(xb, yb) for yb in np.linspace(y0, y1 - ch, GRID) for xb in np.linspace(x0, x1 - cw, GRID)]
    for (xb, yb), (tilt, ratio) in zip(cells, _cell_tilts(gray[y0:y1, x0:x1], [(xb - x0, yb - y0, cw, ch)
                                                                                for xb, yb in cells])):
        if ratio >= PROFILE_MIN_PEAK:
            rows.append([xb + 0.5 * cw, yb + 0.5 * ch, tilt, PROFILE_WEIGHT * min(1.0, ratio - 1.0), 1])
    return np.array(rows, np.float64).reshape(-1, 5)


def _cell_tilts(region: np.ndarray, cells):
    """Tilt and peak ratio of the row profile in every cell (x, y, w, h of
    ``region``), like line_refine._profile_tilt but rotating the whole region
    ONCE per angle and slicing the cells out of it: 25 cells x 22 angles of
    small warps was the slowest step of the page (measured +0.4 s a spread).
    Rotation about the region centre instead of the cell centre only shifts
    a cell's content by a few pixels at these angles, which the profile does
    not mind; the profile is normalised by the valid pixels per row."""
    small = cv2.resize(region, None, fx=PROFILE_SCALE, fy=PROFILE_SCALE, interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = binary.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        big = st[:, cv2.CC_STAT_HEIGHT] > BLOB_MAX_FRAC * (cells[0][3] * PROFILE_SCALE)
        big[0] = False
        if big.any():
            binary[big[lab]] = 0
    binary = binary.astype(np.float32) / 255.0
    valid = np.ones_like(binary)
    centre = (w / 2.0, h / 2.0)
    boxes = [(int(x * PROFILE_SCALE), int(y * PROFILE_SCALE), max(2, int(cw * PROFILE_SCALE)),
              max(2, int(ch * PROFILE_SCALE))) for x, y, cw, ch in cells]

    def score(angles):
        out = np.zeros((len(angles), len(boxes)))
        for i, a in enumerate(angles):
            M = cv2.getRotationMatrix2D(centre, float(a), 1.0)
            rot = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            cnt = cv2.warpAffine(valid, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            # row sums of every cell from one cumulative sum along x
            rot_c = np.cumsum(rot, axis=1, dtype=np.float32)
            cnt_c = np.cumsum(cnt, axis=1, dtype=np.float32)
            for j, (x, y, cw, ch) in enumerate(boxes):
                x2 = min(x + cw, w) - 1
                ink = rot_c[y:y + ch, x2] - (rot_c[y:y + ch, x - 1] if x > 0 else 0)
                c = cnt_c[y:y + ch, x2] - (cnt_c[y:y + ch, x - 1] if x > 0 else 0)
                cmax = c.max()
                if cmax <= 0:
                    continue
                ok = c >= 0.6 * cmax
                prof = ink[ok] / np.maximum(c[ok], 1.0)
                out[i, j] = float(prof.var()) if len(prof) > 2 else 0.0
        return out

    coarse = score(PROFILE_COARSE)
    fine_angles = np.arange(-1.0, 1.001, PROFILE_FINE_STEP)
    results = []
    # one fine pass per distinct coarse peak (usually one or two per page)
    peaks = {}
    for j in range(len(boxes)):
        ib = int(np.argmax(coarse[:, j]))
        if ib == 0 or ib == len(PROFILE_COARSE) - 1 or coarse[ib, j] <= 0:
            peaks[j] = None
        else:
            peaks[j] = ib
    fine_cache = {}
    for ib in set(v for v in peaks.values() if v is not None):
        fine_cache[ib] = score(PROFILE_COARSE[ib] + fine_angles)
    for j in range(len(boxes)):
        ib = peaks[j]
        if ib is None:
            results.append((0.0, 0.0))
            continue
        fs = fine_cache[ib][:, j]
        jb = int(np.argmax(fs))
        best = float(PROFILE_COARSE[ib] + fine_angles[jb])
        if 0 < jb < len(fs) - 1:
            y0_, y1_, y2_ = fs[jb - 1], fs[jb], fs[jb + 1]
            den = y0_ - 2 * y1_ + y2_
            if den < 0:
                best += PROFILE_FINE_STEP * 0.5 * (y0_ - y2_) / den
        med = float(np.median(coarse[:, j]))
        results.append((best, float(fs[jb] / med) if med > 0 else 0.0))
    return results


def _basis(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(xn), xn, yn, xn ** 2, xn * yn], axis=1)


def _fit_tilt(meas: np.ndarray, W: int, H: int) -> np.ndarray:
    """Robust weighted least squares of the tilt polynomial (IRLS, Cauchy)."""
    xn = (meas[:, 0] - W / 2) / (W / 2)
    yn = (meas[:, 1] - H / 2) / (W / 2)
    A = _basis(xn, yn)
    t = meas[:, 2]
    w = meas[:, 3].copy()
    c = np.zeros(A.shape[1])
    for _ in range(IRLS_ITERS):
        c = np.linalg.solve(A.T @ (A * w[:, None]) + RIDGE * np.eye(A.shape[1]), A.T @ (w * t))
        r = t - A @ c
        w = meas[:, 3] / (1.0 + (r / IRLS_SCALE_DEG) ** 2)
    return c


def displacement(c: np.ndarray, W: int, H: int) -> np.ndarray:
    """Vertical displacement map v(x, y) in pixels, shape (H, W): the
    x-integral of the tilt polynomial, zero on the centre column."""
    # the map is a smooth cubic: evaluate it on a coarse grid and resize
    gw, gh = max(2, W // 8), max(2, H // 8)
    xs = ((np.arange(gw) * (W - 1) / (gw - 1)) - W / 2) / (W / 2)
    ys = ((np.arange(gh) * (H - 1) / (gh - 1)) - H / 2) / (W / 2)
    X, Y = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    c0, c1, c2, c3, c4 = np.radians(c)          # tilt in degrees -> slope (small angles)
    v = c0 * X + c1 * X ** 2 / 2 + c2 * X * Y + c3 * X ** 3 / 3 + c4 * X ** 2 * Y / 2
    return cv2.resize((v * (W / 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


def dewarp_by_lines(gray: np.ndarray, inset: int = 0,
                    min_disp_px: float = MIN_DISP_PX) -> Tuple[Optional[np.ndarray], dict]:
    """Displacement map (H, W) float32 that unbends the page (y_src = y + v),
    or None with the reason. ``info`` carries the evidence and the residual
    tilt before/after (weighted median |deg| over the profile cells)."""
    H, W = gray.shape[:2]
    meas = measure_cells(gray, inset)
    cells = meas[meas[:, 4] == 1] if len(meas) else meas
    info = {'n_seg': int((meas[:, 4] == 0).sum()) if len(meas) else 0, 'n_cell': int(len(cells)),
            'applied': False}
    if len(cells) < MIN_CELLS:
        info['reason'] = 'too few cells'
        return None, info
    info['before'] = round(_wmedian(np.abs(cells[:, 2]), cells[:, 3]), 2)
    c = _fit_tilt(meas, W, H)
    info['coef'] = [round(float(v), 3) for v in c]
    v = displacement(c, W, H)
    vmax = float(np.abs(v).max())
    info['max_disp_px'] = round(vmax, 1)
    if vmax < min_disp_px:
        info['reason'] = 'flat enough'
        return None, info
    if vmax > MAX_DISP_FRAC * H:
        info['reason'] = 'bend too large'
        return None, info
    # residual: the tilt each measurement keeps after the map (tilt minus the
    # model's slope there). Judged on ALL the evidence, segments included: a
    # map that satisfies the cells but tilts the long segments is a wrong map
    # (a wrinkled page fooled the cells once: 33 px map, page tilted by 1 deg)
    xn = (meas[:, 0] - W / 2) / (W / 2)
    yn = (meas[:, 1] - H / 2) / (W / 2)
    after_all = np.abs(meas[:, 2] - _basis(xn, yn) @ c)
    before_all = np.abs(meas[:, 2])
    info['after'] = round(_wmedian(after_all[meas[:, 4] == 1], cells[:, 3]), 2)
    info['before_all'] = round(_wmedian(before_all, meas[:, 3]), 2)
    info['after_all'] = round(_wmedian(after_all, meas[:, 3]), 2)
    if info['after'] > (1.0 - MIN_GAIN) * info['before'] or info['after_all'] > info['before_all']:
        info['reason'] = 'no gain'
        return None, info
    info['applied'] = True
    return v, info


def apply_dewarp(page: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = page.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(page, xs, ys + v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
