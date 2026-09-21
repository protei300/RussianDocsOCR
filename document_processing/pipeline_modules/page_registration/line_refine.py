"""Blur-tolerant refinement of a rectified page by its own straight structures.

After the page is warped into its frame (from a Borders quad or a template
homography) every horizontal structure of a flat page - field rules, text
baselines, the MRZ, the red band - must be exactly horizontal and every
vertical one (photo frame, series/number column, page edges) exactly
vertical. Whatever residual tilt is left tells how the warp is off:

- the same tilt everywhere      -> rotation (or shear, if verticals disagree);
- tilt changing with y          -> horizontal lines converge: perspective p1;
- vertical tilt changing with x -> vertical lines converge: perspective p2.

Evidence comes from two sources so that it survives blur, which kills keypoint
matching first: LSD line segments (plenty on a sharp page, few on a blurred
one) and projection-profile tilts of horizontal bands and vertical strips
(text lines and rules make a sharp profile even when badly blurred; measured
on frames where SIFT found 2-5 inliers). A 4-parameter model (rotation, shear,
two perspective terms) is fitted to the angles with a robust loss; parameters
without evidence are pinned to zero by a weak prior. The correction is applied
only if it is small and actually reduces the residual - otherwise the page is
left alone. No template, no weights, deterministic.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

LSD_MIN_LEN_FRAC = 0.04       # segment length, fraction of page width
LSD_MAX_WEIGHT = 3.0          # a segment weighs len / (0.1 * W), capped
ANGLE_TOL_DEG = 12.0          # |angle| below this is 'horizontal', near 90 'vertical'
PROFILE_SCALE = 0.5
PROFILE_COARSE = np.arange(-6.0, 6.01, 1.0)
PROFILE_FINE_STEP = 0.25
PROFILE_MIN_PEAK = 1.15       # best/median score ratio for a band's tilt to count
PROFILE_WEIGHT = 3.0
MIN_HORIZ_WEIGHT = 4.0        # total weight of horizontal evidence needed to act
MAX_ROT_DEG = 5.0
MAX_CORNER_SHIFT_FRAC = 0.08  # of the page width
MIN_GAIN = 0.25               # weighted median |residual| must drop by this fraction
MIN_RESIDUAL_DEG = 0.3        # ... and be worth correcting to begin with
PRIOR_SCALE = (0.03, 0.05, 0.03)  # shear / p1 / p2 of this size cost one degree of residual:
                              # shear and p2 need vertical evidence, which is scarce, so
                              # they are held harder than p1
IRLS_ITERS = 3
IRLS_SCALE_DEG = 1.0          # Cauchy scale of the measurement weights
VERT_MIN_LEN_FRAC = 0.15      # a 'vertical' segment must be this long (of the page height):
                              # page edges, the photo frame, the ornament band - not hair,
                              # not a stroke of a letter (short verticals set p2 to junk)
BLOB_MAX_FRAC = 0.2           # ink components taller (bands) / wider (strips) than this
                              # fraction of the region are not text: dropped from the profile
N_BANDS = 5                   # overlapping bands / strips, each a third of the page

_LSD = None


def _lsd():
    global _LSD
    if _LSD is None:
        _LSD = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    return _LSD


def _segments(gray: np.ndarray, min_len: float) -> np.ndarray:
    lines = _lsd().detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4), np.float64)
    lines = lines.reshape(-1, 4).astype(np.float64)
    L = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return lines[L >= min_len]


def _seg_angle(segs: np.ndarray) -> np.ndarray:
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    return (ang + 90.0) % 180.0 - 90.0          # (-90, 90]


def _profile_tilt(region: np.ndarray, axis: int) -> Tuple[float, float]:
    """Tilt (deg) that makes the projection profile along ``axis`` sharpest,
    and the peak ratio (best score / median score). axis=1: row sums, i.e.
    horizontal text lines; axis=0: column sums, vertical structures."""
    small = region
    if PROFILE_SCALE != 1.0:
        small = cv2.resize(region, None, fx=PROFILE_SCALE, fy=PROFILE_SCALE, interpolation=cv2.INTER_AREA)
    # ink = 1, paper = 0 (Otsu, as the deskewer does) so the empty corners a
    # rotation leaves look like paper; the profile is still normalised by the
    # number of valid pixels per row so a cut corner cannot tilt the score
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # keep text-sized ink only: a photo, a stamp, a thumb or the dark table
    # beyond a page edge is one big blob whose outline - not the text lines -
    # would otherwise own the profile (measured: bands disagreeing in sign)
    h, w = binary.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        big = (st[:, cv2.CC_STAT_HEIGHT] > BLOB_MAX_FRAC * h) if axis == 1 \
            else (st[:, cv2.CC_STAT_WIDTH] > BLOB_MAX_FRAC * w)
        big[0] = False
        if big.any():
            binary[big[lab]] = 0
    binary = binary.astype(np.float32) / 255.0
    valid = np.ones_like(binary)
    centre = (w / 2.0, h / 2.0)

    def score(angles):
        out = np.empty(len(angles))
        for i, a in enumerate(angles):
            M = cv2.getRotationMatrix2D(centre, float(a), 1.0)
            rot = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            cnt = cv2.warpAffine(valid, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0).sum(axis=axis)
            ink = rot.sum(axis=axis)
            ok = cnt >= 0.6 * cnt.max()
            prof = ink[ok] / np.maximum(cnt[ok], 1.0)
            out[i] = float(prof.var()) if len(prof) > 2 else 0.0
        return out

    coarse = score(PROFILE_COARSE)
    ib = int(np.argmax(coarse))
    if ib == 0 or ib == len(PROFILE_COARSE) - 1 or coarse[ib] <= 0:
        return 0.0, 0.0          # no peak inside the range: no evidence
    fine = np.arange(PROFILE_COARSE[ib] - 1.0, PROFILE_COARSE[ib] + 1.001, PROFILE_FINE_STEP)
    fs = score(fine)
    jb = int(np.argmax(fs))
    best = float(fine[jb])
    if 0 < jb < len(fine) - 1:                   # parabolic interpolation
        y0, y1, y2 = fs[jb - 1], fs[jb], fs[jb + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            best += PROFILE_FINE_STEP * 0.5 * (y0 - y2) / den
    med = float(np.median(coarse))
    ratio = float(fs[jb] / med) if med > 0 else 0.0
    # cv2 rotates counter-clockwise for a positive angle, which straightens a
    # line running down to the right, i.e. one with a POSITIVE image angle
    # (atan2 with y down) - so the best angle is the tilt of the structure
    # itself, in the same convention as the LSD segments (pinned by a test)
    return best, ratio


def measure(gray: np.ndarray, inset: int = 0) -> np.ndarray:
    """Straightness evidence of a rectified page.

    Returns rows (x1, y1, x2, y2, kind, weight, source): kind 0 = should be
    horizontal, 1 = should be vertical; source 0 = LSD segment, 1 = band
    profile, 2 = strip profile. ``inset`` excludes the cushion around the page
    from the profile regions (segments are taken everywhere: a page edge
    visible in the cushion is evidence too)."""
    H, W = gray.shape[:2]
    rows: List[list] = []
    segs = _segments(gray, LSD_MIN_LEN_FRAC * W)
    if len(segs):
        ang = _seg_angle(segs)
        L = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        wgt = np.minimum(L / (0.1 * W), LSD_MAX_WEIGHT)
        for s, a, w, l in zip(segs, ang, wgt, L):
            if abs(a) < ANGLE_TOL_DEG:
                rows.append([*s, 0, w, 0])
            elif abs(abs(a) - 90.0) < ANGLE_TOL_DEG and l >= VERT_MIN_LEN_FRAC * H:
                rows.append([*s, 1, w, 0])
    x0, y0, x1, y1 = inset, inset, W - inset, H - inset
    if x1 - x0 < 60 or y1 - y0 < 60:
        return np.array(rows, np.float64).reshape(-1, 7)
    # overlapping horizontal bands: tilt of the text lines at several heights
    bh = (y1 - y0) / 3.0
    for yb in np.linspace(y0, y1 - bh, N_BANDS):
        band = gray[int(yb):int(yb + bh), x0:x1]
        tilt, ratio = _profile_tilt(band, axis=1)
        if ratio >= PROFILE_MIN_PEAK:
            yc = yb + 0.5 * bh
            half = 0.5 * (x1 - x0)
            dy = np.tan(np.radians(tilt)) * half
            rows.append([x0, yc - dy, x1, yc + dy, 0, PROFILE_WEIGHT * min(1.0, ratio - 1.0), 1])
    # vertical strips (column profiles) were tried and dropped: a passport page
    # has almost no vertical text structure, and the strip tilts came out as
    # junk (measured -5 deg on straight pages) that only fed p2 and shear
    return np.array(rows, np.float64).reshape(-1, 7)


def _model(params, W, H) -> np.ndarray:
    """Pixel homography for (rotation rad, shear, p1, p2) about the page centre,
    perspective terms in units of the half page width."""
    th, s, p1, p2 = params
    k = 2.0 / W
    T = np.array([[k, 0, -1.0], [0, k, -H / W], [0, 0, 1.0]])
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    S = np.array([[1.0, s, 0], [0, 1.0, 0], [0, 0, 1.0]])
    P = np.array([[1.0, 0, 0], [0, 1.0, 0], [p1, p2, 1.0]])
    return np.linalg.inv(T) @ P @ R @ S @ T


def _angles_after(Hm, meas) -> np.ndarray:
    p0 = cv2.perspectiveTransform(meas[:, 0:2].reshape(1, -1, 2), Hm).reshape(-1, 2)
    p1 = cv2.perspectiveTransform(meas[:, 2:4].reshape(1, -1, 2), Hm).reshape(-1, 2)
    ang = np.degrees(np.arctan2(p1[:, 1] - p0[:, 1], p1[:, 0] - p0[:, 0]))
    ang = (ang + 90.0) % 180.0 - 90.0
    dev = np.where(meas[:, 4] == 0, ang, np.abs(ang) - 90.0)
    return dev


def _residuals(params, meas, W, H, wgt):
    dev = _angles_after(_model(params, W, H), meas) * np.sqrt(wgt)
    prior = np.array(params[1:]) / np.array(PRIOR_SCALE)
    return np.concatenate([dev, prior])


def _fit(meas, W, H):
    """Robust fit of the 4 parameters: iteratively reweighted least squares
    with a Cauchy weight on every measurement (an outlying band or a segment
    on a hair line stops mattering) while the prior on the parameters stays
    quadratic - a plain robust loss would discount a large prior residual
    exactly like an outlier and let a single junk vertical segment set p2."""
    x = np.zeros(4)
    wgt = meas[:, 5].copy()
    for _ in range(IRLS_ITERS):
        sol = least_squares(_residuals, x, args=(meas, W, H, wgt), max_nfev=100)
        x = sol.x
        r = _angles_after(_model(x, W, H), meas)
        wgt = meas[:, 5] / (1.0 + (r / IRLS_SCALE_DEG) ** 2)
    return x


def _wmedian(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    v, w = values[order], weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def refine_by_lines(gray: np.ndarray, inset: int = 0,
                    min_residual_deg: float = MIN_RESIDUAL_DEG) -> Tuple[Optional[np.ndarray], dict]:
    """Homography (pixel, src->dst for cv2.warpPerspective) that straightens a
    rectified page by its own lines, or None when there is not enough evidence
    or the correction is not warranted. ``info`` carries the evidence counts,
    the residual before/after (weighted median |deg|) and the parameters."""
    H, W = gray.shape[:2]
    meas = measure(gray, inset)
    info = {'n_seg': int((meas[:, 6] == 0).sum()) if len(meas) else 0,
            'n_band': int((meas[:, 6] == 1).sum()) if len(meas) else 0,
            'n_strip': int((meas[:, 6] == 2).sum()) if len(meas) else 0,
            'applied': False}
    if len(meas) == 0:
        info['reason'] = 'no evidence'
        return None, info
    horiz_w = float(meas[meas[:, 4] == 0, 5].sum())
    info['horiz_weight'] = round(horiz_w, 2)
    if horiz_w < MIN_HORIZ_WEIGHT:
        info['reason'] = 'too little horizontal evidence'
        return None, info
    before = np.abs(_angles_after(np.eye(3), meas))
    info['before'] = round(_wmedian(before, meas[:, 5]), 2)
    x = _fit(meas, W, H)
    Hm = _model(x, W, H)
    after = np.abs(_angles_after(Hm, meas))
    info['after'] = round(_wmedian(after, meas[:, 5]), 2)
    info['params'] = {'rot_deg': round(float(np.degrees(x[0])), 2), 'shear': round(float(x[1]), 4),
                      'p1': round(float(x[2]), 4), 'p2': round(float(x[3]), 4)}
    if info['before'] < min_residual_deg:
        info['reason'] = 'already straight'
        return None, info
    if info['after'] > (1.0 - MIN_GAIN) * info['before']:
        info['reason'] = 'no gain'
        return None, info
    if abs(info['params']['rot_deg']) > MAX_ROT_DEG:
        info['reason'] = 'rotation too large'
        return None, info
    corners = np.float64([[inset, inset], [W - inset, inset], [W - inset, H - inset], [inset, H - inset]])
    moved = cv2.perspectiveTransform(corners.reshape(1, -1, 2), Hm).reshape(-1, 2)
    shift = float(np.linalg.norm(moved - corners, axis=1).max())
    info['corner_shift_px'] = round(shift, 1)
    if shift > MAX_CORNER_SHIFT_FRAC * W:
        info['reason'] = 'correction too large'
        return None, info
    info['applied'] = True
    return Hm, info


def apply_refinement(page: np.ndarray, Hm: np.ndarray) -> np.ndarray:
    h, w = page.shape[:2]
    return cv2.warpPerspective(page, Hm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
