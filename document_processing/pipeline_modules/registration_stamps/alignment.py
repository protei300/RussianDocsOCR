"""Residual stamp tilt, with a map from the expanded output to its input.

The stamp detector already removes the coarse OBB angle. Ruling lines refine
that angle; a projection profile handles stamps without reliable ruling lines.
This estimates small residual tilt, not the reading direction of upside-down text.
"""
import math

import cv2
import numpy as np

from ..deskewer import DocDeskewer


def _weighted_median(values, weights):
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum()/2)])


def estimate_stamp_angle(image: np.ndarray) -> float:
    """OpenCV counterclockwise correction, in degrees; zero when unreliable.

    Accepts BGR or grayscale. Reliable long ruling lines take priority over
    text profiles, which can otherwise straighten handwriting against its form.
    """
    h, w = image.shape[:2]
    if min(h, w) < 16:
        return 0.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if int(gray.max()) - int(gray.min()) < 8:
        return 0.
    edges = cv2.Canny(gray, 50, 130)
    lines = cv2.HoughLinesP(edges, 1, np.pi/1800, threshold=max(30, int(w*.1)),
                            minLineLength=max(50, int(w*.3)), maxLineGap=12)
    angles, lengths = [], []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = (math.degrees(math.atan2(float(y2-y1), float(x2-x1)))+90) % 180-90
            if abs(angle) <= 8:
                angles.append(angle); lengths.append(math.hypot(float(x2-x1), float(y2-y1)))
    if angles:
        angle = _weighted_median(angles, lengths)
        mad = _weighted_median(np.abs(np.asarray(angles)-angle), lengths)
        support = sum(l for a, l in zip(angles, lengths) if abs(a-angle) < 1)/sum(lengths)
        if mad < .8 and support >= .6 and (len(angles) >= 2 or max(lengths) > .65*w):
            return angle if abs(angle) >= .5 else 0.
    # Sparse noise and dark/blank patches do not provide a text-line profile.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    share = np.count_nonzero(binary)/binary.size
    if not .005 <= share <= .6:
        return 0.
    # Reuse the document deskewer's bounded coarse-to-fine search. Grayscale
    # avoids its RGB input convention; no rotation/resampling happens there.
    angle = DocDeskewer(angle_range=6, angle_steps=121, min_angle=.5, scale=.8)._find_angle(gray)
    return angle if abs(angle) >= .5 else 0.


def align_stamp(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (upright image, output-to-input homography, correction degrees).

    No-op returns the original array. Rotation keeps the entire input footprint
    on an expanded canvas; callers must dispose of the new array after reading.
    """
    angle = estimate_stamp_angle(image)
    if not angle:
        return image, np.eye(3), 0.
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1.)
    corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], float)
    projected = np.c_[corners, np.ones(4)] @ matrix.T
    lo = np.floor(projected.min(0))-2
    hi = np.ceil(projected.max(0))+2
    matrix[:, 2] -= lo
    size = tuple((hi-lo+1).astype(int))
    aligned = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return aligned, np.linalg.inv(np.vstack([matrix, [0, 0, 1.]])), angle
